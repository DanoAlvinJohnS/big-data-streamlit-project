"""Statistical models behind the Station Atlas dashboard (numpy + pandas only).

Per station
  1. Trend model      temperature ~ time                      (+ a 24 h cycle once
                      a full day of observations exists)
  2. Forecast         next 12 hours with a 95% prediction interval
  3. Anomaly score    studentized residual of the latest observation
  4. Driver model     temperature ~ humidity + wind + pressure (multiple regression)
Across stations
  5. Pooled fit       temperature ~ humidity over all stations
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MEASURES = ["temp_c", "humidity", "wind_kph", "pressure_mb"]
MIN_OBS = 8                 # fewest observations before a trend model is fitted
DRIVER_MIN_OBS = 12         # fewest observations before the driver model is fitted
HARMONIC_MIN_OBS = 24       # daily-cycle terms need at least a day of data
HARMONIC_MIN_SPAN_H = 24.0
HORIZON_H = 12
Z_WATCH = 1.5
Z_ANOMALY = 2.0


# ------------------------------------------------------------- utilities ----
def distinct_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated polls: keep a row only when a measurement changed.

    The producer polls every 15 s but WeatherAPI refreshes roughly every
    10-15 min, so most polls repeat the previous values. Regressing on the
    repeats would overstate the sample size.
    """
    if df.empty:
        return df.copy()
    cols = MEASURES + ["condition"]
    changed = df[cols].ne(df[cols].shift()).any(axis=1)
    return df[changed].reset_index(drop=True)


def to_hours(times) -> np.ndarray:
    t = pd.to_datetime(times, utc=True)
    return ((t - pd.Timestamp("1970-01-01", tz="UTC"))
            .dt.total_seconds().to_numpy(dtype=float) / 3600.0)


def t_crit(dof: float, z: float = 1.959964) -> float:
    """Two-sided 95% Student-t critical value (Cornish-Fisher expansion)."""
    if dof is None or dof <= 0:
        return float("nan")
    g1 = (z**3 + z) / 4.0
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96.0
    return z + g1 / dof + g2 / dof**2


def design(h: np.ndarray, h0: float, harmonic: bool) -> np.ndarray:
    cols = [np.ones_like(h), h - h0]
    if harmonic:
        w = 2 * np.pi / 24.0
        cols += [np.sin(w * h), np.cos(w * h)]
    return np.column_stack(cols)


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    n = X.shape[0]
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    fitted = X @ beta
    resid = y - fitted
    rank = int(np.linalg.matrix_rank(X))
    dof = n - rank
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    sigma2 = rss / dof if dof > 0 else float("nan")
    r2 = 1 - rss / tss if tss > 1e-12 else float("nan")
    adj = (1 - (1 - r2) * (n - 1) / dof) if (dof > 0 and r2 == r2) else float("nan")
    hat = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    denom = np.sqrt(np.clip(sigma2 * (1 - hat), 1e-12, None)) if sigma2 == sigma2 else None
    stud = resid / denom if denom is not None else np.full(n, np.nan)
    if sigma2 == sigma2 and sigma2 < 1e-12:
        stud = np.zeros(n)
    return {
        "beta": beta, "fitted": fitted, "resid": resid, "stud": stud,
        "xtx_inv": xtx_inv, "dof": dof, "rank": rank, "sigma2": sigma2,
        "rmse": float(np.sqrt(rss / n)), "r2": r2, "adj_r2": adj, "n": n,
    }


def status_for(z: float | None, n: int) -> tuple[str, str]:
    """(css key, label)"""
    if z is None or z != z:
        return "pending", f"Collecting {min(n, MIN_OBS)}/{MIN_OBS}"
    a = abs(z)
    if a >= Z_ANOMALY:
        return "anomaly", "Anomaly"
    if a >= Z_WATCH:
        return "watch", "Watch"
    return "ok", "Normal"


# ----------------------------------------------------------- station fits ---
def fit_trend(obs: pd.DataFrame) -> dict:
    """Fit temperature against time and forecast HORIZON_H hours ahead."""
    n = len(obs)
    if n < MIN_OBS:
        return {"ok": False, "n": n, "need": MIN_OBS}

    h = to_hours(obs["time"])
    h0 = h[0]
    y = obs["temp_c"].to_numpy(dtype=float)
    harmonic = n >= HARMONIC_MIN_OBS and (h[-1] - h[0]) >= HARMONIC_MIN_SPAN_H

    m = ols(design(h, h0, harmonic), y)
    if m["dof"] <= 0:
        return {"ok": False, "n": n, "need": MIN_OBS}

    steps = np.arange(1, HORIZON_H + 1, dtype=float)
    hf = h[-1] + steps
    Xf = design(hf, h0, harmonic)
    pred = Xf @ m["beta"]
    sigma = float(np.sqrt(m["sigma2"]))
    lever = np.einsum("ij,jk,ik->i", Xf, m["xtx_inv"], Xf)
    half = t_crit(m["dof"]) * sigma * np.sqrt(1.0 + lever)
    future_times = obs["time"].iloc[-1] + pd.to_timedelta(steps, unit="h")

    z = float(m["stud"][-1])
    key, label = status_for(z, n)
    return {
        "ok": True, "n": n, "harmonic": harmonic,
        "times": obs["time"], "y": y, "fitted": m["fitted"],
        "resid": m["resid"], "stud": m["stud"],
        "r2": m["r2"], "adj_r2": m["adj_r2"], "rmse": m["rmse"],
        "sigma": sigma, "dof": m["dof"],
        "slope_per_day": float(m["beta"][1] * 24.0),
        "now": float(y[-1]), "expected": float(m["fitted"][-1]), "z": z,
        "forecast": pd.DataFrame({"time": future_times, "mean": pred,
                                  "lo": pred - half, "hi": pred + half}),
        "f3": float(pred[2]),
        "status_key": key, "status": label,
    }


def fit_drivers(obs: pd.DataFrame) -> dict:
    """Multiple regression: temperature ~ humidity + wind + pressure."""
    labels = {"humidity": "Humidity (%)", "wind_kph": "Wind (kph)",
              "pressure_mb": "Pressure (mb)"}
    n = len(obs)
    usable = [c for c in labels if obs[c].std(ddof=0) > 1e-9]
    if n < DRIVER_MIN_OBS or not usable:
        return {"ok": False, "n": n, "need": DRIVER_MIN_OBS}

    y = obs["temp_c"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n)] + [obs[c].to_numpy(dtype=float) for c in usable])
    m = ols(X, y)
    if m["dof"] <= 0 or m["sigma2"] != m["sigma2"]:
        return {"ok": False, "n": n, "need": DRIVER_MIN_OBS}

    se = np.sqrt(np.clip(np.diag(m["xtx_inv"]) * m["sigma2"], 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, m["beta"] / se, np.nan)
    sd_y = y.std(ddof=0)
    std_beta = [np.nan] + [m["beta"][i + 1] * obs[c].std(ddof=0) / sd_y
                           if sd_y > 0 else np.nan
                           for i, c in enumerate(usable)]
    table = pd.DataFrame({
        "term": ["Intercept"] + [labels[c] for c in usable],
        "coef": m["beta"], "se": se, "t": tstat, "std_beta": std_beta,
    })
    corr = obs[MEASURES].corr()
    return {"ok": True, "n": n, "table": table, "r2": m["r2"],
            "adj_r2": m["adj_r2"], "rmse": m["rmse"], "corr": corr}


def fit_pooled(stations: dict) -> dict:
    """Simple regression of temperature on humidity over all stations."""
    frames = [s["obs"][["humidity", "temp_c"]] for s in stations.values()
              if len(s["obs"])]
    if not frames:
        return {"ok": False}
    d = pd.concat(frames)
    x = d["humidity"].to_numpy(dtype=float)
    y = d["temp_c"].to_numpy(dtype=float)
    if len(d) < MIN_OBS or x.std() < 1e-9:
        return {"ok": False}
    m = ols(np.column_stack([np.ones(len(x)), x]), y)
    r = float(np.corrcoef(x, y)[0, 1])
    xs = np.array([x.min(), x.max()])
    return {"ok": True, "n": len(d), "intercept": float(m["beta"][0]),
            "slope": float(m["beta"][1]), "r": r, "r2": m["r2"],
            "line_x": xs, "line_y": m["beta"][0] + m["beta"][1] * xs}


# ------------------------------------------------------------ whole atlas ---
def build_atlas(raw: pd.DataFrame) -> dict:
    """raw: all readings with columns city, country, time (tz-aware) + measures."""
    stations: dict[str, dict] = {}
    for city, g in raw.groupby("city"):
        g = g.sort_values("time")
        obs = distinct_observations(g)
        stations[city] = {
            "country": g["country"].iloc[-1] if "country" in g else "",
            "polls": len(g), "obs": obs, "trend": fit_trend(obs),
        }

    rows = []
    for city, s in stations.items():
        t = s["trend"]
        obs = s["obs"]
        spark = obs["temp_c"].tail(24).tolist()
        if t["ok"]:
            rows.append({"city": city, "country": s["country"], "n": t["n"],
                         "now": t["now"], "expected": t["expected"], "z": t["z"],
                         "f3": t["f3"], "r2": t["r2"], "status_key": t["status_key"],
                         "status": t["status"], "spark": spark})
        else:
            key, label = status_for(None, t["n"])
            rows.append({"city": city, "country": s["country"], "n": t["n"],
                         "now": float(obs["temp_c"].iloc[-1]) if len(obs) else np.nan,
                         "expected": np.nan, "z": np.nan, "f3": np.nan,
                         "r2": np.nan, "status_key": key, "status": label,
                         "spark": spark})
    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking["abs_z"] = ranking["z"].abs()
        ranking = (ranking.sort_values(["abs_z", "n"], ascending=[False, False],
                                       na_position="last")
                   .drop(columns="abs_z").reset_index(drop=True))
    return {"stations": stations, "ranking": ranking, "pooled": fit_pooled(stations)}
