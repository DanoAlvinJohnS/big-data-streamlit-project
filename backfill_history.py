"""Backfill the last few days of hourly weather into the pipeline.

Publishes historical readings to the same Kafka topic as weather_producer.py, so
they flow through Spark into MongoDB like any other reading. Run it while
spark_processor.py is running. The regression and forecast models in the
dashboard need at least a day of history to show the daily cycle.

    python backfill_history.py          # last 7 days
    python backfill_history.py 3        # last 3 days
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv
from kafka import KafkaProducer

load_dotenv()
API_KEY = os.environ["WEATHER_API_KEY"]
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "weather_raw")
CITIES = [c.strip() for c in os.getenv("CITIES", "Makati").split(",")]
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 7


def fetch_day(city: str, day) -> list[dict]:
    r = requests.get("https://api.weatherapi.com/v1/history.json",
                     params={"key": API_KEY, "q": city, "dt": day.isoformat()}, timeout=15)
    if r.status_code != 200:
        msg = r.json().get("error", {}).get("message", r.text[:120])
        print(f"  skipped {city} {day}: {msg}")
        return []
    d = r.json()
    loc = d["location"]
    out = []
    for h in d["forecast"]["forecastday"][0]["hour"]:
        ts = datetime.fromtimestamp(h["time_epoch"], timezone.utc)
        if ts > datetime.now(timezone.utc):
            continue
        out.append({
            "city": loc["name"], "country": loc["country"], "lat": loc["lat"], "lon": loc["lon"],
            "temp_c": h["temp_c"], "humidity": h["humidity"], "wind_kph": h["wind_kph"],
            "pressure_mb": h["pressure_mb"], "condition": h["condition"]["text"],
            "ingested_at": ts.isoformat(),
        })
    return out


if __name__ == "__main__":
    today = datetime.now(timezone.utc).date()
    records = []
    for city in CITIES:
        for back in range(DAYS, -1, -1):
            day = today - timedelta(days=back)
            got = fetch_day(city, day)
            records += got
            print(f"{city} {day}: {len(got)} hours")
            time.sleep(0.2)
    if not records:
        sys.exit("No history returned. Check your WeatherAPI plan allows history.json.")

    records.sort(key=lambda r: r["ingested_at"])      # oldest first keeps Spark watermarks happy
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP,
                             value_serializer=lambda v: json.dumps(v).encode("utf-8"))
    for rec in records:
        producer.send(TOPIC, rec)
    producer.flush()
    print(f"Published {len(records)} historical readings to {TOPIC}.")
