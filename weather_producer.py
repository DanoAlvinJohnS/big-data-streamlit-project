"""Polls WeatherAPI and publishes each reading to a Kafka topic."""
import json, os, time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing WEATHER_API_KEY environment variable. Add it to your .env or environment.")

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "weather_raw")
POLL = int(os.getenv("POLL_SECONDS", "15"))
CITIES = [c.strip() for c in os.getenv("CITIES", "Makati").split(",") if c.strip()]

try:
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
except NoBrokersAvailable as exc:
    raise RuntimeError(
        f"Cannot connect to Kafka at {BOOTSTRAP}. Start the Kafka service and try again."
    ) from exc

def fetch(city):
    r = requests.get(
        "https://api.weatherapi.com/v1/current.json",
        params={"key": API_KEY, "q": city, "aqi": "no"},
        timeout=10,
    )
    r.raise_for_status()
    d = r.json()
    return {
        "city": d["location"]["name"],
        "country": d["location"]["country"],
        "lat": d["location"]["lat"],
        "lon": d["location"]["lon"],
        "temp_c": d["current"]["temp_c"],
        "humidity": d["current"]["humidity"],
        "wind_kph": d["current"]["wind_kph"],
        "pressure_mb": d["current"]["pressure_mb"],
        "condition": d["current"]["condition"]["text"],
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

if __name__ == "__main__":
    print(f"Producing to {TOPIC} every {POLL}s for: {CITIES}")
    while True:
        for city in CITIES:
            try:
                rec = fetch(city)
                producer.send(TOPIC, rec)
                print("sent", rec["city"], rec["temp_c"], "C")
            except Exception as e:
                print("error", city, e)
        producer.flush()
        time.sleep(POLL)    