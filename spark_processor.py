"""Reads weather readings from Kafka with Spark Structured Streaming.

Two streaming queries:
  1. raw  -> MongoDB weather_db.raw_readings   (every record, for History tab)
  2. agg  -> MongoDB weather_db.aggregates     (5-min window stats per city)
"""
import os
import sys
from datetime import timezone

# --- Windows / local-mode fixes (must run before Spark starts) --------------
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
_java17 = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
if os.path.isdir(_java17):
    os.environ["JAVA_HOME"] = _java17
    os.environ["PATH"] = _java17 + r"\bin;" + os.environ["PATH"]
if os.path.isdir(r"C:\hadoop\bin"):
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]
# -----------------------------------------------------------------------------

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (DoubleType, IntegerType, StringType,
                               StructField, StructType)

load_dotenv()
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "weather_raw")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB = os.getenv("MONGO_DB", "weather_db")
CHECKPOINT = os.getenv("CHECKPOINT_DIR", "./checkpoints")

SCHEMA = StructType([
    StructField("city", StringType()),
    StructField("country", StringType()),
    StructField("lat", DoubleType()),
    StructField("lon", DoubleType()),
    StructField("temp_c", DoubleType()),
    StructField("humidity", IntegerType()),
    StructField("wind_kph", DoubleType()),
    StructField("pressure_mb", DoubleType()),
    StructField("condition", StringType()),
    StructField("ingested_at", StringType()),
])

spark = (
    SparkSession.builder.appName("WeatherStreamProcessor")
    .master("local[*]")
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.executor.heartbeatInterval", "60s")
    .config("spark.network.timeout", "300s")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ---- Source: Kafka ---------------------------------------------------------
parsed = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", BOOTSTRAP)
    .option("subscribe", TOPIC)
    .option("startingOffsets", "latest")
    .load()
    .select(F.from_json(F.col("value").cast("string"), SCHEMA).alias("d"))
    .select("d.*")
    .withColumn("event_time", F.to_timestamp("ingested_at"))
)


# ---- Sinks -----------------------------------------------------------------
def write_raw(batch_df, batch_id):
    rows = [r.asDict() for r in batch_df.collect()]
    if not rows:
        return
    for row in rows:  # Spark hands back naive local time; store true UTC
        if row.get("event_time") is not None:
            row["event_time"] = row["event_time"].astimezone(timezone.utc)
    coll = MongoClient(MONGO_URI)[DB]["raw_readings"]
    coll.insert_many(rows)
    print(f"[raw] batch {batch_id}: {len(rows)} records")


def write_agg(batch_df, batch_id):
    rows = batch_df.collect()
    if not rows:
        return
    coll = MongoClient(MONGO_URI)[DB]["aggregates"]
    ops = []
    for r in rows:
        start = r["window"]["start"].astimezone(timezone.utc)
        end = r["window"]["end"].astimezone(timezone.utc)
        key = {"city": r["city"], "window_start": start}
        doc = {
            **key,
            "window_end": end,
            "avg_temp": round(r["avg_temp"], 2),
            "min_temp": r["min_temp"],
            "max_temp": r["max_temp"],
            "avg_humidity": round(r["avg_humidity"], 2),
            "avg_wind": round(r["avg_wind"], 2),
            "count": r["count"],
        }
        ops.append(UpdateOne(key, {"$set": doc}, upsert=True))
    coll.bulk_write(ops)
    print(f"[agg] batch {batch_id}: {len(ops)} windows upserted")


# ---- Query 1: raw records --------------------------------------------------
q_raw = (
    parsed.writeStream.foreachBatch(write_raw)
    .option("checkpointLocation", os.path.join(CHECKPOINT, "raw"))
    .trigger(processingTime="15 seconds")
    .start()
)

# ---- Query 2: 5-minute windowed aggregation --------------------------------
agg = (
    parsed.withWatermark("event_time", "10 minutes")
    .groupBy(F.window("event_time", "5 minutes"), "city")
    .agg(
        F.avg("temp_c").alias("avg_temp"),
        F.min("temp_c").alias("min_temp"),
        F.max("temp_c").alias("max_temp"),
        F.avg("humidity").alias("avg_humidity"),
        F.avg("wind_kph").alias("avg_wind"),
        F.count("*").alias("count"),
    )
)

q_agg = (
    agg.writeStream.outputMode("update")
    .foreachBatch(write_agg)
    .option("checkpointLocation", os.path.join(CHECKPOINT, "agg"))
    .trigger(processingTime="15 seconds")
    .start()
)

print("Streaming started. Press Ctrl+C to stop.")
spark.streams.awaitAnyTermination()
