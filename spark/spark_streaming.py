from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, avg, expr, max as spark_max, udf,  round as spark_round
from pyspark.sql.types import StructField, StructType, StringType, FloatType, IntegerType, DoubleType
import os
import psycopg2
from datetime import datetime

# Function to convert time string to minutes
def convert_time_to_minutes(time_str):
    try:
        time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
    except ValueError:
        try:
            time_obj = datetime.strptime(time_str, '%H:%M:%S')
        except ValueError:
            return None
    total_minutes = time_obj.hour * 60 + time_obj.minute + time_obj.second / 60 + time_obj.microsecond / (60 * 1e6)
    return total_minutes

# Register the UDF with Spark
convert_time_to_minutes_udf = udf(convert_time_to_minutes, DoubleType())

# Initialize Spark Session
spark = SparkSession \
    .builder \
    .appName("KafkaSparkStreaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1") \
    .getOrCreate()

# Subscribe to Kafka topic
df = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", "kafka:9092") \
  .option("subscribePattern", ".*-topic") \
  .option("kafka.metadata.max.age.ms", "3000") \
  .load()

# Define schema for incoming data
schema = StructType([
    StructField("username", StringType(), True),
    StructField("Udaljenost", FloatType(), True),
    StructField("Vrijeme", StringType(), True),
    StructField("Prosječni puls", FloatType(), True),
    StructField("Ukupni uspon", FloatType(), True)
])

# Parse JSON and apply schema
df = df.selectExpr("CAST(value AS STRING) as json")
df = df.select(from_json(col("json"), schema).alias("data"))
df = df.select("data.*")

# Convert "Vrijeme" to minutes
df = df.withColumn("Vrijeme", convert_time_to_minutes_udf(col("Vrijeme")))

# Perform analytics
aggregated_df = df.groupBy("username") \
    .agg(
        spark_round(avg("Udaljenost"), 2).alias("avg_distance"),
        spark_round(avg("Vrijeme"), 2).alias("avg_time"),
        spark_round(avg("Prosječni puls"), 2).alias("avg_heart_rate"),
        spark_round(avg("Ukupni uspon"), 2).alias("avg_ascent"),
        spark_round(spark_max("Udaljenost"), 2).alias("max_distance"),
        spark_round(spark_max("Vrijeme"), 2).alias("max_time"),
        spark_round(spark_max("Ukupni uspon"), 2).alias("max_ascent")
    )

# Write analytics results to PostgreSQL
def write_to_postgres(batch_df, batch_id):
    postgres_url = f"jdbc:postgresql://{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    user = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_PASSWORD')
    
    # Establish connection to PostgreSQL
    conn = psycopg2.connect(
        dbname=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        host=os.getenv('POSTGRES_HOST'),
        port=os.getenv('POSTGRES_PORT')
    )
    cursor = conn.cursor()

    # Get unique usernames in this batch
    usernames = [row.username for row in batch_df.select("username").distinct().collect()]

    for username in usernames:
        # Delete existing analytics for this user
        cursor.execute("DELETE FROM user_analytics WHERE username = %s", (username,))
    
    # Commit the transaction and close the cursor
    conn.commit()
    cursor.close()
    conn.close()

    # Write new analytics
    batch_df.write \
        .format("jdbc") \
        .option("url", postgres_url) \
        .option("dbtable", "user_analytics") \
        .option("user", user) \
        .option("password", password) \
        .option("driver", "org.postgresql.Driver") \
        .mode("append") \
        .save()

print(aggregated_df)
query = aggregated_df \
    .writeStream \
    .outputMode("complete") \
    .foreachBatch(write_to_postgres) \
    .start()

query.awaitTermination()
