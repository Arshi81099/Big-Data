from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, lit, length
from pyspark.sql.types import StructType, StructField, StringType
from datetime import datetime
import pandas as pd
from google.cloud import storage

logs = []

def process_batch(batch_df, batch_id):
    print("Table content for Batch ID:", batch_id)
    batch_df.show(truncate=False)
    df = batch_df.toPandas()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Number of records read in 10 secs from batch {batch_id}: {df.shape[0]}\n"
    print("*" * 100)
    print(log_entry)
    print("*" * 100)
    log = {
        "timestamp": timestamp,
        "entry": f"Number of records read in 10 secs from batch {batch_id}: {df.shape[0]}",
    }
    logs.append(log)

    ### Write logs to GCS
    bucket_name = "pyq"
    log_file_name = "log_file.csv"
    write_to_gcs(bucket_name, log_file_name, pd.DataFrame(logs))

def write_to_gcs(bucket_name, file_name, df):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(df.to_csv(index=False), "text/csv")

spark = SparkSession.builder.appName("StreamReader").getOrCreate()

# Updated schema to handle the paragraph data
schema = StructType([StructField("text", StringType())])

### Read data from Kafka using readStream
kafka_topic = "quickstart-events"
kafka_bootstrap_servers = "34.126.220.137:9092"
df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
    .option("subscribe", kafka_topic)
    .load()
)

### Convert the value column from Kafka to a JSON structure
df = (
    df.selectExpr("CAST(value AS STRING)")
    .select(from_json("value", schema).alias("data"))
    .select("data.*")
)

### Add a new column with the first 20 characters of the paragraph
df = df.withColumn("paragraph_start", col("text").substr(1, 20))

### Display the streaming data with additional data in a table format
query = df.writeStream.outputMode("append").foreachBatch(process_batch).start()

query.awaitTermination()
