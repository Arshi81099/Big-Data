from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window
from pyspark.sql.types import StructType, StringType, IntegerType, TimestampType

def process_batch(df, epoch_id):
    # Process the batch for rolling counts
    df.orderBy("window.start", "Station Code").show()
    df.toPandas().to_csv('output_file.csv', sep=",", index=False, header=True)
    print("\nBatch {} completed\n".format(epoch_id))

spark = SparkSession.builder \
    .appName("Train") \
    .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.0") \
    .getOrCreate()

# Read the stream from Kafka
kafka_stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "34.131.9.138:9092") \
    .option("subscribe", 'oppe') \
    .load().selectExpr("CAST(value AS STRING) as json_string")

# Define the schema for the incoming data
json_schema = StructType() \
    .add("Train No", StringType()) \
    .add("Train Name", StringType()) \
    .add("SEQ", IntegerType()) \
    .add("Station Code", StringType()) \
    .add("Station Name", StringType()) \
    .add("Arrival time", TimestampType()) \
    .add("Departure Time", TimestampType()) \
    .add("Distance", IntegerType()) \
    .add("Source Station", StringType()) \
    .add("Source Station Name", StringType()) \
    .add("Destination Station", StringType()) \
    .add("Destination Station Name", StringType())

# Parse the JSON and apply the schema
df = kafka_stream_df.select(from_json("json_string", json_schema).alias("data")).select("data.*")

# Calculate 20-minute rolling count of trains at each station
rolling_counts = df.groupBy(
    window(col("Arrival time"), "20 minutes", "10 seconds"),
    col("Station Code")
).count().alias("train_count")

# Start the query and set trigger interval
query = rolling_counts.writeStream \
    .outputMode("update") \
    .trigger(processingTime='5 seconds') \
    .foreachBatch(process_batch) \
    .start()

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("\nSpark session stopped.\n")
    query.stop()

spark.stop()
