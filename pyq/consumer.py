from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, lit, length
from pyspark.sql.types import StructType, StructField, StringType
from datetime import datetime
import pandas as pd
from google.cloud import storage
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, TimestampType, StringType
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Tokenizer, VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import from_csv


logs = []

schema = StructType([
    StructField("random1", StringType(), True),        
    StructField("random2", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("random4", StringType(), True),
    StructField("random5", StringType(), True),
    StructField("label", DoubleType(), False),
    StructField("reviews", StringType(), True),
    StructField("random8", StringType(), True),
    StructField("random9", StringType(), True),
])

def ml_task(df):
    try:
        # Convert timestamp to numeric (e.g., epoch time)
        df = df.withColumn("timestamp_numeric", F.col("timestamp").cast("long"))

        # Fill missing values with a constant value
        df = df.na.fill({"timestamp_numeric": 0, "label": 0.0, "reviews": ''})

        # Drop rows where all columns are null
        df = df.dropna(how="all")

        # Load the pipeline model
        pipeline_model_path = "gs://pyq/pipeline_model"
        loaded_pipeline_model = PipelineModel.load(pipeline_model_path)

        transformed_data = loaded_pipeline_model.transform(df)

        # Assemble features
        assembler = VectorAssembler(inputCols=["onehot_features", "timestamp_numeric"], outputCol="features")
        assembled_df = assembler.transform(transformed_data)
        
        # Load Logistic Regression model
        model_path = "gs://pyq/lrmodel"
        lr_model = LogisticRegressionModel.load(model_path)
        
        # Make predictions
        prediction = lr_model.transform(assembled_df)
        print("\n\n"*10) 
        prediction.show()
        print("\n\n"*10) 
    except Exception as e:
        print(f"Error occurred in ml_task: {e}")

                  
                 

def process_batch(batch_df, batch_id):
    print("Table content for Batch ID:", batch_id)
    batch_df.show(5, truncate=False)
    ml_task(batch_df)
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


def write_to_gcs(bucket_name, file_name, df):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(df.to_csv(index=False), "text/csv")

spark = SparkSession.builder \
    .appName("Yelp Review Classification") \
    .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.0") \
    .config("spark.driver.memory", "6g") \
    .getOrCreate()

### Read data from Kafka using readStream
kafka_topic = "quickstart-events"
kafka_bootstrap_servers = "34.131.174.82:9092"
df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers)
    .option("subscribe", kafka_topic)
    .load()
)


# Define the schema as a string
schema_string = "random1 STRING, random2 STRING, timestamp TIMESTAMP, random4 STRING, random5 STRING, label DOUBLE, reviews STRING, random8 STRING, random9 STRING"


df = df.selectExpr("CAST(value AS STRING)")
df = df.select(from_csv(col("value"), schema_string).alias("data")).select("data.*")

### Add a new column with the first 20 characters of the paragraph
#df = df.withColumn("paragraph_start", col("text").substr(1, 20))

### Display the streaming data with additional data in a table format
query = df.writeStream.outputMode("append").foreachBatch(process_batch).start()

query.awaitTermination()