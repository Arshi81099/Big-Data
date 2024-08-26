from google.cloud import storage
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, TimestampType, StringType


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

spark = SparkSession.builder.\
        master("local[*]").\
        appName("processing").\
        getOrCreate()
# myTable = spark.read.format("csv").schema(schema).load("gs://pyq/training")
myTable = spark.read.format("csv").schema(schema).load("gs://pyq/output.csv")
from pyspark.sql import functions as F

# Create a DataFrame with the count of null values for each column
null_counts = myTable.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in myTable.columns])

# Show the result
null_counts.show()

columns_to_drop = ['random1', 'random2', 'random4', 'random5', 'random8', 'random9', 'random10']
myTable = myTable.drop(*columns_to_drop)

from pyspark.sql import functions as F

# Convert timestamp to numeric (e.g., epoch time)
myTable = myTable.withColumn("timestamp_numeric", F.col("timestamp").cast("long"))

# Perform your operations on the numeric column
# Example: fill missing values with a constant value or mode
myTable = myTable.na.fill({"timestamp_numeric": 0})

# Convert back to timestamp
myTable = myTable.withColumn("timestamp", F.col("timestamp_numeric").cast("timestamp")).drop("timestamp_numeric")

# Show the result
myTable.show()

from pyspark.ml.feature import Imputer
from pyspark.sql import functions as F

# Convert timestamp to numeric (e.g., epoch time)
myTable = myTable.withColumn("timestamp_numeric", F.col("timestamp").cast("long"))

# Initialize the Imputer
imputer = Imputer(inputCols=["timestamp_numeric"], outputCols=["timestamp_numeric"])

# Fit the Imputer model and transform the DataFrame
imputed_df = imputer.fit(myTable).transform(myTable)

# Convert back to timestamp
imputed_df = imputed_df.withColumn("timestamp", F.col("timestamp_numeric").cast("timestamp")).drop("timestamp_numeric")

# Show the result
imputed_df.show()

# Fill null values in 'label' with a default value (e.g., 0.0)
myTable = myTable.na.fill({"label": 0.0})

# Verify that nulls have been filled
myTable.select(F.sum(F.col("label").isNull().cast("int")).alias("null_count")).show()

# Fill null values in 'label' with a default value (e.g., 0.0)
myTable = myTable.na.fill({"reviews": ''})

# Verify that nulls have been filled
myTable.select(F.sum(F.col("reviews").isNull().cast("int")).alias("null_count")).show()

from pyspark.sql import functions as F

# Create a DataFrame with the count of null values for each column
null_counts = myTable.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in myTable.columns])

# Show the result
null_counts.show()

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Tokenizer
from pyspark.ml.feature import SQLTransformer

# Step 1: Tokenize the text
tokenizer = Tokenizer(inputCol="reviews", outputCol="words")

# Step 2: Explode tokens into separate rows using SQLTransformer
explode_transformer = SQLTransformer(
    statement="SELECT *, EXPLODE(words) AS word FROM __THIS__"
)

# Step 3: Filter out empty tokens using SQLTransformer
filter_transformer = SQLTransformer(
    statement="SELECT * FROM __THIS__ WHERE word != ''"
)

# Step 4: Index the words
#indexer = StringIndexer(inputCol="word", outputCol="word_index")
indexer = StringIndexer(inputCol="word", outputCol="word_index", handleInvalid="keep")


# Step 5: One-hot encode the indexed words
encoder = OneHotEncoder(inputCols=["word_index"], outputCols=["onehot_features"])

# Assemble the stages into a pipeline
pipeline = Pipeline(stages=[tokenizer, explode_transformer, filter_transformer, indexer, encoder])

# Fit the pipeline model
pipeline_model = pipeline.fit(myTable)

# Apply the pipeline model to the DataFrame
transformed_data = pipeline_model.transform(myTable)

# Select specific columns and show the results
transformed_data.select("reviews", "word", "onehot_features").show(5, truncate=False)

# Save the entire pipeline model
pipeline_model_path = "gs://pyq/pipeline_model"
pipeline_model.write().overwrite().save(pipeline_model_path)

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Tokenizer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import explode

# Assemble features
assembler = VectorAssembler(inputCols=["onehot_features", "timestamp_numeric"], outputCol="features")
assembled_df = assembler.transform(transformed_data)

# Split data
train_df, test_df = assembled_df.randomSplit([0.8, 0.2], seed=1234)

# Train the model
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)

# Make predictions
predictions = lr_model.transform(test_df)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Stop Spark session
# spark.stop()

lr_model.write().overwrite().save("gs://pyq/lrmodel")

from pyspark.ml.classification import LogisticRegressionModel

# Load the saved model
model_path = "gs://pyq/lrmodel"
model = LogisticRegressionModel.load(model_path)

# Use the model to make predictions
predictions = model.transform(test_df)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")