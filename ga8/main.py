from sklearn.datasets import fetch_openml
import pandas as pd
from pyspark.sql import SparkSession, Row
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql.types import StructType, StructField, IntegerType
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler


mnist_data_set = fetch_openml("mnist_784", version=1, parser="auto", cache=True)
spark_session = SparkSession.builder.appName("mnist_classification").getOrCreate()

features = mnist_data_set.data[:5000]
labels = mnist_data_set.target[:5000]

features_schema = StructType([StructField(f'pixel_{i+1}', IntegerType(), True) for i in range(784)])
features_df = spark_session.createDataFrame(features, schema=features_schema)

labels_schema = StructType([StructField("label", IntegerType(), True)])
labels_df = spark_session.createDataFrame(pd.DataFrame(labels.apply(int)), schema=labels_schema)

features_df = features_df.withColumn("record_id", F.monotonically_increasing_id())
labels_df = labels_df.withColumn("record_id", F.monotonically_increasing_id())

complete_data = features_df.join(labels_df, "record_id", "inner").drop("record_id")
(training_data, testing_data) = complete_data.randomSplit([0.7, 0.3], seed=42)

label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
decision_tree_classifier = DecisionTreeClassifier(labelCol="indexedLabel")

feature_columns = [f"pixel_{i+1}" for i in range(784)]
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

ml_pipeline = Pipeline(stages=[label_indexer, vector_assembler, decision_tree_classifier])
parameter_grid = ParamGridBuilder().addGrid(decision_tree_classifier.maxDepth, [5, 10, 15]).build()
classification_evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")

cross_validator = CrossValidator(estimator=ml_pipeline,
                                 estimatorParamMaps=parameter_grid,
                                 evaluator=classification_evaluator,
                                 numFolds=3)

cv_trained_model = cross_validator.fit(training_data)

optimal_params = cv_trained_model.bestModel.stages[-1].extractParamMap()
for parameter, value in optimal_params.items():
    print(f"{parameter.name}: {value}")

predicted_results = cv_trained_model.transform(testing_data)
evaluation_metric = classification_evaluator.evaluate(predicted_results)
print(f"Weighted Precision on test data: {evaluation_metric}")

spark_session.stop()
