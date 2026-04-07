from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import when

def train_model():
    # Start Spark
    spark = SparkSession.builder.appName("Sentiment").getOrCreate()

    # Load dataset
    df = spark.read.csv("data/product_reviews.csv", header=True, inferSchema=True)

    # Convert Score → sentiment (0 or 1)
    df = df.withColumn(
        "sentiment",
        when(df["Score"] >= 3, 1).otherwise(0)
    )

    # Text processing
    tokenizer = Tokenizer(inputCol="Text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    # Model
    lr = LogisticRegression(labelCol="sentiment", featuresCol="features")

    # Pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    # Train model
    model = pipeline.fit(df)

    return model