from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer

spark = SparkSession.builder    \
                    .appName("Spark first steps")    \
                    .master("local[*]") \
                    .getOrCreate()

splits = [-float("inf"), -10, 0.0, 10, float("inf")]
rawData = [(-800.0,), (-10.0,), (-1.7,), (0.0,), (8.2,), (90.1,)]

data = spark.createDataFrame(rawData, ["features"])

bucketizer = Bucketizer(splits = splits, inputCol="features", outputCol="bucketedFeatures")

bucketedData = bucketizer.transform(data)

print("Bucketized data")
bucketedData.show()
