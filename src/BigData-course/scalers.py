import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StandardScaler

print(ps.__version__)

spark = SparkSession.builder    \
                    .appName("Spark first steps")    \
                    .master("local[*]") \
                    .getOrCreate()

# Scalers

featuresDf = spark.createDataFrame([
    (1, Vectors.dense([10.0, 10000.0, 1.0])),
    (2, Vectors.dense([20.0, 40000.0, 2.0])),
    (3, Vectors.dense([30.0, 50000.0, 3.0])),
], ["id", "features"])
print("Initial data 1:\n")
featuresDf.show()

data2 = [
    (1, Vectors.dense([1.0, 0.1, -1.0]),),
    (2, Vectors.dense([2.0, 1.1, 1.0]),),
    (3, Vectors.dense([3.0, 10.1, 3.0]),),
]

featuresDf2 = spark.createDataFrame(data2, ["id", "features"])
print("Data 2")
featuresDf2.show()

""" features_df_4 = spark.createDataFrame([     Dataframe elements should be tuples, thats why the comma
    (Vectors.dense([10.0, 10000.0, 1.0])),
    (Vectors.dense([20.0, 40000.0, 2.0])),
    (Vectors.dense([30.0, 50000.0, 3.0])),
], ["features"])
#features_df_4.show() """

mMScaler = MinMaxScaler(inputCol = "features", outputCol = "scaledFeatures")
scaledModel = mMScaler.fit(featuresDf)
scaledFeaturesDf = scaledModel.transform(featuresDf)
print("MinMax scaled:\n")
scaledFeaturesDf.show()

scaledModel2 = mMScaler.fit(featuresDf2)
scaledFeaturesDf2 = scaledModel2.transform(featuresDf2)
print(f"MinMax scaled data 2:\n")
print(str(scaledModel2))
scaledFeaturesDf2.show()


standart_features_scaler = StandardScaler(inputCol="features", outputCol="sfeatures", withMean=True, withStd=True)
standart_scaled_model = standart_features_scaler.fit(featuresDf)
standart_scaled_df = standart_scaled_model.transform(featuresDf)
print("Standart scaled:\n")
standart_scaled_df.show()
