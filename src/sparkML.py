import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import Tokenizer
from pyspark.ml.linalg import Vectors

print(ps.__version__)

spark = SparkSession.builder    \
                    .appName("Spark first steps")    \
                    .master("local[*]") \
                    .getOrCreate()

features_df = spark.createDataFrame([
    (1, Vectors.dense([10.0, 10000.0, 1.0])),
    (2, Vectors.dense([20.0, 40000.0, 2.0])),
    (3, Vectors.dense([30.0, 50000.0, 3.0])),
], ["id", "features"])
features_df.show()

features_df_2 = spark.createDataFrame([
    (1, Vectors.dense([10.0, 10000.0, 1.0]),),
    (2, Vectors.dense([20.0, 40000.0, 2.0]),),
    (3, Vectors.dense([30.0, 50000.0, 3.0]),),
], ["id", "features"])
features_df_2.show()

data = [
    (Vectors.dense([10.0, 10000.0, 1.0]),),
    (Vectors.dense([20.0, 40000.0, 2.0]),),
    (Vectors.dense([30.0, 50000.0, 3.0]),),
]

features_df_3 = spark.createDataFrame(data, ["features"])
features_df_3.show()

""" features_df_4 = spark.createDataFrame([     Dataframe elements should be tuples, thats why the comma
    (Vectors.dense([10.0, 10000.0, 1.0])),
    (Vectors.dense([20.0, 40000.0, 2.0])),
    (Vectors.dense([30.0, 50000.0, 3.0])),
], ["features"])
features_df_4.show() """

features_scaler = MinMaxScaler(inputCol = "features", outputCol = "sfeatures")
scaled_model = features_scaler.fit(features_df)
scaled_features_df = scaled_model.transform(features_df)
scaled_features_df.show()

standart_features_scaler = StandardScaler(inputCol="features", outputCol="sfeatures", withMean=True, withStd=True)
standart_scaled_model = standart_features_scaler.fit(features_df)
standart_scaled_df = standart_scaled_model.transform(features_df)
standart_scaled_df.show()

splits = [-float("inf"), -10, 0.0, 10, float("inf")]
b_data = [(-800.0,), (-10.0), (-1.7), (0.0,), (8.2, 90.1)]
