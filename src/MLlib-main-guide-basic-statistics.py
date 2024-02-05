from pyspark.sql import SparkSession

from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation

spark = SparkSession.builder \
                    .appName("Spark Statistics first steps") \
                    .getOrCreate()

data = [
        (Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),    # Sparse vectors can be initiated using tuples (index, value)
        (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
        (Vectors.dense([5.0, 7.0, 0.0, 8.0]),),
        (Vectors.sparse(4, [0, 3], [1.0, 9.0]),)        # Sparse vectors can aalso be initiated using two lists [indexes], [values]  
        ]

df =spark.createDataFrame(data, ["features"])
df.show()

rPearson = Correlation.corr(df, "features").head()        # Take only the first row, since the result is a dataframe with a single row
rSpearman = Correlation.corr(df, "features", "spearman").head()

print("Pearson correlaiton matrix:\n" + str(rPearson[0]))
print("Spearman correlation matrix:\n" +  str(rSpearman[0]))