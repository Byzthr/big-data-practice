import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["SPARK_HOME"] = "/home/byzthr/Documents/Spark/spark-3.5.0-bin-hadoop3"

import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder    \
                    .appName("Spark Hello World")    \
                    .master("local[*]") \
                    .getOrCreate()

dataFrame = spark.createDataFrame([{"Hello":"World"} for x in range(1000)])
dataFrame.show(3, False)

df = spark.read.json("cars.json")

print()
df.show()

df.printSchema()

df.select('model').show()

df.select(df['model'], 2024-df['fromYear']).show()

df.filter(df['engineType'] == 'GASOLINE').show()

df.groupBy(df['brand']).count().show()

# sqlDf = select