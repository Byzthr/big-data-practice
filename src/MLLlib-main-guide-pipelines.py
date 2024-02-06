from pyspark.sql import SparkSession

from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder \
                    .appName("Pipelines first steps")   \
                    .getOrCreate()

# Parameters

training1 = spark.createDataFrame([
    (1.0, Vectors.dense([0.0, 1.1, 0.1])),
    (0.0, Vectors.dense([2.0, 1.0, -1.0])),
    (0.0, Vectors.dense([2.0, 1.3, 1.0])),
    (1.0, Vectors.dense([0.0, 1.2, -0.5]))
], ["label", "features"])

lr = LogisticRegression(maxIter=10, regParam=0.001)

paramMap1 = {lr.maxIter: 10, lr.regParam: 0.001}      # Other ways to specify parameters
paramMap2 = {lr.probabilityCol: "myProbability"}
paramMap = paramMap1.copy().update(paramMap2)
#print(f"LogisticRegression parameters:\n{lr.explainParams()}\n")

model1 = lr.fit(training1)

#print(f"model1 trained using parameters:\n{model1.extractParamMap()}")

model2 = lr.fit(training1, paramMap2)

#print(f"model2 trained using parameters:\n{model2.extractParamMap()}")

data = spark.createDataFrame([
    (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
    (0.0, Vectors.dense([3.0, 2.0, -0.1])),
    (1.0, Vectors.dense([0.0, 2.2, -1.5]))
], ["label", "features"])

prediction = model2.transform(data)

result = prediction.select("features", "label", "myProbability", "prediction") \

prediction.show()
result.show()

# Pipelines

from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer

training2 = spark.createDataFrame([
    (0, "Me como la sopa con una cuchara de sparko", 1.0),
    (1, "Afganistan es el pais del diablo", 0.0),
    (2, "Sistema estable dependiente de la gente", 1.0),
    (3, "Estabilidad de estado no posible", 0.0)
], ["id", "text", "label"])

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashigTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")

pipeline = Pipeline(stages=[tokenizer, hashigTF, lr])

pipelineModel = pipeline.fit(training2)

test = spark.createDataFrame([
    (4, "Sparking center"),
    (5, "Taliban taliben"),
    (6, "Hay que estar agusto al final"),
    (7, "Apache Hadoop")
], ["id", "text"])

prediction = pipelineModel.transform(test).select("id", "text", "probability", "prediction")

prediction.show()
