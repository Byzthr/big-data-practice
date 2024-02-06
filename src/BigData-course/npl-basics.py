from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

spark = SparkSession.builder    \
                    .appName("Spark first steps")    \
                    .master("local[*]") \
                    .getOrCreate()

sentencesDataFrame = spark.createDataFrame([
    (0, "Buenos dias seniorias"),
    (1, "El que mucho abarca, poco la marca"),
    (2, "Comida, bebida, almuerzo y comida otra vez")
], ["id", "sentence"])

sentenceTokenizer = Tokenizer(inputCol="sentence", outputCol="tokens")
hashinfTF = HashingTF(inputCol="tokens", outputCol="rawFeatures", numFeatures=20)

tokenizedSentences = sentenceTokenizer.transform(sentencesDataFrame)

tokenizedSentences.show()
print(str(tokenizedSentences.take(10)))


