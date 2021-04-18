import re

from urllib.parse import urlparse
from urllib.request import unquote

from pyspark.sql import SparkSession

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import IndexToString


# AFS
model_path = "/user/ubuntu/lab04/lab04s_model.ml"

name = "alexey_shevelev"
topic_in = name + "_lab04_in"
topic_out = name + "_lab04_out"
kafka_bootstrap = "95.163.181.28:6667"

destination_path = "/user/ubuntu/lab04/prediction_4s"
checkpointPath = "/tmp/lab04s_checkpoint"

# Spark init
spark = SparkSession.builder.appName("lab04s_test").getOrCreate()
spark.sparkContext.setLogLevel('WARN')

# Test dataset JSON schema
schema = StructType(
    fields=[
        StructField("uid", StringType(), True),
        StructField("visits", ArrayType(
            StructType(
                fields=[
                    StructField("timestamp", LongType(), True),
                    StructField("url", StringType(), True),

                ])), True),
    ])

# Extract domains from URLs
def url2domain(url):
    url = re.sub('(http(s)*://)+', 'http://', url)
    parsed_url = urlparse(unquote(url.strip()))
    if parsed_url.scheme not in ['http', 'https']: return None
    netloc = re.search("(?:www\.)?(.*)", parsed_url.netloc).group(1)
    if netloc is not None: return str(netloc.encode('utf8')).strip()
    return None


def transform(f, t=StringType()):
    if not isinstance(t, DataType):
        raise TypeError("Invalid type {}".format(type(t)))

    @udf(ArrayType(t))
    def _(xs):
        if xs is not None:
            return [f(x) for x in xs]

    return _


foo_udf = transform(url2domain)

# Model load
model = PipelineModel.load(model_path)

# Read the srtream
st = spark \
    .readStream \
    .format("kafka") \
    .option("checkpointLocation", checkpointPath) \
    .option("kafka.bootstrap.servers", kafka_bootstrap) \
    .option("subscribe", topic_in) \
    .option("startingOffsets", "latest") \
    .load() \
    .selectExpr("CAST(value as string)") \
    .select(F.from_json("value", schema).alias("value")) \
    .select(F.col("value.*")) \
    .select("uid", F.col('visits').url.alias("urls")) \
    .withColumn('domains', foo_udf(F.col('urls')))

# Infer on test data
results = model.transform(st)
converter = IndexToString(inputCol="prediction", outputCol="gender_age", labels=model.stages[1].labels)
converted = converter.transform(results)

# Saving to out topic
query = converted \
    .select(F.to_json(F.struct("uid", "gender_age")).alias("value")) \
    .writeStream \
    .outputMode("append") \
    .format("kafka") \
    .option("checkpointLocation", checkpointPath) \
    .option("kafka.bootstrap.servers", kafka_bootstrap) \
    .option("topic", topic_out) \
    .start()

query.awaitTermination()
