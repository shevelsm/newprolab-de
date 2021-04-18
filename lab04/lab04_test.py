import re
import os

from urllib.parse import urlparse
from urllib.request import urlretrieve, unquote

from pyspark.sql import SparkSession

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import IndexToString

# ENVs
os.environ["PYSPARK_PYTHON"] = "/usr/share/python3-ml/bin/python3.6"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/share/python3-ml/bin/python3.6"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--conf spark.executor.instances=3 --conf spark.executor.cores=1 "\
                                    "--conf spark.executor.memory=4g   --conf spark.driver.memory=1g "\
                                    "--conf spark.sql.shuffle.partitions=3   --conf spark.default.parallelism=3 "\
                                    "--master yarn --deploy-mode client pyspark-shell "\
                                    "--packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.3.2"

# AFS
model_path = "/user/ubuntu/lab04/lab04_model.ml"

name = "alexey_shevelev"
topic_in = name + "_lab04_in"
topic_out = name + "_lab04_out"
kafka_bootstrap = "95.163.181.28:6667"

destination_path = "/user/ubuntu/lab04/prediction"
checkpointPath = "/tmp/lab04_checkpoint"

# Spark init
spark = SparkSession.builder.appName("lab04_test").getOrCreate()
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
