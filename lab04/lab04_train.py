import os
import re
import logging
from urllib.parse import urlparse
from urllib.request import urlretrieve, unquote

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import udf

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StringIndexer, IndexToString

# ENVs
os.environ["PYSPARK_PYTHON"] = "/usr/share/python3-ml/bin/python3.6"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/share/python3-ml/bin/python3.6"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--conf spark.executor.instances=3 --conf spark.executor.cores=1 "\
                                    "--conf spark.executor.memory=4g   --conf spark.driver.memory=1g "\
                                    "--conf spark.sql.shuffle.partitions=3   --conf spark.default.parallelism=3 "\
                                    "--master yarn --deploy-mode client pyspark-shell"

# AFS
train_path = "/user/ubuntu/lab04/lab04_train_merged_labels.json"
model_path = "/user/ubuntu/lab04/lab04_model.ml"

# Spark session
spark = SparkSession.builder.appName("lab04_train").getOrCreate()
spark.sparkContext.setLogLevel('WARN')

# Train dataset JSON schema
schema = StructType(
    fields=[
        StructField("uid", StringType(), True),
        StructField("gender_age", StringType(), True),
        StructField("visits", ArrayType(
            StructType(
                fields=[
                    StructField("timestamp", LongType(), True),
                    StructField("url", StringType(), True),
                ])), True),
    ])

# Read the data
train = spark.read.json(train_path, schema=schema)
train = spark.read.schema(schema).format("json").load(train_path)

logging.info("Original data:")
train.show(5)

# Merge gender and age columns to a single column for 10 class classification
# train1 = train.withColumn("label_string", F.concat(F.col('gender'), F.lit(':'), F.col('age') ))
# logging.info(("Gender + age merged:")
# train1.show(5)

# Extract urls only from visits
train2 = train.select("uid", F.col("visits").url.alias("urls"), "gender_age")
logging.info("URLS extracted from visits:")
train2.show(5)

# Working with url domain
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

# Extract domains
logging.info("Domains extracted")
train2url = train2.withColumn('domains', foo_udf(F.col('urls')))
train2url.show(5, truncate=True)


# Model Definition
cv = CountVectorizer(inputCol="domains", outputCol="features")

lr = LogisticRegression()

indexer = StringIndexer(inputCol="gender_age", outputCol="label")

pipeline = Pipeline(stages=[cv, indexer, lr])

# Train the model
model = pipeline.fit(train2url)

# Save the model
model.write().overwrite().save(model_path)
