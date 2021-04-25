
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext


import pyspark.sql.functions as F
from pyspark.sql.types import *


from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer

from pyspark.ml.feature import  IndexToString #,StringIndexer
from pyspark.sql.functions import from_json

#
# Get app name as this script argument
#
import sys
if len(sys.argv)==0:
    print("Give an argument - an appName for Spark")
    sys.exit(1)

appName = sys.argv[1]
print("APP", appName)

#
# Spark init
#
spark = SparkSession.builder.appName(appName).getOrCreate()
spark.sparkContext.setLogLevel('WARN')

#
# Import the model
#
# needs to be done after context init

from lab04s_model import Url2DomainTransformer
from lab04s_model import test_schema, input_cols, output_cols


#
# App config (from submit arguments or zookeeper)
#

model_path = spark.conf.get("spark."+appName+".model_path")
test_path = spark.conf.get("spark."+appName+".test_path") 
pred_path = spark.conf.get("spark."+appName+".pred_path") 
checkpoint_path = spark.conf.get("spark."+appName+".checkpoint_path") #"/tmp/chkp"

#
# Load the model
#
model = PipelineModel.load(model_path)

print(model.stages)

name_surname = "name_surname"
topic_in = name_surname + "_lab04_in"
topic_out = name_surname + "_lab04_out"
# ! use your own IP
kafka_bootstrap = "85.192.32.243:6667"

#
# Read the stream (files from a dir)
#
st = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", kafka_bootstrap ) \
  .option("subscribe", topic_in) \
  .option("startingOffsets", "latest") \
  .load()\
  .selectExpr("CAST(value as string)")\
  .select(from_json("value", test_schema).alias("value"))\
  .select(F.col("value.uid").alias("uid")\
  ,F.col("value.visits").alias("visits")\
  ,F.lit("").alias("gender_age")
  )\

#
# Apply the model
#
results = model.transform(st)

# Write to Kafka 
#
query = results\
 .select(F.col("uid"),F.col("gender_age_pred").alias("gender_age"))\
 .select(F.to_json(F.struct(*output_cols)).alias("value"))\
 .writeStream \
 .outputMode("update")\
 .format("kafka") \
 .option("checkpointLocation", "/tmp/checkpoint-write")\
 .option("kafka.bootstrap.servers", kafka_bootstrap ) \
 .option("topic", topic_out) \
 .start()\

# (optionally)
# Write stream results to dir as csv 
#
#query = results\
# .select(F.to_json(F.struct(*output_cols)).alias("value"))\
# .writeStream.format("text").outputMode("append")\
# .option("path", pred_path).option("checkpointLocation", checkpoint_path)\
# .start()

query.awaitTermination()
