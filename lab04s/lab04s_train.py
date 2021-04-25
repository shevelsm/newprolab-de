#
# Lab04s - implementation with GridParam search with Cross Validation
#

from pyspark.sql import SparkSession

import pyspark.sql.functions as F
from pyspark.sql.types import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StringIndexer, IndexToString

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

#
# Get app name and app options
#
import sys
if len(sys.argv)==0:
    print("Give an argument - an appName for Spark")
    sys.exit(1)

appName = sys.argv[1]
print("APP", appName)

#
# Spark context init
#
spark = SparkSession.builder.appName(appName).getOrCreate()
spark.sparkContext.setLogLevel('WARN')

#
# App config (from submit command line or from zookeeper)
#
train_path = spark.conf.get("spark."+appName+".train_path")
model_path = spark.conf.get("spark."+appName+".model_path")
print("TRAIN_PATH", train_path)
print("MODEL_PATH", model_path)


# 
# Read the model and model's parameters
#
# this needs to be done after Spark context initialized.

import lab04s_model

from lab04s_model import Url2DomainTransformer, pipeline_crossval
from lab04s_model import input_cols, label_cols, output_cols, train_schema

#
# Read the data
#
train = spark.read.json(train_path, schema = train_schema)

train.show(5)

#
# Train the model
#
cv_model = pipeline_crossval.fit(train)


#
# Save the model
#
best_model = cv_model.bestModel
best_model.write().overwrite().save(model_path)

#
# Hyperparameters
#
lr_model = best_model.stages[-2]

print(
    lr_model.getOrDefault(lr_model.getParam("regParam")),
    lr_model.getOrDefault(lr_model.getParam("maxIter")),
    lr_model.getOrDefault(lr_model.getParam("elasticNetParam"))
)

# This model should give 0.2534 on test.

