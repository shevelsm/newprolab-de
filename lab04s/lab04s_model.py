import re
from urllib.parse import urlparse
from urllib.request import urlretrieve, unquote

import pyspark.sql.functions as F
from pyspark.sql.types import *

from pyspark import keyword_only
from pyspark.ml import Transformer #, Estimator, EstimatorModel
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, HasInputCols, HasOutputCols, Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

# https://stackoverflow.com/questions/41399399/serialize-a-custom-transformer-using-python-to-be-used-within-a-pyspark-ml-pipel



#
# Train dataset JSON schema
#
train_schema = StructType(
   fields = [
      StructField("uid", StringType(), True),
      StructField("gender_age", StringType(), True),
      StructField("visits",  ArrayType( 
          StructType(
           fields = [
            StructField("url", StringType(), True),
            StructField("timestamp", LongType(), True),
      ])), True),
])

#
# Test dataset JSON schema
#
test_schema = StructType(
   fields = [
      StructField("uid", StringType(), True),
      StructField("visits",  ArrayType( 
          StructType(
           fields = [
            StructField("url", StringType(), True),
            StructField("timestamp", LongType(), True),
      ])), True),
])


class Url2DomainTransformer(
    Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable,
):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(Url2DomainTransformer, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, dataset):
        
        def url2domain(url):
            url = re.sub('(http(s)*://)+', 'http://', url)
            parsed_url = urlparse(unquote(url.strip()))
            if parsed_url.scheme not in ['http','https']: return None
            netloc = re.search("(?:www\.)?(.*)", parsed_url.netloc).group(1)
            if netloc is not None: return str(netloc.encode('utf8')).strip()
            return None
        
        @F.udf(ArrayType(StringType()))
        def udf_url2domain(xs):
            if xs is not None:
                return [url2domain(x) for x in xs]
        
        dataset = dataset.withColumn("urls", F.col(self.getInputCol()).url) \
                         .withColumn(self.getOutputCol(), udf_url2domain(F.col("urls")))
        
        return dataset

#
# Model Definition
#
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StringIndexer, IndexToString

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# we know our label values in advance, can force them to be in alphabet order
labels = ((x,y) for x in ["M", "F"] for y in ['18-24', '25-34', '35-44', '45-54', '>=55'])
label_strings = sorted(["{}:{}".format(*x) for x in labels])

#
# Columns
#
input_cols = ["visits"]
label_cols = ["gender_age"]
output_cols = ["uid", "gender_age"]

#
# Pipeline Stages
#
url2dom = Url2DomainTransformer(inputCol="visits", outputCol="domains")

cv = CountVectorizer(inputCol="domains", outputCol="features")

#we force labels to be in alphabet order
indexer = StringIndexer(inputCol="gender_age", outputCol="label", stringOrderType='alphabetAsc')

lr = LogisticRegression() #maxIter=10, regParam=0.01)

#using same labels
converter = IndexToString(inputCol="prediction", outputCol="gender_age_pred", labels = label_strings)

#
# Assemble Pipeline
#
pipeline = Pipeline(stages=[indexer, url2dom, cv, lr, converter])

#
# Hyperparameter tuning
#

lr_elasticnetparams = [0.1, 0.5, 0.9]
lr_maxiter = [5, 10]
lr_regparam = [0.001, 0.01]


lr_elasticnetparams = [0.5]
lr_maxiter = [10]
lr_regparam = [0.01]

paramGrid = ParamGridBuilder() \
    .addGrid(lr.elasticNetParam, lr_elasticnetparams) \
    .addGrid(lr.maxIter, lr_maxiter) \
    .addGrid(lr.regParam, lr_regparam) \
    .build()

evaluator = MulticlassClassificationEvaluator(metricName='accuracy')

pipeline_crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3) 


