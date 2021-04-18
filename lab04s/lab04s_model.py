train_schema = StructType(
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

test_schema = None

class Url2DomainTransformer:
    pass

label_strings = [None]

pipeline = Pipeline(stages=[cv, indexer, lr])

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

input_cols = ["visits"]
label_cols = ["gender_age"]
output_cols = ["uid", "gender_age", "probability"]