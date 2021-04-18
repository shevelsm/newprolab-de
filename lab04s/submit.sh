#!/bin/bash
# Try to start locally
echo $1
PYSPARK_PYTHON=/usr/share/python3-ml/bin/python3.6 spark-submit \
    --conf spark.streaming.batch.duration=10 \
    --conf spark.executor.instances=3 \
    --conf spark.executor.memory=4g \
    --conf spark.driver.memory=1g \
    --conf spark.sql.shuffle.partitions=3 \
    --conf spark.default.parallelism=3 \
    --master yarn \
    --deploy-mode client\
    --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.3.2 \
    $1