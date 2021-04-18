#!/bin/bash
# Try to start locally
echo $1
PYSPARK_PYTHON=/usr/share/python3-ml/bin/python3.6 spark-submit \
    --conf spark.streaming.batch.duration=10 \
    --master local[1] \
    --executor-memory 4G \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.3.2 \
    $1