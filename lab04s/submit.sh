PYSPARK_PYTHON=/home/ubuntu/miniconda3/envs/dsenv/bin/python spark-submit \
    --conf spark.streaming.batch.duration=10 \
    --conf spark.sql.streaming.schemaInference=true \
    --conf spark.model1.model_path="lab04s_model_custom.ml" \
    --conf spark.model1.train_path="lab04_train_merged_labels.json" \
    --conf spark.model1.test_path="lab04test5" \
    --conf spark.model1.pred_path="prediction" \
    --conf spark.model1.checkpoint_path="/tmp/chkp" \
    --py-files=lab04s_model.py \
    --master local[4] \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.3.2 \
    $*

