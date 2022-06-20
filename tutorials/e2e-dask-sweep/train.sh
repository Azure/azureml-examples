  python src/train-xgboost.py \
    --nyc_taxi_parquet ~/localfiles/nyctaxi.parquet/ \
    --model data/fare_predict \
    --tree_method auto \
    --learning_rate 0.3 \
    --gamma 1 \
    --max_depth 7 \
    --num_boost_round 12
