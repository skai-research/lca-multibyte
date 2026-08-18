#!/bin/bash
#download shard_000181 to shard_000200 in to train/
#download shard_000200 to shard_000210 in to validation/
#download shard_000210 to shard_000220 in to test/
#path example: https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main/shard_00042.parquet

BASE_URL="https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"

mkdir -p train validation test

# Download shards 000181 to 000199 into train/
for i in $(seq 000 199); do
    shard="shard_$(printf '%05d' $i).parquet"
    echo "Downloading $shard -> train/"
    wget -q --show-progress -P train/ "${BASE_URL}/${shard}"
done

# Download shards 000200 to 000209 into validation/
for i in $(seq 200 201); do
    shard="shard_$(printf '%05d' $i).parquet"
    echo "Downloading $shard -> validation/"
    wget -q --show-progress -P validation/ "${BASE_URL}/${shard}"
done

# Download shards 000210 to 000219 into test/
for i in $(seq 210 219); do
    shard="shard_$(printf '%05d' $i).parquet"
    echo "Downloading $shard -> test/"
    wget -q --show-progress -P test/ "${BASE_URL}/${shard}"
done
