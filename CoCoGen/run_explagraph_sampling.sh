#!/bin/bash
export PYTHONPATH=".:../:.:src:../:../../"

torchrun --nproc_per_node 2 src/api/query_explagraphs.py --train_size 30 --temperature 0.0  --output_file_path data/explagraphs/output_llama_30.jsonl --llama

# iterating between seeds
for sample in {0..9}
do
    echo "Running sample ${sample}"
    torchrun --nproc_per_node 2 src/api/query_explagraphs.py --train_size 30 --temperature 0.9 --output_file_path data/explagraphs/output_llama_30_${sample}.jsonl --llama
done
