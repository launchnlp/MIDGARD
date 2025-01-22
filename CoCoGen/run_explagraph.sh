#!/bin/bash

conda activate /data/inderjeet/conda_envs/argument_mining
export PYTHONPATH=".:../:.:src:../:../../"

# iterating between seeds
for seed in {0..4}
do
    python src/api/query_explagraphs.py --train_size 11 --temperature 0.0 --random_state ${seed} --output_file_path data/explagraphs/output_${seed}_11.jsonl
done