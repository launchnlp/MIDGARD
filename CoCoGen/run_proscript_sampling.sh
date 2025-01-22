#!/bin/bash
export PYTHONPATH=".:../:.:src:../:../../"

python src/api/query_proscript.py --train_size 15 --temperature 0.0  --output_file_path data/proscript_script_generation/output.jsonl

for sample in {0..9}
do
    echo "Running sample ${sample}"
    python src/api/query_proscript.py --train_size 15 --temperature 0.9  --output_file_path data/proscript_script_generation/output_${sample}.jsonl 
done