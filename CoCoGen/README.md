# Explanation Graph Generation and Script Planning
This folder is heavily inspired from the original [implementation](https://github.com/reasoning-machines/CoCoGen). Our modifications are as follows:
- Incorporating the logic for merging different graphs for script planning and explanation graph generation
- Making the code compatible with OpenAI version==1.40.3
- Including additional features for running open source models such as code-llama

### Prerequisites
To set the OpenAI key, use the following command:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

Moreover, you need to export the following python path:
```bash
export PYTHONPATH=".:../:.:src:../:../../"
```

Additionally, to use code-llama, you need to install the package from the [codellama](https://github.com/meta-llama/codellama) repository.

### Generating single response for each datapoint

To generate a single response for each datapoint in Explanation graph generation, use the following command:
```bash
python src/api/query_explagraphs.py \
    --task_file_path data/explagraphs/dev.jsonl \
    --output_file_path outputs/explagraphs/dev_test.jsonl \
    --train_size 30 \
    --temperature 0.0 \
    --engine gpt-3.5-turbo
```

The meaning of the arguments is as follows:
- `task_file_path`: The path to the input file containing the datapoints
- `output_file_path`: The path to the output file where the responses are stored
- `train_size`: Number of few-shot examples to use while generating responses
- `temperature`: The temperature to use while generating responses
- `engine`: The OpenAI model to use for generating responses.

If one wants to use the code-llama model, then you need to set the `--llama` flag and provide the llama model path and tokenizer path at `--llama_ckpt_dir` and `--llama_tokenizer_path` respectively. Thereafter use the torchrun command along with appropriate number of processes. For example:
```bash
torchrun --nproc_per_node 2 python src/api/query_explagraphs.py \
    --task_file_path data/explagraphs/dev.jsonl \
    --output_file_path outputs/explagraphs/dev_test.jsonl \
    --train_size 30 \
    --temperature 0.0 \
    --engine gpt-3.5-turbo \
    --llama \
    --llama_ckpt_dir /path/to/llama/model \
    --llama_tokenizer_path /path/to/llama/tokenizer \
    --random_state 42
```
The number of processes depends on the size of checkpoint used. For more details, refer to the [codellama](https://github.com/meta-llama/codellama) repository.

To generate a single response for each datapoin in script planning, use the following command:
```bash
python src/api/query_proscript.py \ 
    --train_size 15 \
    --temperature 0.0 \
    --task_file_path data/proscript_script_generation/input.jsonl \
    --output_file_path data/proscript_script_generation/output.jsonl \
    --engine gpt-3.5-turbo
```

To generate multiple responses for each datapoint in explanation graph generation, use the following bash script:
```bash
for seed in {0..4}
do
    python src/api/query_explagraphs.py --train_size 11 --temperature 0.0 --random_state ${seed} --output_file_path data/explagraphs/output_${seed}_11.jsonl
done
```

Similarly, to generate multiple samples using code-llama, use the following bash script:
```bash
for seed in {0..4}
do
    torchrun --nproc_per_node 2 python src/api/query_explagraphs.py --train_size 11 --temperature 0.0 --random_state ${seed} --output_file_path data/explagraphs/output_${seed}_11.jsonl --llama --llama_ckpt_dir /path/to/llama/model --llama_tokenizer_path /path/to/llama/tokenizer
done
```

### Merging the different responses

To merge the different responses generated for each datapoint, use the following command:
```bash
python src/api/ensembling.py \
    --path_patern data/explagraphs/output_*.jsonl \
    --output_path data/explagraphs/aggregated_output.jsonl \
    --sample_length 10 \
    --overlap_threshold 0.7 \
    --edge_threshold 0.2 \
    --node_threshold 0.2 \
    --method mld_dag \
    --node_constraints
```

The meaning of the arguments is as follows:
- `path_patern`: The pattern to the files containing the responses
- `output_path`: The path to the output file where the aggregated responses are stored
- `sample_length`: The number of samples to be used for each datapoint. Note that, if this is set less than the number of samples generated, only the first `sample_length` samples are used.
- `overlap_threshold`: The threshold for the overlap between two nodes to be considered as the same node
- `edge_threshold`: The larger this value, the more edges are pruned. This is equivalent to (1 - \lambda_1) in the paper
- `node_threshold`: The larger this value, the more nodes are pruned. This is equivalent to (1 - \lambda_2) in the paper
- `method`: `mld_dag` corresponds to the method described in the paper.

### Evaluation
To evaluate a jsonl file for explanation graph generation, use the following command:
```bash
python src/eval/explagraph_eval_final.py --output_file data/explagraphs/aggregated_output.jsonl
```

To evaluate a jsonl file for script planning, use the following command:
```bash
python src/eval/edge_pred_eval.py path/to/predictions.jsonl path/to/save_results.json
```