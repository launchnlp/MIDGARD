# Argument Structure Extraction
In this folder, we provide the code for merging different argument structures obtained from temperature sampling. Throughout our code, we represent the argument structure as a programming script. For instance, consider the following programming script

```python
class argument_structure:
    def __init__(self):

        # node declarations
        claim_1 = 'cloning will be beneificial for many people who are in need of organ transplants'
        premise_1 = 'Cloned organs will match perfectly to the blood group and tissue of patients'
        premise_2 = 'they can be raised from the cloned stem cells of the patients'
        premise_3 = 'it shortens the healing process'
        premise_4 = 'it is very rare to find the appropriate organ donor'
        premise_5 = 'by using cloning in order to raise the required organs the waiting time can be shortened tremedously'

        # edge declarations
        add_edge(premise_1, claim_1, 'supports')
        add_edge(premise_3, claim_1, 'supports')
        add_edge(premise_2, premise_1, 'supports')
        add_edge(premise_4, premise_3, 'supports')
        add_edge(premise_5, premise_3, 'supports')
```

Our code can be executed for the following three popular argument mining datasets:
- [Essays](https://direct.mit.edu/coli/article/43/3/619/1573/Parsing-Argumentation-Structures-in-Persuasive) present at `data/ArgumentAnnotatedEssays-2.0`
- [AbstRCT](https://ebooks.iospress.nl/volumearticle/55129) present at `data/abstrct-master`
- [CDCP](https://aclanthology.org/L18-1257/) present at `data/cdcp`

You can execute our code in 3 steps assuming that you have installed all the required modifications: entering the open ai credentials, running the temperature sampling to generate multiple argument structures, and finally merging the argument structures.

### Providing OpenAI credentials
To set the OpenAI key, use the following command:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Generating multiple responses using temperature sampling
To generate multiple argument structures, use the following command:
```bash
python executor.py \
    --train_size 3 \
    --test_size 50 \
    --random_state 42 \
    --data essay \
    --data_dir data/ArgumentAnnotatedEssays-2.0/brat-project-final/ \
    --output_dir data/output_essay_3_50_42 \
    --engine gpt-3.5-turbo \
    --sample_multiple_outputs \
    --num_samples 10
```

The meaning of the arguments is as follows:
- `train_size`: Number of few-shot examples to use while generating multiple responses
- `test_size`: Number of test examples for which multiple responses are generated
- `random_state`: Random seed for sampling train and test examples
- `data`: The dataset to use for generating multiple responses. It can be one of `essay`, `abstract`, or `cdcp`
- `data_dir`: The directory where the dataset is present. For `essay`, it is `data/ArgumentAnnotatedEssays-2.0/brat-project-final/`, for `abstract`, it is `data/abstrct-master`, and for `cdcp`, it is `data/cdcp`
- `output_dir`: The directory where the output is stored
- `engine`: The OpenAI model to use for generating multiple responses.
- `sample_multiple_outputs`: Flag to sample multiple responses
- `num_samples`: Number of samples to generate for each input

### Merging the argument structures
To run the merging code, use the following command:
```bash
python ensembling.py \
    --dir_name data/output_essay_3_50_42 \
    --target_dir_name data/output_ensembled_essay_3_50_42 \
    --sample_length 10 \
    --overlap_threshold 0.7 \
    --edge_threshold 0.1 \
    --node_threshold 0.1 \
    --method mld_dag \
    --node_constraints
```

The meaning of the arguments is as follows:
- `dir_name`: The directory where the multiple responses are stored
- `target_dir_name`: The directory where the ensembled responses are stored
- `sample_length`: The number of samples to be used for each datapoint. Note that, if this is set less than the number of samples generated, only the first `sample_length` samples are used.
- `overlap_threshold`: The threshold for the overlap between two nodes to be considered as the same node
- `edge_threshold`: The larger this value, the more edges are pruned. This is equivalent to (1 - \lambda_1) in the paper
- `node_threshold`: The larger this value, the more nodes are pruned. This is equivalent to (1 - \lambda_2) in the paper
- `method`: `mld_dag` corresponds to the method described in the paper.
- `node_constraints`: Flag to use node constraints while merging the argument structures. This will ensure that the objective not just accounts for edges but also for nodes.

If you wish to automatically estimate the parameters `edge_threshold` and `node_threshold` from the few shot examples used for generating multiple responses, you can provide `--estimate_parameters` as an argument. This will automatically estimate the parameters and use them for merging the argument structures.

### Evaluation
For quantitative evaluation of the merged argument structures, you can use the following command:
```bash
python evaluator.py \
    --input_dir data/output_ensembled_essay_3_50_42 \
    --edge_evaluation
```
