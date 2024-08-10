# MIDGARD: Self-consistency approach using LLMs for structured commonsense reasoning
This is the official repository for ["MIDGARD: Self-consistency approach using LLMs for structured commonsense reasoning"](https://arxiv.org/pdf/2405.05189), which was accepted to the proceedings of ACL, 2024. 

This work proposes a novel technique for the task of structured commonsense reasoning using the principle of Minimum Description Length. The code is structured into 4 folders (`argument_structure_extraction`, `explanation_graph_generation`, `script_planning`, `semantic_graph_generation`) for each of the structured commonsense reasoning tasks. Our code has been tested on Python 3.8+. Please install the requirements before running our scripts as follows:

```
pip install -r requirements.txt
```

Please cite our paper, if you found this repo useful for your project:
```
@misc{nair2024midgardselfconsistencyusingminimum,
    title={MIDGARD: Self-Consistency Using Minimum Description Length for Structured Commonsense Reasoning}, 
    author={Inderjeet Nair and Lu Wang},
    year={2024},
    eprint={2405.05189},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2405.05189}, 
}
```