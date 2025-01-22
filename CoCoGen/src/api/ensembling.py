import glob
import jsonlines
import numpy as np
from pulp import *
from argparse import ArgumentParser
from statistics import mode
from typing import List, Dict, Any, Union, Tuple
from networkx import DiGraph
from tqdm import tqdm
from src.api.openai_api_wrapper import openai_chat_api
from src.eval.explagraph_eval_final import generate_results
from src.prompting.constants import END, SEPARATOR
from converters.get_converter import ConverterFactory

def compute_threshold(
    sent_a = str,
    sent_b = str
):
    '''
        Computes the set overlap between the tokens in the two sentences
    '''
    tokens_a = set(sent_a)
    tokens_b = set(sent_b)
    return len(tokens_a.intersection(tokens_b)) / max(len(tokens_a), len(tokens_b))

def find_representative_sentence(sentences: List[str]):
    '''
        Returns the sentence which has the maximum overlap with the other sentences
    '''
    
    # creating the similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    # iterate through the sentences
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            similarity_matrix[i][j] = compute_threshold(sentences[i], sentences[j])

    # summing along the rows to get the similarity score for each sentence
    similarity_score = np.sum(similarity_matrix, axis=1)

    # returning the sentence with the maximum similarity score
    return sentences[np.argmax(similarity_score)]


def check_common(common_node_dict: List[Dict[str, List[str]]], node_dict: Dict[str, str], threshold: float = 0.7) -> Union[int, None]:
    '''
        Check if the text variable of node_dict overlaps with that of common_node_dict
    '''

    common_node = None
    max_overlap = 0
    for node_id in range(len(common_node_dict)):
        for text_var in common_node_dict[node_id]['text']:
            overlap = compute_threshold(text_var, node_dict['text'])
            if overlap > max_overlap and overlap >= threshold:
                max_overlap = overlap
                common_node = node_id

    return common_node

def process_explagraph(input_string: str) -> List[Tuple[str, str, str]]:
    '''
        Converts explagraph string to list of tuples
    '''
    # Remove the outer single quotes and split the string into individual tuples  
    tuple_strings = input_string[1:-1].split(')(')  
    
    # Process each tuple string and split it into individual elements  
    tuples = [tuple_string.strip('()').split('; ') for tuple_string in tuple_strings]  
    
    # Convert the list of lists into a list of tuples  
    result = [tuple(tuple_string) for tuple_string in tuples]

    return result

def graph_to_string(graph: DiGraph) -> str:
    '''
        Converts the graph to string
    '''

    # iterating over the graph edges
    edge_list = []
    for edge in graph.edges():
        edge_list.append('({}; {}; {})'.format(graph.nodes[edge[0]]['text'], graph[edge[0]][edge[1]]['stance'], graph.nodes[edge[1]]['text']))
    return ''.join(edge_list)

def aggregate_graph_texts(graph_text_list: List[str], threshold: float = 0.8) -> DiGraph:
    '''
        Aggregate the graph texts from the graph_text_list
    '''
    
    common_node_list = []
    common_edge_list = []

    # iterate through the graph_text_list
    for graph_text in graph_text_list:

        # getting the graph information
        graph = process_explagraph(graph_text)
        
        # adding the nodes
        for edge_triplet in graph:
            
            # check if the edge triplet is valid
            if len(edge_triplet) < 3:
                continue
            
            # getting the source and target
            source = edge_triplet[0]
            target = edge_triplet[2]

            # check if the source node is present
            source_dict = {'text': source}
            source_node = check_common(common_node_list, source_dict, threshold)
            if source_node is None:
                source_node = len(common_node_list)
                common_node_list.append({
                    'text': [source],
                    'type': ['Node'],
                })
            else:
                common_node_list[source_node]['text'].append(source)
                common_node_list[source_node]['type'].append('Node')
            
            # check if the target node is present
            target_dict = {'text': target}
            target_node = check_common(common_node_list, target_dict, threshold)
            if target_node is None:
                target_node = len(common_node_list)
                common_node_list.append({
                    'text': [target],
                    'type': ['Node'],
                })
            else:
                common_node_list[target_node]['text'].append(target)
                common_node_list[target_node]['type'].append('Node')

            # adding the edge
            common_edge_list.append((source_node, target_node, edge_triplet[1]))

    # creating the graph
    graph = DiGraph()

    # adding the nodes
    for node_id, node_dict in enumerate(common_node_list):
        graph.add_node(
            node_id,
            weight=len(node_dict['text']) / len(graph_text_list), 
            text=find_representative_sentence(node_dict['text']),
            type=mode(node_dict['type'])
        )

    # adding the edges
    for edge in common_edge_list:
        
        # check if the edge already exists
        if graph.has_edge(edge[0], edge[1]):
            graph[edge[0]][edge[1]]['stance'].append(edge[2])
            graph[edge[0]][edge[1]]['weight'] += (1 / len(graph_text_list))
        else:
            graph.add_edge(
                edge[0],
                edge[1],
                stance=[edge[2]],
                weight=(1 / len(graph_text_list))
            )

    # assigning the mode of the stance
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['stance'] = mode(graph[edge[0]][edge[1]]['stance'])

    return graph


def mld_exact(
    graph: DiGraph,
    tree_constraints: bool = False,
    dag_constraints: bool = True,
    edge_threshold: float = 0.5,
    node_constraints: bool = False,
    node_threshold: float = 0.3,
) -> DiGraph:
    '''
        Applies the exact MLD formulation
    '''
    
    # creating the problem
    optimization_prob = LpProblem('MLD', LpMaximize)

    # creating the variables
    num_nodes = len(graph.nodes())
    all_pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes)]
    x = LpVariable.dicts('x', all_pairs, lowBound=0, upBound=1, cat=LpInteger)

    # creating variables for the dag constraints
    if dag_constraints:
        b = LpVariable.dicts('b', all_pairs, lowBound=0, upBound=1, cat=LpInteger)
    
    # creating variables for the nodes
    if node_constraints:
        y = LpVariable.dicts('y', range(num_nodes), lowBound=0, upBound=1, cat=LpInteger)

    # creating the objective function
    optimization_summands = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_weight = graph[i][j]['weight'] if graph.has_edge(i, j) else 0.0
            optimization_summands.append(x[(i, j)] * (edge_weight - edge_threshold))

        # adding the node constraints if node_constraints is True
        if node_constraints:
            node_weight = graph.nodes[i]['weight'] if graph.has_node(i) else 0.0
            optimization_summands.append(y[i] * (node_weight - node_threshold))
    optimization_prob += (lpSum(optimization_summands), 'Objective Function')

    # creating constraints between the xij and yi and yj
    if node_constraints:
        for i in range(num_nodes):
            for j in range(num_nodes):
                optimization_prob += (2 * x[(i, j)] - y[i] - y[j] <= 0, 'Node Constraint {}, {}'.format(i, j))

    # tree constraints
    if tree_constraints:
        edge_count_constraint = []
        for i in range(num_nodes):
            parent_constraint = []
            for j in range(num_nodes):
                parent_constraint.append(x[(i, j)])
                edge_count_constraint.append(x[(i, j)])
            optimization_prob += (lpSum(parent_constraint) <= 1, 'Parent Constraint {}'.format(i))
        optimization_prob += (lpSum(edge_count_constraint) <= num_nodes - 1, 'Edge Count Constraint')

    # no self loop constraint
    no_self_loop = []
    for i in range(num_nodes):
        no_self_loop.append(x[(i, i)])
    optimization_prob += (lpSum(no_self_loop) == 0, 'No Self Loop Constraint')

    # dag constraints
    if dag_constraints:

        # relation between xij and bij
        for i in range(num_nodes):
            for j in range(num_nodes):
                optimization_prob += (x[(i, j)] - b[(i, j)] <= 0, 'Relation Constraint 1 {}, {}'.format(i, j))

        # transtivity constraint on bik - bik = 1 if bij = 1 and bjk = 1
        for i in range(num_nodes):
            for j in range(num_nodes):
                for k in range(num_nodes):
                    optimization_prob += (b[(i, k)] - b[(i, j)] - b[(j, k)] >= -1, 'Relation Constraint 2 {}, {}, {}'.format(i, j, k))

        # no self path constraint
        for i in range(num_nodes):
            optimization_prob += (b[(i, i)] == 0, 'No Self Path Constraint {}'.format(i))

    # solving the problem
    optimization_prob.solve()

    # edges to delete
    edges_to_delete = []
    for edge in graph.edges():
        if x[edge].varValue == 0:
            edges_to_delete.append(edge)
    graph.remove_edges_from(edges_to_delete)

    # nodes to delete
    if not node_constraints:
        nodes_to_delete = []
        for node in graph.nodes():
            if graph.degree(node) == 0:
                nodes_to_delete.append(node)
    else:
        nodes_to_delete = []
        for node in graph.nodes():
            if y[node].varValue == 0:
                nodes_to_delete.append(node)
    graph.remove_nodes_from(nodes_to_delete)

    return graph

def automated_estimation_of_parameters(
    few_shot_prompt_str: str,
    sample_length: int = 50
):
    '''
        For automatically estimating the node and edge threshold
    '''

    # processing the few shot prompt
    few_shot_instances = few_shot_prompt_str.split(END)

    # constructing the actual few shot prompt
    few_shot_prompt = [{'role': 'system', 'content': 'You are an AI language model created by OpenAI. You will provide information and answer questions to the best of your knowledge and abilities.'}]
    for few_shot_instance in few_shot_instances:
        if len(few_shot_instance.split(SEPARATOR)) == 2:
            few_shot_prompt.append({'role': 'user', 'content': few_shot_instance.split(SEPARATOR)[0]})
            few_shot_prompt.append({'role': 'assistant', 'content': few_shot_instance.split(SEPARATOR)[0] + few_shot_instance.split(SEPARATOR)[1]})

    # storing different samples for each of the examples
    num_datapoints = len(few_shot_prompt) // 2
    inferred_samples = [[] for _ in range(num_datapoints)]
    input_samples = []
    actual_samples = []

    # getting the converter
    converter = ConverterFactory.get_converter('explagraphs-literal')

    # iterate through the few_shot_prompt
    for sample_index in range(num_datapoints):

        # logging the sample_index
        print('Data Index: {}'.format(sample_index))

        # constructing the few_shot_prompt
        input_text = few_shot_prompt[2 * sample_index + 1]['content']
        output_text = few_shot_prompt[2 * sample_index + 2]['content']
        input_samples.append(input_text)
        actual_samples.append(converter.python_to_graph(output_text, logging=False))
        current_few_shot_prompt = few_shot_prompt[:2 * sample_index + 1] + few_shot_prompt[2 * sample_index + 3:]

        # creating multiple inferrences for the same input
        for _ in tqdm(range(sample_length)):
            current_message = current_few_shot_prompt + [{'role': 'user', 'content': input_text}]
            response = openai_chat_api(current_message, temperature=0.9)
            if response is not None:
                python_output_code = response.choices[0].message.content
                try:
                    inferred_samples[sample_index].append(converter.python_to_graph(python_output_code, logging=False)['graph'])
                except:
                    inferred_samples[sample_index].append('(random; capable of; random)')

    # estimating the parameters for every 10 samples
    for num_samples in range(10, sample_length + 1, 10):

        # logging the num_samples
        print('Number of Samples: {}'.format(num_samples))

        # storing the aggregated graphs
        aggregated_graphs = []

        # iterate through the samples
        for sample_index in range(num_datapoints):

            # aggregating the graph texts
            graph = aggregate_graph_texts(inferred_samples[sample_index][:num_samples])
            aggregated_graphs.append(graph)

        # estimating the edge_threshold and node_threshold
        edge_threshold = threshold_grid_search(aggregated_graphs, actual_samples, input_samples, num_samples=num_samples)

        # logging the edge_threshold and node_threshold
        print('Edge Threshold: {}'.format(edge_threshold))


def threshold_grid_search(
    inference_graphs: List[DiGraph],
    actual_samples: List[Dict[str, str]],
    input_text_list: List[str],
    num_samples: int = 10,
):
    '''
        Grid search over each value in the set [0, 1 / num_samples, 2 / num_samples, ..., 1]
        to determine the edge_threshold
    '''

    # storing the edge_threshold_list, best_threshold and best_score
    threshold_list = [i / num_samples for i in range(1, num_samples + 1)]
    edge_best_threshold = None
    edge_best_score = 0.0

    # iterate through the threshold_list
    for threshold in threshold_list:

        # copying the inference_graphs
        inference_copy_graphs = [graph.copy() for graph in inference_graphs]

        # applying the mld algorithm
        mld_graphs = [mld_exact(graph, edge_threshold=threshold) for graph in inference_copy_graphs]

        # stringifying the graphs
        mld_graph_strings = [graph_to_string(graph) for graph in mld_graphs]

        # computing the performance
        io_data = []
        for mld_graph_string, actual_sample in zip(mld_graph_strings, actual_samples):
            io_data.append({
                'generated_graph': {
                    'graph': mld_graph_string,
                    'stance': actual_sample['stance'],
                },
                'reference_graph': actual_sample
            })
        result_data = generate_results(io_data, compute_bert_score=False)

        # updating the best_threshold and best_score
        if result_data['structural_correctness_accuracy'] > edge_best_score:
            edge_best_threshold = threshold
            edge_best_score = result_data['structural_correctness_accuracy']

    return edge_best_threshold
            
def aggregator(
    path_pattern: str,
    output_path: str,
    sample_length: int = 10,
    overlap_threshold: float = 0.7,
    edge_threshold: float = 0.2,
    node_threshold: float = 0.3,
    node_constraints: bool = False,
    method: str = 'mld_simplified',
    estimate_parameters: bool = False,
    prompt_file: str = 'data/explagraphs/prompt.txt',
):
    '''
        Aggregates the graph texts from the dir_name using the method
    '''

    # if the parameters needs to be estimated
    if estimate_parameters:
        with open(prompt_file, "r") as f:
            prompt = f.read()
        automated_estimation_of_parameters(prompt, sample_length)

    # getting all the files with the path_pattern
    file_list = glob.glob(path_pattern)
    file_list = file_list[:sample_length]

    # store all the data from the first file
    data_list = []
    with jsonlines.open(file_list[0], 'r') as f:
        for data in f:
            data_list.append(data)
    data_length = len(data_list)

    # saving all samples for each datapoint
    graph_samples = [[] for _ in range(data_length)]
    for file_path in file_list:
        with jsonlines.open(file_path, 'r') as f:
            for sample_index, data in enumerate(f):
                graph_samples[sample_index].append(data['generated_graph']['graph'])

    # aggregating the graph samples
    for graph_list, data in tqdm(zip(graph_samples, data_list)):

        # aggregate the graph texts
        graph = aggregate_graph_texts(graph_list, overlap_threshold)

        # simplify the graph
        if method == 'mld_simplified':
            graph = mld_exact(graph, edge_threshold=edge_threshold, tree_constraints=False, dag_constraints=False, node_constraints=node_constraints, node_threshold=node_threshold)
        elif method == 'mld_tree':
            graph = mld_exact(graph, tree_constraints=True, dag_constraints=True, edge_threshold=edge_threshold, node_constraints=node_constraints, node_threshold=node_threshold)
        elif method == 'mld_dag':
            graph = mld_exact(graph, tree_constraints=False, dag_constraints=True, edge_threshold=edge_threshold, node_constraints=node_constraints, node_threshold=node_threshold)
        elif method == 'non_mld_tree':
            graph = mld_exact(graph, tree_constraints=True, dag_constraints=True, edge_threshold=0.0, node_constraints=node_constraints, node_threshold=node_threshold)
        elif method == 'non_mld_dag':
            graph = mld_exact(graph, tree_constraints=False, dag_constraints=True, edge_threshold=0.0, node_constraints=node_constraints, node_threshold=node_threshold)

        # saving the graph
        graph_string = graph_to_string(graph)
        data['generated_graph']['graph'] = graph_string

    # saving the data as jsonl
    with jsonlines.open(output_path, 'w') as f:
        for data in data_list:
            f.write(data)


if __name__ == '__main__':

    # creating the argument parser
    parser = ArgumentParser()
    parser.add_argument('--path_patern', type=str, default='data/explagraphs/output_llama_30_*.jsonl')
    parser.add_argument('--output_path', type=str, default='data/explagraphs/aggregated_output.jsonl')
    parser.add_argument('--sample_length', type=int, default=10)
    parser.add_argument('--overlap_threshold', type=float, default=0.8)
    parser.add_argument('--edge_threshold', type=float, default=0.2)
    parser.add_argument('--node_threshold', type=float, default=0.3)
    parser.add_argument('--method', type=str, default='mld_simplified')
    parser.add_argument('--node_constraints', action='store_true')
    parser.add_argument('--estimate_parameters', action='store_true')
    parser.add_argument('--prompt_file', type=str, default='data/explagraphs/prompt.txt')
    args = parser.parse_args()

    aggregator(
        path_pattern=args.path_patern,
        output_path=args.output_path,
        sample_length=args.sample_length,
        overlap_threshold=args.overlap_threshold,
        edge_threshold=args.edge_threshold,
        node_threshold=args.node_threshold,
        method=args.method,
        node_constraints=args.node_constraints,
        estimate_parameters=args.estimate_parameters,
        prompt_file=args.prompt_file,
    )