import os
import json
import numpy as np
import networkx as nx
from pulp import *
from argparse import ArgumentParser
from statistics import mode
from typing import List, Dict, Union
from networkx import DiGraph
from tqdm import tqdm
from prompt_scheme import code_prompt_postprocessor, code_prompt_nx
from executor import openai_chat_api
from evaluator import node_level_evaluation_standardized, edge_level_performance_threshold

def compute_threshold(
    sent_a = str,
    sent_b = str
):
    '''
        Computes the set overlap between the tokens in the two sentences
    '''
    tokens_a = set(sent_a.split())
    tokens_b = set(sent_b.split())
    return len(tokens_a.intersection(tokens_b)) / max(len(tokens_a), len(tokens_b), 1)

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

def join_major_claims(graph: DiGraph) -> DiGraph:
    '''
        Helper function that merge all the major claims determined by voting
    '''
    
    # get the major claims
    major_claim_ids = []
    for node_id in graph.nodes():
        if graph.nodes[node_id]['type'] == 'MajorClaim':
            major_claim_ids.append(node_id)
    
    # check if the major claims are present
    if len(major_claim_ids) == 0:
        return graph

    # create a new node with the text as the concatenation of the major claims
    new_node_id = len(graph.nodes())
    new_node_text = [graph.nodes[node_id]['text'] for node_id in major_claim_ids]
    new_weight = [graph.nodes[node_id]['weight'] for node_id in major_claim_ids]
    max_weight = max(new_weight) if len(new_weight) > 0 else 0.0
    graph.add_node(new_node_id, weight=max_weight, text=new_node_text, type='MajorClaim')

    # assign max weighted edge to the new node
    for node_id in graph.nodes():
        for major_claim_id in major_claim_ids:
            
            # edge not present between node_id and new_node_id
            if graph.has_edge(node_id, major_claim_id) and not graph.has_edge(node_id, new_node_id):
                graph.add_edge(node_id, new_node_id, stance=graph[node_id][major_claim_id]['stance'], weight=graph[node_id][major_claim_id]['weight'])
            
            # edge present between node_id and new_node_id and the weight of the edge is greater than the current edge
            elif graph.has_edge(node_id, major_claim_id) and graph[node_id][major_claim_id]['weight'] > graph[node_id][new_node_id]['weight']:
                graph[node_id][new_node_id]['stance'] = graph[node_id][major_claim_id]['stance']
                graph[node_id][new_node_id]['weight'] = graph[node_id][major_claim_id]['weight']
                
    # remove the major claims
    graph.remove_nodes_from(major_claim_ids)

    # converting it into standard form
    return nx.convert_node_labels_to_integers(graph)


def aggregate_graph_texts(
    graph_text_list: List[str],
    threshold: float = 0.7,
    major_claim_merging: bool = True
) -> DiGraph:
    '''
        Aggregate the graph texts from the graph_text_list
    '''
    
    common_node_list = []
    common_edge_list = []
    total_node_weight = 0.0
    total_edge_weight = 0.0

    # iterate through the graph_text_list
    for graph_index, graph_text in enumerate(graph_text_list):

        # will be used to map the id to the actual id
        node_mapper = {}
        current_major_claims = []
        
        # processing the graph_text
        node_declaration, edge_declarations = code_prompt_postprocessor(graph_text)

        # updating the total_node_weight and total_edge_weight
        total_node_weight += 1
        total_edge_weight += 1

        # iterate through the node_declaration
        for node_id, node_dict in node_declaration.items():

            # check common
            common_node = check_common(common_node_list, node_dict, threshold)
            if common_node is None:
                node_mapper[node_id] = len(common_node_list)
                common_node_list.append({
                    'text': [node_dict['text']],
                    'type': [node_dict['type']],
                    'weight': [1.0]
                })
            else:
                node_mapper[node_id] = common_node
                common_node_list[node_mapper[node_id]]['text'].append(node_dict['text'])
                common_node_list[node_mapper[node_id]]['type'].append(node_dict['type'])
                common_node_list[node_mapper[node_id]]['weight'].append(1.0)

            # checking if the node is a major claim
            if node_dict['type'] == 'MajorClaim':
                current_major_claims.append(node_id)

        # iterate through the edge_declarations
        for edge_dict in edge_declarations:
            from_node = edge_dict['from_node']
            to_node = edge_dict['to_node']
            stance = edge_dict['stance']
            if from_node in node_mapper and to_node in node_mapper:
                common_edge_list.append((node_mapper[from_node], node_mapper[to_node], stance, 1.0))

                # if the to_node lies in current_major_claims
                if to_node in current_major_claims:
                    for major_claim in current_major_claims:
                        if major_claim != to_node:
                            common_edge_list.append((node_mapper[from_node], node_mapper[major_claim], stance, 1.0))

    # creating the graph
    graph = DiGraph()

    # adding the nodes
    for node_id, node_dict in enumerate(common_node_list):

        # computing the weight
        weight = sum(node_dict['weight']) / total_node_weight

        # adding the node
        graph.add_node(
            node_id,
            weight=weight, 
            text=find_representative_sentence(node_dict['text']),
            type=mode(node_dict['type'])
        )

    # adding the edges
    for edge in common_edge_list:

        # computing the weight increment
        weight_increment = edge[3] / total_edge_weight
        
        # check if the edge already exists
        if graph.has_edge(edge[0], edge[1]):
            graph[edge[0]][edge[1]]['stance'].append(edge[2])
            graph[edge[0]][edge[1]]['weight'] += weight_increment
        else:
            graph.add_edge(
                edge[0],
                edge[1],
                stance=[edge[2]],
                weight=weight_increment
            )

    # assigning the mode of the stance
    for edge in graph.edges():
        graph[edge[0]][edge[1]]['stance'] = mode(graph[edge[0]][edge[1]]['stance'])

    # joining the major claims
    if major_claim_merging:
        final_graph = join_major_claims(graph)
        return final_graph
    return graph

def mld_simplified(graph: DiGraph, edge_threshold: float = 0.5) -> DiGraph:
    '''
        Simplifies the graph using the MLD algorithm
    '''
    
    # remove edges whose weight is less than edge_threshold
    edges_to_remove = []
    for edge in graph.edges():
        if graph[edge[0]][edge[1]]['weight'] < edge_threshold:
            edges_to_remove.append(edge)
    graph.remove_edges_from(edges_to_remove)

    # remove singleton nodes
    nodes_to_remove = []
    for node in graph.nodes():
        if graph.degree(node) == 0:
            nodes_to_remove.append(node)
    graph.remove_nodes_from(nodes_to_remove)

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

        # transtivity constraint on bik :- bik = 1 if bij = 1 and bjk = 1
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


def edge_threshold_from_few_shot(few_shot_prompt: List[Dict[str, str]]) -> float:
    '''
        Automatically estimates the edge threshold using information theory
    '''

    # to store the input output pairs
    exhaustive_num_nodes = []
    actual_num_nodes = []

    # iterate through the few_shot_prompt
    for prompt_dict in few_shot_prompt:
        
        # actual text input
        if prompt_dict['role'] == 'user':
            num_elements = len(prompt_dict['content'].split())
            exhaustive_num_nodes.append(num_elements * (num_elements + 1) / 2)
        
        # graph text output
        if prompt_dict['role'] == 'assistant':
            actual_num_nodes.append(str(prompt_dict['content']).count('='))

    # checking if the exhaustive_num_nodes and actual_num_nodes are of the same length
    if len(exhaustive_num_nodes) != len(actual_num_nodes):
        raise ValueError('The length of exhaustive_num_nodes and actual_num_nodes should be the same')
    
    # computing the edge_threshold
    N = np.log2(exhaustive_num_nodes)
    n = np.log2(actual_num_nodes)

    return np.mean(n / (n + N))


def automated_estimation_of_parameters(
    few_shot_prompt: List[Dict[str, str]],
    sample_length: int = 50
):
    '''
        For automatically estimating the node and edge threshold
    '''
    
    # storing different samples for each of the examples
    num_datapoints = len(few_shot_prompt) // 2
    inferred_samples = [[] for _ in range(num_datapoints)]
    input_samples = []
    actual_samples = []

    # iterate through the few_shot_prompt
    for sample_index in range(num_datapoints):

        # logging the sample_index
        print('Sample Index: {}'.format(sample_index))

        # constructing the few_shot_prompt
        input_text = few_shot_prompt[2 * sample_index + 1]['content']
        output_text = few_shot_prompt[2 * sample_index + 2]['content']
        input_samples.append(input_text)
        actual_samples.append(output_text)
        current_few_shot_prompt = few_shot_prompt[:2 * sample_index + 1] + few_shot_prompt[2 * sample_index + 3:]

        # creating multiple inferrences for the same input
        for _ in tqdm(range(sample_length)):
            current_message = current_few_shot_prompt + [{'role': 'user', 'content': input_text}]
            response = openai_chat_api(current_message, temperature=0.9)
            if response is not None:
                inferred_samples[sample_index].append(response.choices[0].message.content)

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
        edge_threshold, node_threshold = threshold_grid_search(aggregated_graphs, actual_samples, input_samples, num_samples=num_samples)

        # logging the edge_threshold and node_threshold
        print('Edge Threshold: {}'.format(edge_threshold))
        print('Node Threshold: {}'.format(node_threshold))
        

def threshold_grid_search(
    inference_graphs: List[DiGraph],
    actual_samples: List[str],
    input_text_list: List[str],
    num_samples: int = 10,
):
    '''
        Grid search over each value in the set [0, 1 / num_samples, 2 / num_samples, ..., 1]
        to determine the edge_threshold
    '''

    # storing the edge_threshold_list, best_threshold and best_score
    threshold_list = [i / num_samples for i in range(1, (num_samples + 1) // 2)]
    edge_best_threshold = None
    edge_best_score = 0.0
    node_best_threshold = None
    node_best_score = 0.0

    # getting the nodes and edges for actual_samples
    actual_graph_information = [code_prompt_postprocessor(sample) for sample in actual_samples]
    actual_nodes = [node_declaration for node_declaration, _ in actual_graph_information]
    actual_edges = [[[obj['from'], obj['stance'], obj['to']] for obj in edge_declarations] for _, edge_declarations in actual_graph_information]

    # iterate through the threshold_list
    for threshold in threshold_list:

        # copying the inference_graphs
        edge_inference_graphs = [graph.copy() for graph in inference_graphs]
        node_inference_graphs = [graph.copy() for graph in inference_graphs]

        # removing the edges whose weight is less than threshold
        for graph in edge_inference_graphs:
            edges_to_remove = []
            for edge in graph.edges():
                if graph[edge[0]][edge[1]]['weight'] < threshold:
                    edges_to_remove.append(edge)
            graph.remove_edges_from(edges_to_remove)

        # computing the edge_level_performance_threshold for the current threshold
        edge_inference_information = [code_prompt_postprocessor(code_prompt_nx(graph)) for graph in edge_inference_graphs]
        inferred_edges = [[[obj['from'], obj['stance'], obj['to']] for obj in edge_declarations] for _, edge_declarations in edge_inference_information]
        current_edge_score = edge_level_performance_threshold(inferred_edges, actual_edges)['f1']
        if current_edge_score > edge_best_score:
            edge_best_score = current_edge_score
            edge_best_threshold = threshold

        # removing the nodes whose weight is less than threshold
        for graph in node_inference_graphs:
            nodes_to_remove = []
            for node in graph.nodes():
                if graph.nodes[node]['weight'] < threshold:
                    nodes_to_remove.append(node)
            graph.remove_nodes_from(nodes_to_remove)

        # computing the node_level_evaluation_standardized for the current threshold
        node_inference_information = [code_prompt_postprocessor(code_prompt_nx(graph)) for graph in node_inference_graphs]
        inferred_nodes = [node_declaration for node_declaration, _ in node_inference_information]
        current_node_score = node_level_evaluation_standardized(actual_nodes, inferred_nodes, input_text_list)['Detection_macro']['f1']
        if current_node_score > node_best_score:
            node_best_score = current_node_score
            node_best_threshold = threshold

        # logging the current threshold and corresponding scores
        print('Threshold: {}'.format(threshold))
        print('Edge Score: {}'.format(current_edge_score))
        print('Node Score: {}'.format(current_node_score))

    return edge_best_threshold, node_best_threshold
            
def aggregator(
    dir_name: str = 'test',
    target_dir_name: str = 'aggregated_test',
    sample_length: int = 10,
    overlap_threshold: float = 0.7,
    edge_threshold: float = 0.2,
    node_threshold: float = 0.3,
    node_constraints: bool = False,
    method: str = 'mld_simplified',
    estimate_edge_threshold: bool = False,
    estimate_parameters: bool = False
):
    '''
        Aggregates the graph texts from the dir_name using the method
    '''

    # reading the config file
    with open(os.path.join(dir_name, 'config.json'), 'r') as f:
        config = json.load(f)
    data_length = config['test_size']

    # if the parameters needs to be estimated
    if estimate_parameters:
        automated_estimation_of_parameters(config['few_shot_prompt'], sample_length=sample_length)
        return

    # estimating the edge_threshold
    if estimate_edge_threshold:
        edge_threshold = edge_threshold_from_few_shot(config['few_shot_prompt'])

    # creating target_dir_name and saving the config file
    if not os.path.isdir(target_dir_name):
        os.mkdir(target_dir_name)
    with open(os.path.join(target_dir_name, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    for sample_index in tqdm(range(data_length)):

        # aggregating the graph for the data corresponding to the sample_index
        sample_name = 'result_{}.json'.format(sample_index)

        # sampling
        graph_text_list = []
        input_text = None
        ground_truth = None
        sample_dir_list = []
        for sample_dir in os.listdir(dir_name):
        
            # get the actual dir
            actual_dir = os.path.join(dir_name, sample_dir)
            if os.path.isdir(actual_dir):
                graph_file_path = os.path.join(actual_dir, sample_name)
                with open(graph_file_path, 'r') as f:
                    json_data = json.load(f)
                    if json_data['inference'] is not None:
                        graph_text_list.append(json_data['inference'])

                        # saving the input_text and ground_truth
                        if input_text is None:
                            input_text = json_data['input']
                        if ground_truth is None:
                            ground_truth = json_data['output']

            if len(graph_text_list) >= sample_length:
                break
            sample_dir_list.append(actual_dir)

        # aggregate the graph texts
        graph = aggregate_graph_texts(
            graph_text_list,
            overlap_threshold,
            major_claim_merging=True
        )

        # simplify the graph
        try:
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
        except:
            print('Error in sample: {}'.format(sample_name))

        # saving the graph
        result_json = {
            'input': input_text,
            'output': ground_truth,
            'inference': code_prompt_nx(graph)
        }

        with open(os.path.join(target_dir_name, sample_name), 'w') as f:
            json.dump(result_json, f, indent=4)


if __name__ == '__main__':

    # creating the argument parser
    parser = ArgumentParser()
    parser.add_argument('--dir_name', type=str, default='/home/inair/data/ArgumentAnnotatedEssays-2.0/final_output/code_prompt_7_0_sampling_llama_data/50_50')
    parser.add_argument('--target_dir_name', type=str, default='/home/inair/data/ArgumentAnnotatedEssays-2.0/final_output/test')
    parser.add_argument('--sample_length', type=int, default=10)
    parser.add_argument('--overlap_threshold', type=float, default=0.7)
    parser.add_argument('--edge_threshold', type=float, default=0.2)
    parser.add_argument('--node_threshold', type=float, default=0.3)
    parser.add_argument('--method', type=str, default='mld_simplified')
    parser.add_argument('--estimate_edge_threshold', action='store_true')
    parser.add_argument('--node_constraints', action='store_true')
    parser.add_argument('--estimate_parameters', action='store_true')
    args = parser.parse_args()

    aggregator(
        dir_name=args.dir_name,
        target_dir_name=args.target_dir_name,
        sample_length=args.sample_length,
        overlap_threshold=args.overlap_threshold,
        edge_threshold=args.edge_threshold,
        node_threshold=args.node_threshold,
        method=args.method,
        estimate_edge_threshold=args.estimate_edge_threshold,
        node_constraints=args.node_constraints,
        estimate_parameters=args.estimate_parameters
    )