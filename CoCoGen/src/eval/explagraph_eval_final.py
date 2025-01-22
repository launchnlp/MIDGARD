import argparse
import jsonlines
import networkx as nx
import numpy as np
from typing import List, Dict
from src.eval.graph_matching import split_to_edges, get_tokens, get_bleu_rouge, get_bert_score, get_ged

relations = ['antonym of', 'synonym of', 'at location', 'not at location', 'capable of', 'not capable of', 'causes', 'not causes', 'created by', 'not created by', 'is a', 'is not a', 'desires', 'not desires', 'has subevent', 'not has subevent', 'part of', 'not part of', 'has context', 'not has context', 'has property', 'not has property', 'made of', 'not made of', 'receives action', 'not receives action', 'used for', 'not used for']


def is_edge_count_correct(edges):
    if len(edges) < 3:
        return False
    else:
        return True


def is_graph(edges):
    for edge in edges:
        components = edge.split("; ")
        if len(components) != 3:
            return False

    return True


def is_edge_structure_correct(edges, relations):
    for edge in edges:
        components = edge.split("; ")
        if components[0] == "" or len(components[0].split(" ")) > 3:
            return False
        if components[1] not in relations:
            return False
        if components[2] == "" or len(components[2].split(" ")) > 3:
            return False

    return True


def two_concepts_belief_argument(edges, belief, argument):
    belief_concepts = {}
    argument_concepts = {}
    for edge in edges:
        components = edge.split("; ")
        if components[0] in belief:
            belief_concepts[components[0]] = True

        if components[2] in belief:
            belief_concepts[components[2]] = True

        if components[0] in argument:
            argument_concepts[components[0]] = True

        if components[2] in argument:
            argument_concepts[components[2]] = True

    if len(belief_concepts) < 2 or len(argument_concepts) < 2:
        return False
    else:
        return True


def is_connected_DAG(edges):
    g = nx.DiGraph()
    for edge in edges:
        components = edge.split("; ")
        g.add_edge(components[0], components[2])

    return nx.is_weakly_connected(g) and nx.is_directed_acyclic_graph(g)


def get_max(first_precisions, first_recalls, first_f1s, second_precisions, second_recalls, second_f1s):
    max_indices = np.argmax(np.concatenate((np.expand_dims(first_f1s, axis=1),
                                            np.expand_dims(second_f1s, axis=1)), axis=1), axis=1)

    precisions = np.concatenate((np.expand_dims(first_precisions, axis=1),
                                 np.expand_dims(second_precisions, axis=1)), axis=1)
    precisions = np.choose(max_indices, precisions.T)

    recalls = np.concatenate((np.expand_dims(first_recalls, axis=1),
                              np.expand_dims(second_recalls, axis=1)), axis=1)
    recalls = np.choose(max_indices, recalls.T)

    f1s = np.maximum(first_f1s, second_f1s)

    return precisions, recalls, f1s

def generate_results(data_list: List[Dict], compute_bert_score: bool = True) -> Dict[str, float]:
    '''
        Generates the results in a dictionary format
    '''
    
    stance_correct_count = 0
    structurally_correct_graphs_count = 0
    structurally_correct_gold_graphs, structurally_correct_pred_graphs = [], []
    overall_ged = 0.
    for data in data_list:

        # getting the prediction
        pred_stance = data['generated_graph']['stance']
        pred_graph = data['generated_graph']['graph'].lower()

        assert pred_stance in ["support", "counter"]

        # getting the gold
        belief = data['reference_graph']['belief'].lower()
        argument = data['reference_graph']['argument'].lower()
        gold_stance = data['reference_graph']['stance']
        gold_graph = data['reference_graph']['graph'].lower()

        # Check for Stance Correctness first
        if pred_stance == gold_stance:
            stance_correct_count += 1
            edges = pred_graph[1:-1].split(")(")
            # Check for Structural Correctness of graphs
            if is_edge_count_correct(edges) and is_graph(edges) and is_edge_structure_correct(edges,
                                                                                              relations) and two_concepts_belief_argument(
                    edges, belief, argument) and is_connected_DAG(edges):
                structurally_correct_graphs_count += 1

                # Save the graphs for Graph Matching or Semantic Correctness Evaluation
                structurally_correct_gold_graphs.append(gold_graph)
                structurally_correct_pred_graphs.append(pred_graph)

                # Compute GED
                ged = get_ged(gold_graph, pred_graph)

            else:
                # GED needs to be computed as the upper bound for structurally incorrect graphs
                ged = get_ged(gold_graph)
        else:
            # GED also needs to be computed as the upper bound for samples with incorrect stance
            ged = get_ged(gold_graph)

        overall_ged += ged


    # Evaluate for Graph Matching
    gold_edges = split_to_edges(structurally_correct_gold_graphs)
    second_gold_edges = None
    pred_edges = split_to_edges(structurally_correct_pred_graphs)

    gold_tokens, pred_tokens, _ = get_tokens(gold_edges, pred_edges, second_gold_edges)

    precisions_rouge, recalls_rouge, f1s_rouge, precisions_bleu, recalls_bleu, f1s_bleu = get_bleu_rouge(
        gold_tokens, pred_tokens, gold_edges, pred_edges)

    if compute_bert_score:
        precisions_BS, recalls_BS, f1s_BS = get_bert_score(gold_edges, pred_edges)

    # storing the results
    result_dict = {}
    result_dict['stance_accuracy'] = stance_correct_count / len(data_list)
    result_dict['structural_correctness_accuracy'] = structurally_correct_graphs_count / len(data_list)
    result_dict['graph_edit_distance'] = overall_ged / len(data_list)
    result_dict['ged'] = overall_ged / len(data_list)
    if compute_bert_score:
        result_dict['g_bertscore_f1'] = f1s_BS.sum() / len(data_list)

    print(f'Stance Accuracy (SA): {stance_correct_count / len(data_list):.4f}')
    print(f'Structural Correctness Accuracy (StCA): {structurally_correct_graphs_count / len(data_list):.4f}')

    print(f'G-BLEU Precision: {precisions_bleu.sum() / len(data_list):.4f}')
    print(f'G-BLEU Recall: {recalls_bleu.sum() / len(data_list):.4f}')
    print(f'G-BLEU F1: {f1s_bleu.sum() / len(data_list):.4f}\n')

    print(f'G-Rouge Precision: {precisions_rouge.sum() / len(data_list):.4f}')
    print(f'G-Rouge Recall Score: {recalls_rouge.sum() / len(data_list):.4f}')
    print(f'G-Rouge F1 Score: {f1s_rouge.sum() / len(data_list):.4f}')

    if compute_bert_score:
        print(f'G-BertScore Precision Score: {precisions_BS.sum() / len(data_list):.4f}')
        print(f'G-BertScore Recall Score: {recalls_BS.sum() / len(data_list):.4f}')
        print(f'G-BertScore F1 Score: {f1s_BS.sum() / len(data_list):.4f}\n')

    print(f'Graph Edit Distance (GED): {overall_ged / len(data_list):.4f}\n')

    return result_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default=None, type=str, required=True)

    args = parser.parse_args()

    # reading the data
    data_list = []
    with jsonlines.open(args.output_file) as reader:
        for obj in reader:
            data_list.append(obj)

    # generating the results
    result_dict = generate_results(data_list)

    