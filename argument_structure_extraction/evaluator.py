import os
import json
import argparse
import numpy as np
from pprint import pprint
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Optional
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from prompt_scheme import code_prompt_postprocessor, seq_label_generator
from graph_matching import split_to_edges, get_tokens, get_bleu_rouge

# temporary code
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

def confusion_matrix_analysis(y_true, y_pred, labels=['MajorClaim', 'Claim', 'Premise', 'None']):

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    
    # computing precision and recall for each class
    result_dict = {}
    for index, label in enumerate(labels):
        precision = matrix[index, index] / (np.sum(matrix[:, index]) + 1e-11)
        recall = matrix[index, index] / (np.sum(matrix[index, :]) + 1e-11)
        f1 = 2 * precision * recall / (precision + recall + 1e-11)
        if label != 'None':
            result_dict[label] = {
                'precision': precision, 'recall': recall, 'f1': f1
            }

    # computing the non none f1 score
    y_true_non_none = np.array(y_true) != 'None'
    y_pred_non_none = np.array(y_pred) != 'None'
    result_dict['Detection'] = {
        'precision': precision_score(y_true_non_none, y_pred_non_none),
        'recall': recall_score(y_true_non_none, y_pred_non_none),
        'f1': f1_score(y_true_non_none, y_pred_non_none)
    }

    return result_dict, matrix


def node_level_evaluation_standardized(
    gt_node_declarations: List[Dict[str, dict]],
    inference_node_declarations: List[Dict[str, dict]],
    text_list: List[str],
) -> Dict[str, dict]:
    '''
        Evaluate the node level predictions with partial rewards for partial matches
    '''
    
    # for computing the final results
    fine_grained_labels = []
    fine_grained_inferences = []
    detection_labels = []
    detection_inferences = []
    node_types = set()

    # ampere style evaluation
    fine_grained_ampere_labels = []
    fine_grained_ampere_inferences = []

    # iterating over the instances
    for gt_nodes, inference_nodes, text in tqdm(zip(gt_node_declarations, inference_node_declarations, text_list)):

        # getting the labels
        fine_grained_label, detection_label, node_type = seq_label_generator(gt_nodes, text)
        fine_grained_labels += fine_grained_label
        detection_labels += detection_label
        node_types = node_types.union(node_type)

        # getting the inferences
        fine_grained_inference, detection_inference, _ = seq_label_generator(inference_nodes, text)
        fine_grained_inferences += fine_grained_inference
        detection_inferences += detection_inference

        # ampere style evaluation
        fine_grained_ampere_label = ['B' + t[1:] if t.startswith('I') else t for t in fine_grained_label]
        fine_grained_ampere_inference = ['B' + t[1:] if t.startswith('I') else t for t in fine_grained_inference]
        fine_grained_ampere_labels += fine_grained_ampere_label
        fine_grained_ampere_inferences += fine_grained_ampere_inference

    # to be returned
    result_dict = dict()

    # exact evaluation using seqeval metric
    result_dict['exact'] = classification_report(
        [fine_grained_labels],
        [fine_grained_inferences],
        output_dict=True,
        digits=4
    )

    # computing fine-grained identification results
    relevant_labels = []
    result_dict['overall_f1_macro'] = {
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0
    }
    for node_type in node_types:
        result_dict[node_type] = {
            'precision': precision_score(
                fine_grained_labels,
                fine_grained_inferences,
                labels=['B-{}'.format(node_type), 'I-{}'.format(node_type)],
                average='micro'
            ),
            'recall': recall_score(
                fine_grained_labels,
                fine_grained_inferences,
                labels=['B-{}'.format(node_type), 'I-{}'.format(node_type)],
                average='micro'
            ),
            'f1': f1_score(
                fine_grained_labels,
                fine_grained_inferences,
                labels=['B-{}'.format(node_type), 'I-{}'.format(node_type)],
                average='micro'
            )
        }
        relevant_labels += ['B-{}'.format(node_type), 'I-{}'.format(node_type)]

        # computing the overall f1 macro
        result_dict['overall_f1_macro']['precision'] += result_dict[node_type]['precision']
        result_dict['overall_f1_macro']['recall'] += result_dict[node_type]['recall']
        result_dict['overall_f1_macro']['f1'] += result_dict[node_type]['f1']

    # computing the overall f1 macro
    result_dict['overall_f1_macro']['precision'] /= len(node_types)
    result_dict['overall_f1_macro']['recall'] /= len(node_types)
    result_dict['overall_f1_macro']['f1'] /= len(node_types)

    # computing overall ampere style results
    result_dict['ampere_macro'] = {
        'precision': precision_score(
            fine_grained_ampere_labels,
            fine_grained_ampere_inferences,
            labels = ['B-' + node_type for node_type in node_types],
            average='macro'
        ),
        'recall': recall_score(
            fine_grained_ampere_labels,
            fine_grained_ampere_inferences,
            labels = ['B-' + node_type for node_type in node_types],
            average='macro'
        ),
        'f1': f1_score(
            fine_grained_ampere_labels,
            fine_grained_ampere_inferences,
            labels = ['B-' + node_type for node_type in node_types],
            average='macro'
        )
    }

    # computing overall identification results
    result_dict['Identification_micro'] = {
        'precision': precision_score(
            fine_grained_labels,
            fine_grained_inferences,
            labels=relevant_labels,
            average='micro'
        ),
        'recall': recall_score(
            fine_grained_labels,
            fine_grained_inferences,
            labels=relevant_labels,
            average='micro'
        ),
        'f1': f1_score(
            fine_grained_labels,
            fine_grained_inferences,
            labels=relevant_labels,
            average='micro'
        )
    }
    result_dict['Identification_macro'] = {
        'precision': precision_score(
            fine_grained_labels,
            fine_grained_inferences,
            labels=relevant_labels,
            average='macro'
        ),
        'recall': recall_score(
            fine_grained_labels,
            fine_grained_inferences,
            labels=relevant_labels,
            average='macro'
        ),
        'f1': f1_score(
            fine_grained_labels,
            fine_grained_inferences,
            labels=relevant_labels,
            average='macro'
        )
    }

    # computing detection results overall
    result_dict['Detection_micro'] = {
        'precision': precision_score(
            detection_labels,
            detection_inferences,
            labels=['B-Detection', 'I-Detection'],
            average='micro'
        ),
        'recall': recall_score(
            detection_labels,
            detection_inferences,
            labels=['B-Detection', 'I-Detection'],
            average='micro'
        ),
        'f1': f1_score(
            detection_labels,
            detection_inferences,
            labels=['B-Detection', 'I-Detection'],
            average='micro'
        )
    }
    result_dict['Detection_macro'] = {
        'precision': precision_score(
            detection_labels,
            detection_inferences,
            labels=['B-Detection', 'I-Detection'],
            average='macro'
        ),
        'recall': recall_score(
            detection_labels,
            detection_inferences,
            labels=['B-Detection', 'I-Detection'],
            average='macro'
        ),
        'f1': f1_score(
            detection_labels,
            detection_inferences,
            labels=['B-Detection', 'I-Detection'],
            average='macro'
        )
    }

    return result_dict



def node_level_evaluation(
    gt_node_declarations: List[Dict[str, dict]],
    inference_node_declarations: List[Dict[str, dict]],
    data: str = 'essay'
) -> None:
    '''
        Evaluate the node level predictions
    '''

    # For storing the values to report aggregate results
    if data == 'cdcp':
        label_list = ['value', 'proposition', 'testimony', 'fact', 'reference', 'Claim', 'Detection']
    else:
        label_list = ['MajorClaim', 'Claim', 'Premise', 'Detection']
    matrix = np.zeros((len(label_list), len(label_list)))
    result_dict = {}
    for label in label_list:
        result_dict[label] = {
            'precision': 0, 'recall': 0, 'f1': 0
        }

    for gt_nodes, inference_nodes in zip(gt_node_declarations, inference_node_declarations):
        
        # will be used for confusion matrix
        sentence_list = []
        predicted_label = []
        gt_label = []

        # iterate over the nodes in inference
        for inference_node in inference_nodes.values():
            sentence_list.append(inference_node['text'].lower())
            predicted_label.append(inference_node['type'])
            gt_label.append('None')

        # iterate over the nodes in gt
        for gt_node in gt_nodes.values():
            if gt_node['text'].lower() in sentence_list:
                index = sentence_list.index(gt_node['text'].lower())
                gt_label[index] = gt_node['type']
            else:
                sentence_list.append(gt_node['text'].lower())
                predicted_label.append('None')
                gt_label.append(gt_node['type'])

        # storing the results
        instance_result_dict, instance_matrix = confusion_matrix_analysis(
            gt_label,
            predicted_label,
            labels=label_list[:-1] + ['None']
        )
        for label in label_list:
            result_dict[label]['precision'] += instance_result_dict[label]['precision']
            result_dict[label]['recall'] += instance_result_dict[label]['recall']
            result_dict[label]['f1'] += instance_result_dict[label]['f1']
        matrix += instance_matrix

    # computing the aggregate results
    for label in label_list:
        result_dict[label]['precision'] /= len(gt_node_declarations)
        result_dict[label]['recall'] /= len(gt_node_declarations)
        result_dict[label]['f1'] /= len(gt_node_declarations)
    return result_dict, matrix


def compute_threshold(
    sent_a = str,
    sent_b = str
):
    '''
        Computes the set overlap between the tokens in the two sentences
    '''
    tokens_a = set(sent_a.split())
    tokens_b = set(sent_b.split())
    return len(tokens_a.intersection(tokens_b)) / max(len(tokens_a), len(tokens_b))


def edge_level_performance_threshold(
    pred_graphs: List[List[List[str]]],
    gt_graphs: List[List[List[str]]],
    threshold: float = 0.5,
    type_matching: bool = False
):
    '''
        Compute the edge level performance for a given threshold
    '''

    # computing the number of reversed edges
    num_reversed_edges = 0

    # computing the edge level performance
    tp, fp, fn = 0, 0, 0
    for pred_graph, gt_graph in zip(pred_graphs, gt_graphs):

        # creating a variable to store which of the gt edges have been matched
        gt_matched = [False] * len(gt_graph)

        # iterating over the edges in pred
        for pred_edge in pred_graph:
            matched = False
            reversed_edge = False
            for index, gt_edge in enumerate(gt_graph):
                if (type_matching and pred_edge[1] == gt_edge[1]) or (not type_matching):
                    
                    # detection of edge with actual direction
                    if compute_threshold(pred_edge[0], gt_edge[0]) > threshold and compute_threshold(pred_edge[2], gt_edge[2]) > threshold:
                        matched = True
                        gt_matched[index] = True

                    # detection of edge with reversed direction
                    elif compute_threshold(pred_edge[0], gt_edge[2]) > threshold and compute_threshold(pred_edge[2], gt_edge[0]) > threshold:
                        reversed_edge = True

            if matched:
                tp += 1
            else:
                fp += 1

            if reversed_edge:
                num_reversed_edges += 1
    
        # computing the false negatives
        fn += (len(gt_graph) - sum(gt_matched))

    # computing precision, recall and f1
    precision = tp / (tp + fp + 1e-11)
    recall = tp / (tp + fn + 1e-11)
    f1 = 2 * precision * recall / (precision + recall + 1e-11)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'num_reversed_edges': num_reversed_edges
    }


def edge_level_evaluation(
    gt_edge_declarations: List[List[Dict[str, str]]],
    inference_edge_declarations: List[List[Dict[str, str]]],
    data: str = 'essay'
) -> None:
    '''
        Evaluate the edge level predictions
    '''

    # For storing the values to report aggregate results
    result_dict = {}
    if data == 'essay':
        labels = ['supports', 'attacks', 'Detection']
    elif data == 'abstract':
        labels = ['Support', 'Partial-Attack', 'Attack', 'Detection']
    elif data == 'echr':
        labels = ['supports', 'Detection']
    elif data == 'cdcp':
        labels = ['supports', 'Detection']
    elif data == 'ampere':
        labels = ['support', 'attack', 'Detection']
    else:
        raise NotImplementedError('Data {} not implemented'.format(data))
    matrix = np.zeros((len(labels), len(labels)))
    for label in labels:
        result_dict[label] = {
            'precision': 0, 'recall': 0, 'f1': 0
        }

    # For storing the graph matching results
    gt_graphs = []
    pred_graphs = []

    for gt_edges, inference_edges in zip(gt_edge_declarations, inference_edge_declarations):

        # will be used for confusion matrix
        sentence_list = []
        predicted_label = []
        gt_label = []

        # will be appended to gt_graphs and pred_graphs
        gt_graph = []
        pred_graph = []

        # iterate over the edges in inference
        for inference_edge in inference_edges:
            sentence_list.append(inference_edge['from'].lower() + ' | ' + inference_edge['to'].lower())
            predicted_label.append(inference_edge['stance'])
            gt_label.append('None')
            pred_graph.append([inference_edge['from'].lower(), inference_edge['stance'], inference_edge['to'].lower()])

        # iterate over the edges in gt
        for gt_edge in gt_edges:
            if gt_edge['from'].lower() + ' | ' + gt_edge['to'].lower() in sentence_list:
                index = sentence_list.index(gt_edge['from'].lower() + ' | ' + gt_edge['to'].lower())
                gt_label[index] = gt_edge['stance']
            else:
                sentence_list.append(gt_edge['from'].lower() + ' | ' + gt_edge['to'].lower())
                predicted_label.append('None')
                gt_label.append(gt_edge['stance'])
            gt_graph.append([gt_edge['from'].lower(), gt_edge['stance'], gt_edge['to'].lower()])

        # storing the results
        instance_result_dict, instance_matrix = confusion_matrix_analysis(
            gt_label,
            predicted_label,
            labels=labels[:-1] + ['None']
        )
        for label in labels:
            result_dict[label]['precision'] += instance_result_dict[label]['precision']
            result_dict[label]['recall'] += instance_result_dict[label]['recall']
            result_dict[label]['f1'] += instance_result_dict[label]['f1']
        matrix += instance_matrix

        # adding the graphs to the list
        gt_graphs.append(gt_graph)
        pred_graphs.append(pred_graph)

    # computing the aggregate results
    for label in labels:
        result_dict[label]['precision'] /= len(gt_edge_declarations)
        result_dict[label]['recall'] /= len(gt_edge_declarations)
        result_dict[label]['f1'] /= len(gt_edge_declarations)

    # computing the graph matching results
    gold_edges = split_to_edges(gt_graphs)
    pred_edges = split_to_edges(pred_graphs)
    gold_tokens, pred_tokens = get_tokens(gold_edges, pred_edges)
    precisions_rouge, recalls_rouge, f1s_rouge, precisions_bleu, recalls_bleu, f1s_bleu = get_bleu_rouge(
        gold_tokens, pred_tokens, gold_edges, pred_edges
    )

    # # handle zero elements cases
    # non_zero_gold_edges, non_zero_pred_edges = [], []
    # num_zero = 0
    # for gold_edge, pred_edge in zip(gold_edges, pred_edges):
    #     if len(gold_edge) > 0 and len(pred_edge) > 0:
    #         non_zero_gold_edges.append(gold_edge)
    #         non_zero_pred_edges.append(pred_edge)
    #     else:
    #         num_zero += 1
    # if len(non_zero_gold_edges) == 0:
    #     precisions_BS, recalls_BS, f1s_BS = [], [], []
    # else:
    #     precisions_BS, recalls_BS, f1s_BS = get_bert_score(non_zero_gold_edges, non_zero_pred_edges)
    
    result_dict['graph_matching'] = {
        'G-Rouge': {
            'precision': np.mean(precisions_rouge),
            'recall': np.mean(recalls_rouge),
            'f1': np.mean(f1s_rouge)
        },
        'G-Bleu': {
            'precision': np.mean(precisions_bleu),
            'recall': np.mean(recalls_bleu),
            'f1': np.mean(f1s_bleu)
        },
        # 'G-BertScore': {
        #     'precision': np.mean(precisions_BS + [0] * num_zero),
        #     'recall': np.mean(recalls_BS + [0] * num_zero),
        #     'f1': np.mean(f1s_BS + [0] * num_zero)
        # }
    }

    # adding the threshold_results
    result_dict['threshold_results'] = {
        'type_100': edge_level_performance_threshold(pred_graphs, gt_graphs, threshold=0.99, type_matching=True),
        'type_80': edge_level_performance_threshold(pred_graphs, gt_graphs, threshold=0.8, type_matching=True),
        'type_50': edge_level_performance_threshold(pred_graphs, gt_graphs, threshold=0.5, type_matching=True),
        'no_type_100': edge_level_performance_threshold(pred_graphs, gt_graphs, threshold=0.99, type_matching=False),
        'no_type_80': edge_level_performance_threshold(pred_graphs, gt_graphs, threshold=0.8, type_matching=False),
        'no_type_50': edge_level_performance_threshold(pred_graphs, gt_graphs, threshold=0.5, type_matching=False)
    }

    return result_dict, matrix

def evaluate_wrapper(input_dir: str, use_old_node_eval: bool, edge_evaluation: bool, logging: bool = True) -> Tuple[Dict[str, dict], Optional[Dict[str, dict]]]:
    '''
        Wrapper function for evaluating the generated prompts
    '''

    # read the config file
    config_file = os.path.join(input_dir, 'config.json')
    with open(config_file, 'r') as f:
        config = json.load(f)
    prompt_scheme = config['prompt']
    
    # assigning the data
    if 'data' in config.keys():
        data_split = config['data']
    else:
        data_split = 'essay'

    # checking if the component has been forced
    if 'force_component_identification' in config.keys():
        force_component_identification = config['force_component_identification']
    else:
        force_component_identification = False

    # loading the prompt postprocessor
    if prompt_scheme == 'code_prompt':
        prompt_postprocessor = code_prompt_postprocessor
    else:
        raise NotImplementedError('Post Processing for Prompt scheme not implemented yet')
        
    # reading the outputs
    output_processed = []
    inference_processed = []
    text_list = []
    for result_id in range(config['test_size']):
        result_file = os.path.join(input_dir, 'result_{}.json'.format(result_id))
        if not os.path.exists(result_file):
            continue
        with open(result_file, 'r') as f:
            data = json.load(f)

            # checking if the inference is None
            if data['inference'] is None:
                continue

            # post processing the output and inference
            if force_component_identification:
                output_processed.append(prompt_postprocessor(data['input'] + '\n' + data['output']))
                inference_processed.append(prompt_postprocessor(data['input'] + '\n' + data['inference']))
            else:
                output_processed.append(prompt_postprocessor(data['output']))
                inference_processed.append(prompt_postprocessor(data['inference']))

            # appending the text to the text list
            text_list.append(data['input'])

    # node level evaluation
    output_nodes = [output[0] for output in output_processed]
    inference_nodes = [inference[0] for inference in inference_processed]
    if use_old_node_eval:
        node_result_dict, node_confusion_matrix = node_level_evaluation(output_nodes, inference_nodes, data=data_split)
    else:
        node_result_dict = node_level_evaluation_standardized(output_nodes, inference_nodes, text_list)
        
    # edge level evaluation
    if ('remove_edge_prediction' not in config.keys() or config['remove_edge_prediction'] == False) and edge_evaluation:
        output_edges = [output[1] for output in output_processed]
        inference_edges = [inference[1] for inference in inference_processed]
        edge_result_dict, edge_confusion_matrix = edge_level_evaluation(output_edges, inference_edges, data=data_split)

    # printing the results
    if logging:
        print('Node Level Results')
        pprint(node_result_dict)
    if use_old_node_eval:
        if logging:
            print('Node Level Confusion Matrix')
            pprint(node_confusion_matrix)
    if ('remove_edge_prediction' not in config.keys() or config['remove_edge_prediction'] == False) and edge_evaluation:
        if logging:
            print('Edge Level Results')
            pprint(edge_result_dict)
            print('Edge Level Confusion Matrix')
            pprint(edge_confusion_matrix)

        return node_result_dict, edge_result_dict
    
    return node_result_dict, None


def evaluate_wrapper_sampling(input_dir: str, use_old_node_eval: bool, edge_evaluation: bool, num_samples: int = 10) -> Tuple[Dict[str, dict], Optional[Dict[str, dict]]]:
    '''
        Wrapper function for evaluating the generated results for sampling
    '''

    # reading the config file
    config_file = os.path.join(input_dir, 'config.json')
    with open(config_file, 'r') as f:
        config = json.load(f)

    # computing the best performing models for 10, 20, ... num_samples
    for current_samples in range(10, num_samples + 1, 10):
        
        # for storing the best samples
        best_samples = []
        text_list = []
        best_result_list = []
        
        if ('remove_edge_prediction' not in config.keys() or config['remove_edge_prediction'] == False) and edge_evaluation:
            best_edge_samples = []
            best_edge_result_list = []

        # iterating between the samples
        for result_id in range(config['test_size']):
            result_file = 'result_{}.json'
            
            # iterating over the folders in input_dir
            num_samples_found = 0
            for sample_dir in os.listdir(input_dir):
                sample_dir_path = os.path.join(input_dir, sample_dir)
                
                # checking if the folder is a directory
                if not os.path.isdir(sample_dir_path):
                    continue

                # checking if the result file exists
                result_file_path = os.path.join(sample_dir_path, result_file.format(result_id))
                if not os.path.exists(result_file_path):
                    continue

                # getting the performance only for the result_file_path
                with open(result_file_path, 'r') as f:
                    data = json.load(f)

                    # checking if the inference is None
                    if data['inference'] is None:
                        continue

                    # post processing the output and inference
                    output_processed = code_prompt_postprocessor(data['output'])
                    inference_processed = code_prompt_postprocessor(data['inference'])

                    # appending the text to the text list
                    if len(text_list) < result_id + 1:
                        text_list.append(data['input'])

                    # getting its node level performance
                    identification_micro = node_level_evaluation_standardized(
                        [output_processed[0]],
                        [inference_processed[0]],
                        [data['input']]
                    )['Identification_micro']['f1']

                    # printing the edge level performance
                    if ('remove_edge_prediction' not in config.keys() or config['remove_edge_prediction'] == False) and edge_evaluation:
                        edge_threshold_50 = edge_level_evaluation(
                            [output_processed[1]],
                            [inference_processed[1]]
                        )[0]['threshold_results']['type_50']['f1']

                    # printing the performance of the current sample
                    print('Sample: {}, Result: {}, Identification Micro F1: {}'.format(
                        sample_dir, result_id, identification_micro
                    ))
                    print('Sample: {}, Result: {}, threshold_results type 50 F1: {}'.format(
                        sample_dir, result_id, edge_threshold_50
                    ))

                    if len(best_samples) < result_id + 1:
                        best_samples.append(identification_micro)
                        best_result_list.append((output_processed, inference_processed))
                    else:
                        if identification_micro > best_samples[result_id]:
                            best_samples[result_id] = identification_micro
                            best_result_list[result_id] = (output_processed, inference_processed)

                    # adding the edge level performance
                    if ('remove_edge_prediction' not in config.keys() or config['remove_edge_prediction'] == False) and edge_evaluation:
                        if len(best_edge_samples) < result_id + 1:
                            best_edge_samples.append(edge_threshold_50)
                            best_edge_result_list.append((output_processed, inference_processed))
                        else:
                            if edge_threshold_50 > best_edge_samples[result_id]:
                                best_edge_samples[result_id] = edge_threshold_50
                                best_edge_result_list[result_id] = (output_processed, inference_processed)

                # incrementing the num_samples_found
                num_samples_found += 1

                # checking if the num_samples_found is equal to current_samples
                if num_samples_found == current_samples:
                    break

        # printing the results for the current_samples
        print('Results for {} samples with best component identification performance'.format(current_samples))

        # node level evaluation
        output_nodes = [output[0] for output, _ in best_result_list]
        inference_nodes = [inference[0] for _, inference in best_result_list]
        if use_old_node_eval:
            node_result_dict, _ = node_level_evaluation(output_nodes, inference_nodes)
        else:
            node_result_dict = node_level_evaluation_standardized(output_nodes, inference_nodes, text_list)
        print('Identification Micro F1: {}'.format(node_result_dict['Identification_micro']['f1']))

        # edge level evaluation
        if ('remove_edge_prediction' not in config.keys() or config['remove_edge_prediction'] == False) and edge_evaluation:
            output_edges = [output[1] for output, _ in best_result_list]
            inference_edges = [inference[1] for _, inference in best_result_list]
            edge_result_dict, _ = edge_level_evaluation(output_edges, inference_edges)
            
            # print the threshold result for 100 and 50
            print('Threshold Results:')
            print('100: {}'.format(edge_result_dict['threshold_results']['type_100']))
            print('50: {}'.format(edge_result_dict['threshold_results']['type_50']))

        # printing the results for the current_samples for edge level
        print('Results for {} samples with best edge level performance'.format(current_samples))

        # node level evaluation
        output_nodes = [output[0] for output, _ in best_edge_result_list]
        inference_nodes = [inference[0] for _, inference in best_edge_result_list]
        if use_old_node_eval:
            node_result_dict, _ = node_level_evaluation(output_nodes, inference_nodes)
        else:
            node_result_dict = node_level_evaluation_standardized(output_nodes, inference_nodes, text_list)
        print('Identification Micro F1: {}'.format(node_result_dict['Identification_micro']['f1']))

        # edge level evaluation
        if ('remove_edge_prediction' not in config.keys() or config['remove_edge_prediction'] == False) and edge_evaluation:
            output_edges = [output[1] for output, _ in best_edge_result_list]
            inference_edges = [inference[1] for _, inference in best_edge_result_list]
            edge_result_dict, _ = edge_level_evaluation(output_edges, inference_edges)
            
            # print the threshold result for 100 and 50
            print('Threshold Results:')
            print('100: {}'.format(edge_result_dict['threshold_results']['type_100']))
            print('50: {}'.format(edge_result_dict['threshold_results']['type_50']))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/home/inair/data/ArgumentAnnotatedEssays-2.0/final_output/test')
    parser.add_argument('--old_node_evaluation', action='store_true')
    parser.add_argument('--edge_evaluation', action='store_true')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--sampling', action='store_true')
    args = parser.parse_args()

    if args.sampling:
        evaluate_wrapper_sampling(args.input_dir, args.old_node_evaluation, args.edge_evaluation, args.num_samples)
    else:
        evaluate_wrapper(args.input_dir, args.old_node_evaluation, args.edge_evaluation)
