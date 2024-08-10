'''
    Contains the functions to process the data and load the data
'''

import os
import random
import json
import jsonlines
import networkx as nx
import pandas as pd
from typing import List, Tuple, Union, Dict
from networkx import DiGraph

class Node:
    '''
        Create a class node to hold the information of each node
        Will not be used - A newer version has been implemented using networkx
    '''

    def __init__(self, id=None, type=None, text=None, start=None, end=None) -> None:
        self.id = id
        self.type = type
        self.text: Union[str, List[str]] = text
        self.children: List[Tuple[Node, str]] = []
        self.start = start
        self.end = end

    def sort(self):
        self.children.sort(key=lambda x: (x[0].start, x[0].end))

    def __repr__(self) -> str:
        ret_val = f'Node: {self.type} | Node ID: {self.id} | Text: {self.text if type(self.text) == str else "<br>".join(self.text)} | Coordinates {self.start} {self.end}\n'
        for child, child_stance in self.children:
            ret_val += f'Stance: {child_stance} w.r.t to {self.id} | ' + child.__repr__()
        return ret_val

def essay_parser(graph_info: List[str]) -> Node:
    '''
        Converts annotation file to graph in the essay dataset - This will not be used
        A newer version has been implemented using networkx
    '''

    # create a node dictionary for the nodes present in the graph info
    node_info = dict()
    root_node_id = None

    # processing the node initiations
    for line in graph_info:
        
        # if the line is a node
        if line.startswith('T'):
            try:
                node_id, node_attributes, node_content = line.strip().split('\t')
                node_type, node_start, node_end = node_attributes.split(' ')
                node_start, node_end = int(node_start), int(node_end)

                # major claim is already present in the node dictionary
                if node_type == 'MajorClaim' and root_node_id is not None:
                    node_info[root_node_id].text.append(node_content)
                    continue

                # create a new node
                node = Node(id=node_id, type=node_type, text=node_content, start=node_start, end=node_end)
                node_info[node_id] = node

                # if the node is a major claim, then it is the root node
                if node_type == 'MajorClaim' and root_node_id is None:
                    root_node_id = node_id
                    node_info[root_node_id].text = [node_content]
                
            except:
                print('Error in splitting the line')
                return None

    # processing the relations
    for line in graph_info:

        # if the stance of the claim is present
        if line.startswith('A'):

            try:
                _, arg_info = line.strip().split('\t')
                _, node_id, stance = arg_info.split(' ')
                stance_formatted = 'supports' if stance == 'For' else 'attacks'
                node_info[root_node_id].children.append((node_info[node_id], stance_formatted))
            except:
                print('Error formating the line: {}'.format(line))

        # if the relation between two nodes is present
        elif line.startswith('R'):

            try:
                _, rel_info = line.strip().split('\t')
                stance, node1_id, node2_id = rel_info.split(' ')
                node1_id, node2_id = node1_id.split(':')[1], node2_id.split(':')[1]
                node_info[node2_id].children.append((node_info[node1_id], stance))
            except:
                print('Error formating the line: {}'.format(line))


    # sort the children of each node
    for node in node_info.values():
        node.sort()

    return node_info[root_node_id]

def preorder_traversal(graph: nx.DiGraph, node: Union[str, int], sorting_property: str) -> Union[List[str], List[int]]:  
    visited = [node]  
  
    children = sorted(list(graph.neighbors(node)), key=lambda x: graph.nodes[x][sorting_property])  
    for child in children:  
        visited.extend(preorder_traversal(graph, child, sorting_property))  
  
    return visited  

def essay_parser_nx(
    graph_info: List[str], is_abstract: bool = False, sorting: str = 'reading_order'
) -> DiGraph:
    '''
        Converts annotation file to graph in the essay dataset
    '''

    # creating a directed graph to store the information
    graph = DiGraph()
    root_node_id = None

    # processing the node initiations
    for line in graph_info:

        # if the line is a node
        if line.startswith('T'):
            try:
                node_id, node_attributes, node_content = line.strip().split('\t')
                node_type, node_start, node_end = node_attributes.split(' ')
                node_start, node_end = int(node_start), int(node_end)
                
                # handling the case for abstract
                if is_abstract:
                    node_type = 'Claim' if 'Claim' in node_type else 'Premise'


                # major claim is already present in the node dictionary
                if node_type == 'MajorClaim' and root_node_id is not None:
                    graph.nodes[root_node_id]['text'].append(node_content)
                    graph.nodes[root_node_id]['start'] = min(graph.nodes[root_node_id]['start'], node_start)
                    graph.nodes[root_node_id]['end'] = min(graph.nodes[root_node_id]['end'], node_end)
                    continue

                # create a new node
                graph.add_node(node_id, **{
                    'type': node_type,
                    'text': node_content,
                    'start': node_start,
                    'end': node_end
                })

                # if the node is a major claim, then it is the root node
                if node_type == 'MajorClaim' and root_node_id is None:
                    root_node_id = node_id
                    graph.nodes[node_id]['text'] = [node_content]
                
            except Exception as e:
                print('Error in splitting the line:', e)
                return None

    # processing the relations
    for line in graph_info:

        # if the stance of the claim is present
        if line.startswith('A'):

            try:
                _, arg_info = line.strip().split('\t')
                _, node_id, stance = arg_info.split(' ')
                stance_formatted = 'supports' if stance == 'For' else 'attacks'
                graph.add_edge(node_id, root_node_id, stance=stance_formatted)
            except Exception as e:
                print('Error formating the line: {}\nError: {}'.format(line, e))

        # if the relation between two nodes is present
        elif line.startswith('R'):

            try:
                _, rel_info = line.strip().split('\t')
                stance, node1_id, node2_id = rel_info.split(' ')
                node1_id, node2_id = node1_id.split(':')[1], node2_id.split(':')[1]
                graph.add_edge(node1_id, node2_id, stance=stance)
            except Exception as e:
                print('Error formating the line: {}\nError: {}'.format(line, e))

    # creating a new graph with nodes sorted by start and end
    new_graph = DiGraph()
    if sorting == 'reading_order':
        for node in sorted(graph.nodes, key=lambda x: (graph.nodes[x]['start'], graph.nodes[x]['end'])):
            new_graph.add_node(node, **graph.nodes[node])
    
    elif sorting == 'random':
        graph_nodes = list(graph.nodes)
        random.shuffle(graph_nodes)
        for node in graph_nodes:
            new_graph.add_node(node, **graph.nodes[node])
    
    elif sorting == 'topological_generation':
        
        # create a temporary graph with edges reversed and all nodes included
        temp_graph = DiGraph()
        for node in graph.nodes:
            temp_graph.add_node(node, **graph.nodes[node])
        for edge in graph.edges:
            temp_graph.add_edge(edge[1], edge[0])
        
        # sorting using topological generation
        graph_nodes = []
        for generation in nx.topological_generations(temp_graph):
            graph_nodes += sorted(generation, key=lambda x: (graph.nodes[x]['start'], graph.nodes[x]['end']))

        # adding the nodes in topological order
        for node in graph_nodes:
            new_graph.add_node(node, **graph.nodes[node])

    elif sorting == 'topological_dfs':

        # create a temporary graph with edges reversed and all nodes included
        temp_graph = DiGraph()
        for node in graph.nodes:
            temp_graph.add_node(node, **graph.nodes[node])
        for edge in graph.edges:
            temp_graph.add_edge(edge[1], edge[0])

        # sorting using topological dfs
        if is_abstract:
            graph_nodes = []

            # get nodes that have no incoming edges
            for node in temp_graph.nodes:
                if temp_graph.in_degree(node) == 0:
                    graph_nodes += preorder_traversal(temp_graph, node, 'start')

        else:
            graph_nodes = preorder_traversal(temp_graph, root_node_id, 'start')

        # adding the nodes in topological order
        for node in graph_nodes:
            new_graph.add_node(node, **graph.nodes[node])

    for edge in graph.edges:
        new_graph.add_edge(edge[0], edge[1], **graph.edges[edge])

    return new_graph

def ampere_parser_nx(node_info: List[str], edge_info: Dict) -> DiGraph:
    '''
        Converts the dictionary object to graph in the ampere dataset
    '''

    # checking if nodes in edge_info and node_info are consistent
    assert(len(edge_info['text']) == len(node_info))

    # creating a directed graph to store the information
    graph = DiGraph()
    overall_text = ''

    # iterating over the graph to collect all propositions
    for index, info in enumerate(node_info):
        info_split = info.split('\t')
        node_type = info_split[0]
        text = '\t'.join(info_split[1:])
        graph.add_node(index, **{
            'type': node_type,
            'text': text.strip(),
            'start': None,
            'end': None
        })
        overall_text += text.strip() + ' '

    # iterating over the graph to collect all relations
    for relation in edge_info['relations']:
        node1_id = relation['head']
        node2_id = relation['tail']
        graph.add_edge(node1_id, node2_id, stance=relation['type'])

    return graph, overall_text
    

def echr_parser_nx(graph_info: Dict) -> DiGraph:
    '''
        Converts dictionary object to graph in the echr dataset
    '''

    # creating a directed graph to store the information
    graph = DiGraph()
    text = graph_info['text']

    # iterating over each argument cluster
    for arg in graph_info['arguments']:

        # adding the nodes
        if not graph.has_node(arg['conclusion']):
            graph.add_node(arg['conclusion'], **{
                'type': 'Claim',
                'text': None,
                'start': None,
                'end': None
            })
        for premise in arg['premises']:
            graph.add_node(premise, **{
                'type': 'Premise',
                'text': None,
                'start': None,
                'end': None
            })

        # adding the edges
        for premise in arg['premises']:
            graph.add_edge(premise, arg['conclusion'], stance='supports')

    # assigning other properties to the graph nodes
    for clause in graph_info['clauses']:
        if graph.has_node(clause['_id']):
            graph.nodes[clause['_id']]['text'] = text[clause['start']: clause['end']]
            graph.nodes[clause['_id']]['start'] = clause['start']
            graph.nodes[clause['_id']]['end'] = clause['end']

    # creating a new graph with nodes sorted by start and end
    new_graph = DiGraph()
    for node in sorted(graph.nodes, key=lambda x: (graph.nodes[x]['start'], graph.nodes[x]['end'])):
        new_graph.add_node(node, **graph.nodes[node])
    for edge in graph.edges:
        new_graph.add_edge(edge[0], edge[1], **graph.edges[edge])

    return new_graph

def cdcp_parser_nx(graph_info: Dict) -> DiGraph:
    '''
        Converts the dictionary object to graph in the cdcp dataset        
    '''

    # creating a directed graph to store the information
    graph = DiGraph()
    text = graph_info['text']

    # iterating over the graph to collect all propositions
    assert(len(graph_info['prop_labels']) == len(graph_info['prop_offsets']))
    for index, (label, offset) in enumerate(zip(graph_info['prop_labels'], graph_info['prop_offsets'])):
        graph.add_node(index, **{
            'type': label,
            'text': text[offset[0]: offset[1]],
            'start': offset[0],
            'end': offset[1]
        })

    # iterating over the graph to collect all relations
    for support_relation in graph_info['evidences'] + graph_info['reasons']:
        source_range = list(range(support_relation[0][0], support_relation[0][1] + 1))
        target = support_relation[1]
        for source in source_range:
            graph.add_edge(source, target, stance='supports')

    # creating a new graph with nodes sorted by start and end
    new_graph = DiGraph()
    for node in sorted(graph.nodes, key=lambda x: (graph.nodes[x]['start'], graph.nodes[x]['end'])):
        new_graph.add_node(node, **graph.nodes[node])
    for edge in graph.edges:
        new_graph.add_edge(edge[0], edge[1], **graph.edges[edge])

    return new_graph


def id_to_graph(id: str, data_dir: str, is_abstract: bool = False, sorting: str = 'reading_order') -> Tuple[str, DiGraph]:
    '''
        Converts id to graph and text
    '''

    # annotation file and text file
    ann_file = os.path.join(data_dir, id) + '.ann'
    txt_file = os.path.join(data_dir, id) + '.txt'

    # reading the annotation file
    with open(ann_file, 'r') as f:
        graph = essay_parser_nx(f.readlines(), is_abstract, sorting)
    
    # reading the text file
    with open(txt_file, 'r') as f:
        text = f.read()

    return text, graph

def id_to_graph_cdcp(id: str, data_dir: str) -> Tuple[str, DiGraph]:
    '''
        Converts id to graph and text
    '''

    # annotation file and text file
    ann_file = os.path.join(data_dir, id) + '.ann.json'
    txt_file = os.path.join(data_dir, id) + '.txt'

    # reading the annotation file
    with open(ann_file, 'r') as f:
        graph_info = json.load(f)
    
    # reading the text file
    with open(txt_file, 'r') as f:
        graph_info['text'] = f.read()

    return graph_info['text'], cdcp_parser_nx(graph_info)

def dict_to_graph_ampere(graph_info: Dict, text_dir: str) -> Tuple[str, DiGraph]:
    '''
        Uses the graph info to find the relevant text file which is used to create the graph
    '''

    # getting the text file
    text_file = find_file_with_prefix(text_dir, graph_info['doc_id'])
    if text_file is None:
        return None, None
    
    # reading the text file as a list of string
    with open(text_file, 'r') as f:
        text = f.readlines()

    # creating the graph
    graph, text = ampere_parser_nx(text, graph_info)
    return text, graph

def sample_essay_dataset(
    train_test_split_csv='/home/inair/data/ArgumentAnnotatedEssays-2.0/train-test-split.csv',
    data_dir='/home/inair/data/ArgumentAnnotatedEssays-2.0/brat-project-final/',
    train_size=3,
    test_size=50,
    random_state=42,
    sorting='reading_order',
    split_train_only=False
):
    '''
        Sampling from the essay dataset
    '''
    
    # reading the train_test_split csv
    train_test_split_df = pd.read_csv(train_test_split_csv, sep=';')
    train_df = train_test_split_df[train_test_split_df['SET'] == 'TRAIN']
    test_df = train_test_split_df[train_test_split_df['SET'] == 'TEST']
    train_size = min(train_size, len(train_df))
    test_size = min(test_size, len(test_df))
    train_df = train_df.sample(n=train_size, random_state=random_state)
    test_df = test_df.sample(n=test_size, random_state=random_state)

    # converting dataframe to list
    train_id = train_df['ID'].tolist()
    test_id = test_df['ID'].tolist()

    # getting the datapoints
    train_data = [id_to_graph(id, data_dir, sorting=sorting) for id in train_id]
    test_data = [id_to_graph(id, data_dir, sorting=sorting) for id in test_id]
    return train_data, test_data

def sample_abstract_dataset(
    train_test_split_csv=None,
    data_dir='/home/inair/data/abstrct-master/AbstRCT_corpus/data',
    train_size=3,
    test_size=50,
    random_state=42,
    mode='mixed',
    sorting='reading_order'
):
    '''
        Sampling from the abstract dataset
    '''
    
    # reading the train_test_split csv
    train_dir = os.path.join(data_dir, 'train', 'neoplasm_train')
    test_dir = os.path.join(data_dir, 'test', '{}_test'.format(mode))

    # converting dataframe to list
    train_id = []
    test_id = []
    for file in os.listdir(train_dir):
        if file.endswith('.ann'):
            file_id = file.split('.')[0]
            train_id.append(file_id)
    for file in os.listdir(test_dir):
        if file.endswith('.ann'):
            file_id = file.split('.')[0]
            test_id.append(file_id)

    # sampling the data
    train_size = min(train_size, len(train_id))
    test_size = min(test_size, len(test_id))
    train_id = pd.Series(train_id).sample(n=train_size, random_state=random_state).tolist()
    test_id = pd.Series(test_id).sample(n=test_size, random_state=random_state).tolist()

    # getting the datapoints
    train_data = [id_to_graph(id, train_dir, True, sorting=sorting) for id in train_id]
    test_data = [id_to_graph(id, test_dir, True, sorting=sorting) for id in test_id]
    return train_data, test_data

def sample_echr_dataset(
    train_test_split_csv=None,
    data_dir='/home/inair/data/echr_corpus',
    train_size=3,
    test_size=50,
    random_state=42
):
    '''
        Sampling from the echr dataset
    '''
    
    # reading the train_test_split_csv
    train_test_split_csv = os.path.join(data_dir, 'ECHR_Corpus.json')
    with open(train_test_split_csv, 'r') as f:
        data_list = json.load(f)
    new_data_list = []
    for data in data_list:
        if len(data['text']) < 14000:
            new_data_list.append(data)
    data_list = new_data_list
    
    # splitting the data
    train_size = min(train_size, len(data_list))
    test_size = min(test_size, len(data_list) - train_size)
    random.seed(random_state)
    random.shuffle(data_list)
    train_data = data_list[:train_size]
    test_data = data_list[train_size: train_size + test_size]

    # getting the datapoints
    train_data = [(data['text'], echr_parser_nx(data)) for data in train_data]
    test_data = [(data['text'], echr_parser_nx(data)) for data in test_data]
    return train_data, test_data

def sample_cdcp_dataset(
    train_test_split_csv=None,
    data_dir='/home/inair/data/cdcp',
    train_size=3,
    test_size=50,
    random_state=42
):
    '''
        Sampling from the cdcp dataset
    '''

    # getting all the ids from the train directory
    train_dir = os.path.join(data_dir, 'train')
    train_ids = []
    for file in os.listdir(train_dir):
        if file.endswith('.txt'):
            train_ids.append(file.split('.')[0])

    # getting all the ids from the test directory
    test_dir = os.path.join(data_dir, 'test')
    test_ids = []
    for file in os.listdir(test_dir):
        if file.endswith('.txt'):
            test_ids.append(file.split('.')[0])

    # sampling the data
    train_size = min(train_size, len(train_ids))
    test_size = min(test_size, len(test_ids))
    random.seed(random_state)
    random.shuffle(train_ids)
    random.seed(random_state)
    random.shuffle(test_ids)
    train_ids = train_ids[:train_size]
    test_ids = test_ids[:test_size]

    # getting the datapoints
    train_data = [id_to_graph_cdcp(id, train_dir) for id in train_ids]
    test_data = [id_to_graph_cdcp(id, test_dir) for id in test_ids]
    return train_data, test_data

def find_file_with_prefix(directory: str, prefix: str) -> str:
    """
    Searches for a file with a given prefix in a given directory.
    Returns the path of the first file found with the given prefix.
    """
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            return os.path.join(directory, filename)
    return None


def sample_ampere_dataset(
    train_test_split_csv=None,
    data_dir='/home/inair/data/dataset',
    train_size=3,
    test_size=50,
    random_state=42
):
    
    '''
        Sampling from the ampere dataset
    '''

    # initializing the variables for various directory names
    text_dir = os.path.join(data_dir, 'iclr_anno_final')
    graph_dir = os.path.join(data_dir, 'release_modified')

    # reading the graph data
    with jsonlines.open(os.path.join(graph_dir, 'ampere_train.jsonl')) as reader:
        train_graph_list = list(reader)
    with jsonlines.open(os.path.join(graph_dir, 'ampere_test.jsonl')) as reader:
        test_graph_list = list(reader)

    # sampling the data
    train_size = min(train_size, len(train_graph_list))
    test_size = min(test_size, len(test_graph_list))
    random.seed(random_state)
    random.shuffle(train_graph_list)
    random.seed(random_state)
    random.shuffle(test_graph_list)
    train_graph_list = train_graph_list[:train_size]
    test_graph_list = test_graph_list[:test_size]

    # getting the datapoints
    train_data = [dict_to_graph_ampere(graph, text_dir) for graph in train_graph_list]
    test_data = [dict_to_graph_ampere(graph, text_dir) for graph in test_graph_list]
    return train_data, test_data

if __name__ == '__main__':

    # graph_file = '/home/inair/data/ArgumentAnnotatedEssays-2.0/brat-project-final/essay002.ann'
    # with open(graph_file, 'r') as f:
    #     graph_info = f.readlines()

    # graph_nx = essay_parser_nx(graph_info)
    # for node in graph_nx.nodes:
    #     print(node, graph_nx.nodes[node])
    # for edge in graph_nx.edges:
    #     print(edge, graph_nx.edges[edge]['stance'])

    train_data, test_data = sample_abstract_dataset(train_size=3, test_size=50, random_state=12, sorting='random')
    print(len(train_data), len(test_data))
    for node in train_data[0][1].nodes:
        print(node, train_data[0][1].nodes[node])
    for edge in train_data[0][1].edges:
        print(edge, train_data[0][1].edges[edge]['stance'])
    print(train_data[0][0])