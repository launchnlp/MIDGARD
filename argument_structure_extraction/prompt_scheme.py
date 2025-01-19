'''
    Prompt scheme to be used by openai
'''
import re
import networkx as nx
from networkx import DiGraph
from ast import literal_eval
from typing import List, Tuple, Dict, Callable

def longest_common_contiguous_sequence(list1, list2):
    '''
        Find the longest common contiguous sequence
    '''

    # Create a table to store lengths of the common contiguous subsequences
    table = [[0] * (len(list2) + 1) for _ in range(len(list1) + 1)]

    max_length = 0  # Length of the longest common contiguous subsequence
    end_index = 0   # Ending index of the longest common contiguous subsequence in list1

    for i in range(1, len(list1) + 1):
        for j in range(1, len(list2) + 1):
            if list1[i - 1] == list2[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
                if table[i][j] > max_length:
                    max_length = table[i][j]
                    end_index = i

    # Extract the longest common contiguous subsequence
    return (end_index - max_length, end_index)

def data_to_io(
    data: List[Tuple[str, DiGraph]],
    prompt_fn,
    force_component_identification=None
) -> Tuple[List[str], List[str]]:
    '''
        Helper function for converting list of tuples (node, string) to input-output format
    '''

    output_list = []
    input_list = []
    for essay, node in data:
        input_list.append(essay)

        # if the component identification is not forced
        if force_component_identification is None:
            output_list.append(prompt_fn(node))
        else:
            
            # check if the force component identification is a string
            assert(type(force_component_identification) == str)
            output_string = prompt_fn(node)
            output_string_elements = output_string.split(force_component_identification)
            if len(output_string_elements) == 1:
                raise ValueError('Unable to force components')
            
            # adding the first element
            input_list[-1] += '\n' + output_string_elements[0]
            output_list.append(force_component_identification.join(output_string_elements[1:]))

    return input_list, output_list


def code_prompt_generation_nx(graph: DiGraph, node_identity: bool = True, edge_prediction: bool = True) -> List[str]:
    '''
        Iteratively generating the nodes and edges from the parent generation to the subsequent
        ones.
        For instance - (n1 -> n2), (n2 -> n4) and (n3 -> n2), then the approach is to generate
        graph generation by generation
        1st round - n1, n3
        2nd round - n2
        3rd round - n4 
    '''

    # creating a graph that will be built incrementally
    incremental_graph = DiGraph()
    prompt_list = []
    
    # temp graph with reversed edges
    temp_graph = DiGraph()
    for node in graph.nodes:
        temp_graph.add_node(node, **graph.nodes[node])
    for edge in graph.edges:
        temp_graph.add_edge(edge[1], edge[0])

    # sorting using topological generation
    for generation in nx.topological_generations(temp_graph):
        current_nodes = sorted(generation, key=lambda x: (graph.nodes[x]['start'], graph.nodes[x]['end']))

        # add nodes in current_nodes
        for node in current_nodes:
            incremental_graph.add_node(node, **graph.nodes[node])

        # add edges whose tail is in current_nodes
        for edge in graph.edges:
            if edge[0] in current_nodes:
                incremental_graph.add_edge(edge[0], edge[1], **graph.edges[edge])
        
        prompt_list.append(code_prompt_nx(incremental_graph, node_identity=node_identity, edge_prediction=edge_prediction))

    return prompt_list


def code_prompt_expand_one_node_nx(graph: DiGraph, node_identity: bool = True, edge_prediction: bool = True) -> List[List[Tuple[str, str]]]:
    '''
        Iteratively generate the graph by expanding one node at a time
        For instance - (n1 -> n2), (n2 -> n4) and (n3 -> n2), then the approach is to generate
        graph generation by generation
        1st round - phi | n1, n3
        2nd round - n1 | n2, (n1 -> n2)
        3rd round - n3 | phi
        4rd round - n2 | n4, (n2 -> n4)
    '''
    
    # generations contains lists where each list contain the node and neighbors
    generation_list = []

    # temp graph with reversed edges
    temp_graph = DiGraph()
    for node in graph.nodes:
        temp_graph.add_node(node, **graph.nodes[node])
    for edge in graph.edges:
        temp_graph.add_edge(edge[1], edge[0])

    # getting the sources in the dag
    sources = [node for node in temp_graph.nodes if temp_graph.in_degree(node) == 0]
    
    # adding first generation to the generation list
    new_graph = DiGraph()
    for node in sources:
        new_graph.add_node(node, **graph.nodes[node])
    generation_list.append([('[NONE]', code_prompt_nx(new_graph, node_identity=node_identity, edge_prediction=edge_prediction, ignore_node_list=[]))])

    # iterating over a list until its empty
    while len(sources) > 0:

        # this will become the next sources
        next_sources = []
        next_generation = []

        # iterating over the sources
        for source in sources:

            # creating new graph that will be part of the new generation
            new_graph = DiGraph()
            new_graph.add_node(source, **graph.nodes[source])
                
            # adding the neighbors of the source
            for neighbor in temp_graph.neighbors(source):
                new_graph.add_node(neighbor, **graph.nodes[neighbor])
                new_graph.add_edge(neighbor, source, **graph.edges[(neighbor, source)])
                next_sources.append(neighbor)
            
            # adding the current node to the generation list
            source_node_type = graph.nodes[source]['type'] if node_identity else 'Claim'
            next_generation.append((
                '{} = {}'.format(source_node_type + '_0', graph.nodes[source]['text']),
                code_prompt_nx(new_graph, node_identity=node_identity, edge_prediction=edge_prediction, ignore_node_list=[source]) if len(new_graph.nodes) > 1 else '[EXIT]'
            ))

        # adding the next generation to the generation list
        generation_list.append(next_generation)

        # updating the sources
        sources = next_sources

    return generation_list


def data_to_iterative_io(
    data: List[Tuple[str, DiGraph]],
    prompt_fn: Callable[[DiGraph], List[str]] = code_prompt_generation_nx, 
    generations: int = 5,
    strategy: str = 'generation'
) -> Dict[int, Dict[str, List[str]]]:
    '''
        Dissociates the input into several examples so that the output is constructed iteratively
    '''
    generation_data_mapping = {
        index: {'input_list': [], 'output_list': []} for index in range(generations)
    }

    for essay, node in data:
        
        # creating iterative outputs
        prompt_list = prompt_fn(node)
        
        # if the strategy is generation
        if strategy == 'generation':

            # creating multiple examples from the above prompt
            for index, prompt in enumerate(prompt_list):
                input = 'Input: {}\n\nOutput: {}'.format(essay, '' if index == 0 else prompt_list[index - 1])
                generation_data_mapping[index]['input_list'].append(input)
                generation_data_mapping[index]['output_list'].append(prompt)

            # adding the last prompt without any modifications for the rest of the generations
            index += 1
            while index < generations:
                input = 'Input: {}\n\nOutput: {}'.format(essay, prompt_list[-1])
                generation_data_mapping[index]['input_list'].append(input)
                generation_data_mapping[index]['output_list'].append(prompt_list[-1])
                index += 1

        # if the strategy is expansion
        elif strategy == 'expand_one_node':

            # iterating over the generations
            for index, generation_list in enumerate(prompt_list):
                for generation_item in generation_list:
                    node_info, generation_prompt = generation_item
                    input = 'Input: {}\n\nCurrent Node: {}'.format(essay, node_info)
                    generation_data_mapping[index]['input_list'].append(input)
                    generation_data_mapping[index]['output_list'].append(generation_prompt)

            # adding the last prompt without any modifications for the rest of the generations
            index += 1
            while index < generations:
                for generation_item in prompt_list[-1]:
                    node_info, generation_prompt = generation_item
                    input = 'Input: {}\n\nCurrent Node: {}'.format(essay, node_info)
                    generation_data_mapping[index]['input_list'].append(input)
                    generation_data_mapping[index]['output_list'].append(generation_prompt)
                index += 1

    return generation_data_mapping

def few_shot_chat_prompt(input_list: List[str], output_list: List[str], system_prompt: str) -> List[Dict[str, str]]:
    '''
        Creating in context learning chat prompt
    '''

    messages = [{'role': 'system', 'content': system_prompt}]
    assert(len(input_list) == len(output_list))
    for input, output in zip(input_list, output_list):
        messages.append({'role': 'user', 'content': input})
        messages.append({'role': 'assistant', 'content': output})
    return messages

def convert_declarations_to_prompt(node_declarations: Dict[str, str], edge_declarations: List[dict] = None) -> str:
    '''
        Converts the node and edge declarations to prompt
    '''

    prompt = '''class argument_structure:\n\tdef __init__(self):\n\t\t# node declarations\n'''
    for node_declaration in node_declarations:
        prompt += '\t\t{} = {}\n'.format(node_declaration['name'], node_declaration['text'])
    if edge_declarations is not None or len(edge_declarations) > 0:
        prompt += '\t\t# edge declarations\n'
        for edge_declaration in edge_declarations:
            prompt += '\t\tadd_edge({}, {}, {})\n'.format(edge_declaration['from'], edge_declaration['to'], edge_declaration['stance'])

    return prompt
    
def code_prompt_nx(node: DiGraph, node_identity: bool = True, edge_prediction: bool = True, ignore_node_list: List[str] = [], add_cot: bool = False) -> str:
    '''
        Code prompt for the essay dataset using the strategy described in the paper
        "Language Models of Code are Few-Shot Commonsense Learners"
        Will not be used - newer version is implemented below
    '''
    
    # creating node names for each node id
    node_names = dict()
    for node_id in node.nodes:
        node_type = node.nodes[node_id]['type'] if node_identity else 'Claim'
        node_names[node_id] = node_type + '_' + str(len(node_names))

    # creating the prompt
    prompt = '''class argument_structure:\n\tdef __init__(self):\n\t\t# node declarations\n'''
    for node_id in node.nodes:
        if node_id in ignore_node_list: continue
        node_declaration = node.nodes[node_id]

        # adding chain of thoughts
        if add_cot:
            for neighbour in node.neighbors(node_id):
                if (node_id, neighbour) in node.edges:
                    comment_string = '\t\t# "{}" | {} | "{}"\n'.format(node_declaration['text'], node.edges[(node_id, neighbour)]['stance'], node.nodes[neighbour]['text'])
                    prompt += comment_string

        prompt += '\t\t{} = {}\n'.format(node_names[node_id], str(node_declaration['text']))

    # adding the edge declarations if edge prediction is True
    if edge_prediction and len(node.edges) > 0:
        prompt += '\t\t# edge declarations\n'
        for edge in node.edges:
            if edge[0] in node_names and edge[1] in node_names:
                prompt += '\t\tadd_edge({}, {}, {})\n'.format(node_names[edge[0]], node_names[edge[1]], node.edges[edge]['stance'])
    return prompt


def code_prompt_postprocessor(response: str, MajorClaim_postprocessor: bool = True) -> Tuple[Dict[str, Dict[str, str]], List[dict]]:
    '''
        Postprocessor for code prompt
    '''

    # elements to be returned
    node_declarations = dict()
    edge_declarations = []
    code_started = False

    # parsing the response
    elements = response.split('\n')
    for element in elements:

        # checking if the code has started
        if 'class argument_structure:' in element:
            code_started = True

        # code not yet started
        if not code_started:
            continue

        # if its a major claim
        if element.strip().startswith('MajorClaim'):
            element_info = element.strip().split(' = ')
            node_identity = element_info[0].strip()
            node_content = ' = '.join(element_info[1:]).strip()
            
            # if the node content itself is a list
            if node_content.startswith('[') and MajorClaim_postprocessor:
                try:
                    node_content = literal_eval(node_content)
                except:
                    print('Error in parsing the node content: {}'.format(node_content))
                    node_content = re.findall(r"'(.*?)'", node_content)
                cur_identity = node_identity
                for index, node_text in enumerate(node_content):
                    node_declarations[cur_identity] = {
                        'type': 'MajorClaim',
                        'text': str(node_text).strip()
                    }
                    cur_identity = node_identity + '[{}]'.format(index + 1)
            else:
                node_declarations[node_identity] = {
                    'type': 'MajorClaim',
                    'text': node_content.strip()
                }

        # if its a claim or premise
        elif element.strip().startswith('Claim') or element.strip().startswith('Premise'):
            element_info = element.strip().split(' = ')
            node_identity = element_info[0].strip()
            node_content = ' = '.join(element_info[1:]).strip()
            node_declarations[node_identity] = {
                'type': 'Claim' if 'Claim' in node_identity else 'Premise',
                'text': node_content.strip()
            }

        # it is something else
        elif '=' in element.strip():
            element_info = element.strip().split(' = ')
            node_identity = element_info[0].strip()
            node_content = ' = '.join(element_info[1:]).strip()
            node_declarations[node_identity] = {
                'type': node_identity.split('_')[0],
                'text': node_content.strip()
            }
        
        # if its an edge declaration
        elif element.strip().startswith('add_edge'):
            match = re.match(r'add_edge\(([^,]+), ([^,]+), ([^)]+)\)', element.strip())
            if match is None:
                print('Error in parsing the edge declaration: {}'.format(element))
                continue
            from_node, to_node, stance = match.group(1), match.group(2), match.group(3)
            if not from_node.startswith('MajorClaim') and from_node in node_declarations:
                from_node_text = node_declarations[from_node]['text']
            else:
                from_node_text = from_node
            if to_node in node_declarations:
                to_node_text = node_declarations[to_node]['text']

                # adding the edge declaration only if the to_node is present in the node_declarations
                edge_declarations.append({
                    'from': from_node_text,
                    'to': to_node_text,
                    'from_node': from_node,
                    'to_node': to_node,
                    'stance': stance.strip()
                })
    
    return node_declarations, edge_declarations

def seq_label_generator(
    node_declarations: Dict[str, str],
    text: str,
) -> Tuple[List[str], List[str], set]:
    '''
        Splits the text into tokens and labels them using the node_declarations
    '''

    # splitting the text into tokens
    tokens = text.lower().strip().split()
    classification_label = ['O'] * len(tokens)
    detection_label = ['O'] * len(tokens)
    node_types = set()

    # iterating over the node_declarations
    for node_key in node_declarations:
        node_text = node_declarations[node_key]['text']
        node_tokens = node_text.lower().strip().split()
        node_type = node_declarations[node_key]['type']
        start_index, end_index = longest_common_contiguous_sequence(tokens, node_tokens)
        for index in range(start_index, end_index):
            if index == start_index:
                classification_label[index] = 'B-{}'.format(node_type)
                detection_label[index] = 'B-Detection'
            else:
                classification_label[index] = 'I-{}'.format(node_type)
                detection_label[index] = 'I-Detection'
            node_types.add(node_type)
            

    return classification_label, detection_label, node_types

if __name__ == '__main__':

    with open('data/ArgumentAnnotatedEssays-2.0/brat-project-final/essay002.ann', 'r') as f:
        graph_info = f.readlines()
    with open('data/ArgumentAnnotatedEssays-2.0/brat-project-final/essay002.txt', 'r') as f:
        essay = f.read()
    from data_processor import essay_parser_nx
    graph_info = essay_parser_nx(graph_info)
    generation = data_to_iterative_io([(essay, graph_info)], prompt_fn=code_prompt_expand_one_node_nx, generations=5, strategy='expand_one_node')
    from pprint import pprint
    for i in range(5):
        print('Generation: {}'.format(i))
        pprint(generation[i]['input_list'])
        pprint(len(generation[i]['input_list']))
        print('=====================')
    print('=====================')
    # node_declarations, edge_declarations = code_prompt_postprocessor(prompt)
    # labels, labels_without_types = seq_label_generator(node_declarations, essay)
    # from seqeval.metrics import classification_report, performance_measure
    # pprint(classification_report([labels], [labels], output_dict=True))
    # pprint(performance_measure([labels], [labels]))

