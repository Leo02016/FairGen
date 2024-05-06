import numpy as np
import pickle
import os
import random
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def graph_similarity(graph):
    alpha = 0.99
    row_sums = graph.sum(axis=1)
    normalized_graph = graph / row_sums[:, np.newaxis]
    x = np.identity(graph.shape[0], dtype=np.int8) - alpha * normalized_graph
    node_similarity = np.linalg.inv(x)
    for i in range(len(node_similarity)):
        node_similarity[i, i] = node_similarity[i, i] / 2
    return node_similarity


def data_process(args, biased, data_directory='./data/DBLP/edgelist.txt', output_directory="./data/DBLP", directed=False):
    node_dict, node_index, original_node_index, node_value = preprocess_edgelist(data_directory, directed)
    if not os.path.exists("{}/{}_emb".format(output_directory, args.data)):
        os.system("python ./deepwalk/main.py --representation-size {} --input {} --output {}/{}_emb".format(
            args.emb_size, data_directory[:-4] + '_new' + data_directory[-4:], output_directory, args.data))
    # original graph
    graph = np.zeros((len(original_node_index), len(original_node_index)), dtype=np.int8)
    with open(data_directory, 'r+') as f:
        for line in f:
            nodes = list(map(int, line.split()))
            i = original_node_index[nodes[0]]
            j = original_node_index[nodes[1]]
            graph[i, j] = 1
            graph[j, i] = 1
    for i in range(len(original_node_index)):
        graph[i, i] = 1
    degree = np.array(np.sum(graph, axis=1), dtype=np.int64)
    node_similarity = graph_similarity(graph)
    data = dict()
    data['original_index'] = original_node_index
    data['index'] = node_index
    data['value'] = node_value
    data['dict'] = node_dict
    data['graph'] = graph
    pickle.dump(data, open(output_directory + '/graph.pickle', "wb"))
    random_walks(node_dict, node_similarity, degree, output_directory, biased, node_index, data_directory)


def dictionary_search(dictionary, search_value):
    for key, value in dictionary.items():
        if value == search_value:
            return key


def random_walks(node_dict, node_similarity, degree, output_directory, biased, node_index, data_directory):
    walk_length = 10
    if biased:
        random_walk_strategy = biased_neighbor_selection_with_prob
        repeated = list(map(int, np.array(degree / 5) + 1))
    else:
        # less sequences but faster speed
        random_walk_strategy = unbiased_neighbor_selection_with_prob
        repeated = [3 for _ in range(len(degree))]
    count = 0
    with open('{}/sequences_0_0.txt'.format(output_directory), 'w+') as f:
        with open('{}/sequences_generation_0_0.txt'.format(output_directory), 'w+') as f_g:
            with open('{}/sequences_negative_0_0.txt'.format(output_directory), 'w+') as f_neg:
                for item, v in node_dict.items():
                    for j in range(repeated[count]):
                        # adding the initial node to the sequence and initializing the neighbor list
                        sequence = [int(node_index[item])]
                        neighbor_list = v
                        node = item
                        for i in range(1, walk_length):
                            # select neighbors
                            prob_list = random_walk_strategy(neighbor_list, node_similarity, node, node_index)
                            index = choose_action(prob_list)
                            while index is None:
                                index = choose_action(prob_list)
                            node = str(neighbor_list[index])
                            # adding the selected node to the list
                            sequence.append(node_index[node])
                            # reinitialized the neighbor_list
                            neighbor_list = node_dict[node]
                        f.write(', '.join(map(str, sequence)) + '\n')
                        # generate sequence for sequence generation
                        f_g.write(', '.join(map(str, sequence[:10])) + '\n')
                        # generate sequence for negative sample by substituting only 1-3 nodes in the sequence
                        neg_seq = np.array(sequence, copy=True)
                        indices = random.choices(range(len(sequence)), k=random.randint(2, 3))
                        for index in indices:
                            neg_seq[index] = random.randint(0, len(node_dict)-1)
                        f_neg.write(', '.join(map(str, neg_seq)) + '\n')
                    count += 1


# Pr(w) = 1/|Γt (v)| , where the mapping function Γt (v) = T - time_windows (T >= time_windows)
def unbiased_neighbor_selection_with_prob(neighbor_list, node_similarity=None, k=None, node_index=None):
    prob_list = [1.0 / len(neighbor_list) for _ in range(len(neighbor_list))]
    return prob_list


def biased_neighbor_selection_with_prob(neighbor_list, node_similarity, k, node_index):
    prob_list = [node_similarity[node_index[k], node_index[str(item)]] for item in neighbor_list]
    prob_list = prob_list / np.sum(prob_list)
    return prob_list


def choose_action(c):
    r = np.random.random()
    c = np.array(c)
    for i in range(1, len(c)):
        c[i] = c[i]+c[i-1]
    for i in range(len(c)):
        if c[i] >= r:
            return i


def preprocess_edgelist(data_directory, directed):
    node_dict = dict()
    node_index = dict()
    node_value = dict()
    original_node_index = dict()
    count = 0
    original_node_count = 0
    filename = data_directory[:-3] + 'mat'
    data = loadmat(filename)
    n = data['Network'].shape[0]
    with open(data_directory[:-3] + 'txt', 'w') as f:
        index = data['Network'].indptr[count+1]
        for j in range(len(data['Network'].indices)):
            ind = data['Network'].indices[j]
            f.write('{} {} 1\n'.format(ind, count))
            if count > n:
                break
            if j >= index:
                count += 1
                index = data['Network'].indptr[count + 1]
    count = 0
    with open(data_directory, 'r') as f:
        with open(data_directory[:-4] + '_new' + data_directory[-4:], 'w') as f_out:
            for line in f:
                nodes = list(map(int, line.split()))
                # index dictionary by which we could map newly-created nodes back to original graph
                if nodes[0] not in original_node_index:
                    original_node_index[nodes[0]] = original_node_count
                    original_node_count += 1
                if nodes[1] not in original_node_index:
                    original_node_index[nodes[1]] = original_node_count
                    original_node_count += 1
                # map original nodes to newly-created nodes to encode the time info
                if str(nodes[0]) not in node_index:
                    node_value[count] = str(nodes[0])
                    node_index[str(nodes[0])] = count
                    node_dict[str(nodes[0])] = [(nodes[0])]
                    count += 1
                # if the second node does not exist in the dictionary, then add it to the dictionary
                if str(nodes[1]) not in node_index:
                    node_value[count] = str(nodes[1])
                    node_index[str(nodes[1])] = count
                    node_dict[str(nodes[1])] = [(nodes[1])]
                    count += 1
                if (nodes[1]) not in node_dict[str(nodes[0])]:
                    node_dict[str(nodes[0])].append((nodes[1]))
                if not directed:
                    if (nodes[0]) not in node_dict[str(nodes[1])]:
                        node_dict[str(nodes[1])].append((nodes[0]))
                # f_out.write("{} {}\n".format(node_index[str(nodes[1])], node_index[str(nodes[0])]))
                f_out.write("{} {}\n".format(node_index[str(nodes[0])], node_index[str(nodes[1])]))
    print("Finish Remapping nodes! Total number of nodes = {}".format(count))
    y = data['Class']
    new_y = []
    identity = data['Label']
    new_idenity = []
    for i in range(len(y)):
        new_y.append(y[original_node_index[i]][0])
        new_idenity.append(identity[original_node_index[i]][0])
    y = np.array(new_y)
    identity = np.array(new_idenity)
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(data['Network'], y):
        label_info = {'label': y, 'identity': identity, 'train_index': test_index, 'test_index':  train_index}
        savemat(data_directory[:-4] + '_label_0_0.mat', label_info)
        break
    return node_dict, node_index, original_node_index, node_value
