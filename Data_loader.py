from graph_utils import *
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold



class data_loader():
    def __init__(self, output_directory, embedding, i, j, stage='train', mode=False):
        self.node = []
        self.label = []
        if mode:
            node_level_emb = dict()
            with open(embedding, 'r') as f_emb:
                next(f_emb)
                for line in f_emb:
                    line = line.split()
                    node_level_emb[line[0]] = np.array(list(map(float, line[1:])))
            node_seq = []
            with open('{}/sequences_{}_{}.txt'.format(output_directory, i, j), 'r') as f:
                for line in f:
                    line = line.rstrip("\n")
                    nodes = list(map(int, line.split(',')))
                    node_embed = []
                    sequences = []
                    for node in nodes:
                        node_embed.append(node_level_emb[str(node)])
                        sequences.append(node)
                    node_seq.append(sequences)
                    self.node.append(node_embed)
                    self.label.append([0, 1])
            sio.savemat('{}/node_sequence.mat'.format(output_directory), {'data':np.array(node_seq)})
            f.close()
            with open('{}/sequences_negative_{}_{}.txt'.format(output_directory, i, j), 'r') as f:
                for line in f:
                    line = line.rstrip("\n")
                    nodes = list(map(int, line.split(',')))
                    node_embed = []
                    for node in nodes:
                        node_embed.append(node_level_emb[str(node)])
                    self.node.append(node_embed)
                    self.label.append([1, 0])
            f.close()
            self.node = np.array(self.node)
            self.label = np.array(self.label)
            skf = StratifiedKFold(n_splits=5)
            for train_index, test_index in skf.split(self.node, self.label[:, 0]):
                node_train = self.node[train_index]
                node_val = self.node[test_index]
                y_train = self.label[train_index]
                y_val = self.label[test_index]
                break
            train_data = {'label': y_train, 'node': node_train}
            val_data = {'label': y_val, 'node': node_val}
            sio.savemat('{}/train_{}_{}.mat'.format(output_directory, i, j), train_data)
            sio.savemat('{}/val_{}_{}.mat'.format(output_directory, i, j), val_data)
            self.node = []
            with open('{}/sequences_generation_{}_{}.txt'.format(output_directory, i, j), 'r') as f:
                self.node_list = []
                for line in f:
                    line = line.rstrip("\n")
                    nodes = list(map(int, line.split(',')))
                    node_embed = []
                    sequences = []
                    for node in nodes:
                        node_embed.append(node_level_emb[str(node)])
                        sequences.append(node)
                    self.node.append(node_embed)
                    self.node_list.append(sequences)
                generation_data = {'data': self.node_list, 'emb': self.node}
                generation_data['emb'] = np.array(generation_data['emb'])
                sio.savemat('{}/generation.mat'.format(output_directory), generation_data)
            f.close()
        if stage == 'train' or stage == 'Train':
            data = sio.loadmat('{}/train_{}_{}.mat'.format(output_directory, i, j))
            self.label = data['label']
            self.node = data['node']
        elif stage == 'val' or stage == 'Val':
            data = sio.loadmat('{}/val_{}_{}.mat'.format(output_directory, i, j))
            self.label = data['label']
            self.node = data['node']
        else:
            raise (NameError('The stage should be either train or test'))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.node[idx]
        y = self.label[idx]
        sample = {'node': np.array(x), 'label': y}
        return sample


class node_loader():
    def __init__(self, output_directory, config, i, j, stage='train'):
        data_directory ='{}/{}_emb_layer_{}_iter_{}'.format(output_directory, config.data, i, j)
        if stage == 'train':
            node_level_emb = dict()
            with open(data_directory, 'r') as f_emb:
                next(f_emb)
                for line in f_emb:
                    line = line.split()
                    node_level_emb[line[0]] = np.array(list(map(float, line[1:])))
            # we need to split the data set into training set and test set.
            f_emb.close()
            data = loadmat(output_directory + '/{}_label_{}_{}.mat'.format(config.data, i, j))
            labels = self.label_encoding(data['label'][0])
            identity = data['identity'].reshape(labels.shape[0])
            train_data = []
            train_labels = []
            train_identity = []
            for index in data['train_index'][0]:
                train_data.append(node_level_emb[str(index)])
                train_labels.append(labels[index])
                train_identity.append(identity[index])
            self.data = np.array(train_data)
            self.label = np.array(train_labels)
            self.identity = np.array(train_identity)
            test_data = []
            test_labels = []
            for index in data['test_index'][0]:
                test_data.append(node_level_emb[str(index)])
                test_labels.append(labels[index])
            test_identity = identity[data['test_index'][0]]
            savemat(output_directory + '/minotrity_test_{}_{}.mat'.format(i,j),
                    {'test_data':test_data, 'test_label': test_labels, 'test_identity': test_identity,
                     'data': train_data, 'label': train_labels, 'identity': train_identity})
        else:
            data = loadmat(output_directory + '/minotrity_test_{}_{}.mat'.format(i, j))
            self.data = np.array(data['test_data'])
            self.label = np.array(data['test_label'])
            self.identity = np.array(data['test_identity'][0])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        identity = self.identity[idx]
        sample = {'data': x, 'label': y, 'identity': identity}
        return sample

    def label_encoding(self, label):
        num = np.unique(label).shape[0]
        arr = np.zeros((len(label), num))
        min_num = min(np.unique(label))
        for i in range(len(label)):
            arr[i, label[i]-min_num] = 1.0
        return arr
