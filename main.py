import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import argparse
from utils import *
import scipy.io as sio
from torch.utils.data import DataLoader
import time
from Data_loader import data_loader, node_loader
from FairGen import Transformer, Discriminator


# Configuration
class Config(object):
    # parameters for transformer
    N = 1
    d_model = 80
    d_ff = 128
    h = 4
    dropout = 0.2
    output_size = 2
    lr = 0.03
    max_epochs = 10
    batch_size = 64
    max_sen_len = 15
    action_prob = [0.5, 0.4, 0.1]
    search_size = 500
    sample_time = 20
    # parameter for accelerating the computation
    windows_size = 2


# Train the FairGen model
def run_epoch(train_iterator, val_iterator, epoch, model):
    train_losses = []
    losses = []
    optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate, momentum=0.7)
    criteria = F.binary_cross_entropy_with_logits
    start_time = time.time()
    for k in range(epoch):
        for i, batch in enumerate(train_iterator):
            optimizer.zero_grad()
            x = batch['node'].double().to(model.device)
            y = batch['label'].double().to(model.device)
            y_pred = model(x)
            loss = criteria(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            optimizer.step()
            if (i+1) % 100 == 0:
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                losses = []
                val_accuracy = evaluate_model(model, val_iterator)
                print("Epoch: [{}/{}],  iter: {}, average training loss: {:.5f}, val accuracy: {:.4f}, training time = {:.4f}".format(
                    k+1, epoch, i + 1, avg_train_loss, val_accuracy, time.time() - start_time))
                model.train()


# Selecting threshold
def threshold(train_iterator, model):
    all_preds = []
    for idx, batch in enumerate(train_iterator):
        x = batch['node'].double().to(model.device)
        y_pred = model(x)
        predicted = y_pred.cpu().data[:, 1]
        indices = torch.max(y_pred.cpu().data, 1)[1]
        all_preds.extend(predicted[np.where(indices == 1)])
    return sum(all_preds)/len(all_preds)


# Evaluation on validation set
def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        x = batch['node'].double().to(model.device)
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1]
        all_preds.extend(predicted.numpy())
        all_y.extend(np.array([0 if i[0] else 1 for i in batch['label'].numpy()]))
    score = accuracy_score(all_y, all_preds)
    return score


# Generate random walk sequence
def generate_sequence(config, output_directory, model, index_i, index_j, final=False):
    model.eval()
    sequences_node_emb = sio.loadmat('{}/generation.mat'.format(output_directory))['emb']
    node_sequences = sio.loadmat('{}/generation.mat'.format(output_directory))['data']
    node_sequences_train = sio.loadmat('{}/node_sequence.mat'.format(output_directory))['data']
    node_level_emb = dict()
    with open(config.embedding, 'r') as f_emb:
        next(f_emb)
        for line in f_emb:
            line = line.split()
            node_level_emb[line[0]] = np.array(list(map(float, line[1:])))
    f_emb.close()
    start_time = time.time()
    data = loadmat(data_directory[:-4] + '_label_{}_{}.mat'.format(index_i, index_j))
    identity = data['identity'][0]
    for i in range(len(sequences_node_emb)):
        if i % 1000 == 0:
            print('Generating {} sequences in {} seconds'.format(i, time.time() - start_time))
        sequence_node_level = sequences_node_emb[i].reshape(1, sequences_node_emb[i].shape[0], config.d_model)
        node_sequence = node_sequences[i]
        pos = 1
        key_nodes = [0 for _ in range(len(node_sequence)-1)]
        key_nodes.insert(0, 1)
        action = 0
        min_length = min(len(node_sequence), 2)
        if not final:
            if node_sequence[0] in data['train_index'][0]:
                if identity[node_sequence[0]] == 1:
                    generate_seq(index_i, index_j, config, sequence_node_level, node_level_emb, model, min_length,
                                 node_sequence, key_nodes, pos, action)
                    continue
            node_sequence = node_sequences_train[i]
            # with open('{}'.format(config.use_output_path[:-4] + '_{}_{}'.format(
            #         index_i, index_j) + config.use_output_path[-4:]), 'a') as g:
            #     g.write(', '.join(map(str, node_sequence)) + '\n')
            with open('{}'.format(config.use_output_path_1[:-4] + '_{}_{}'.format(
                    index_i, index_j) + config.use_output_path_1[-4:]),'a') as g:
                for j in range(len(node_sequence)):
                    if j + 1 < len(node_sequence):
                        g.write('{} {} 1\n'.format(node_sequence[j], node_sequence[j + 1]))
        else:
            generate_seq(index_i, index_j, config, sequence_node_level, node_level_emb, model, min_length,
                         node_sequence, key_nodes, pos, action)


# Generate new random walk sequences by inserting node and deleting node and discriminator determine which sequences
# follow the distribution of real-world graph.
def generate_seq(index_i, index_j, config, sequence_node_level, node_level_emb, model, min_length, node_sequence,
                 key_nodes, pos, action):
    for j in range(config.sample_time):
        ind = random.randint(0, sequence_node_level.shape[1] - 1)
        if ind == 0 or ind == len(key_nodes):
            ind = random.randint(0, sequence_node_level.shape[1] - 1)
        # insertion (action:0)
        if action == 0:
            # sampling nodes
            sampling_nodes = np.random.permutation(range(len(node_level_emb)))[:config.search_size]
            sampling_node_level_emb = [node_level_emb[str(node)] for node in sampling_nodes]
            n = len(sampling_nodes)
            candidate_key_nodes = np.zeros((n, len(key_nodes) + 1), dtype=np.int32)
            candidate_sequence_node_level = np.zeros((n, len(key_nodes) + 1, config.d_model))
            for k in range(n):
                candidate_key_nodes[k] = np.concatenate([key_nodes[:ind], [0], key_nodes[ind:]])
                candidate_sequence_node_level[k] = np.concatenate(
                    [sequence_node_level[0, :ind], sampling_node_level_emb[k].reshape(1, config.d_model),
                     sequence_node_level[0, ind:]], axis=0)
            candidate_sequence_node_level = torch.from_numpy(candidate_sequence_node_level).to(model.device)
            if candidate_sequence_node_level.shape[1] <= 5:
                y_pred = model(candidate_sequence_node_level)
            else:
                if ind + config.windows_size > sequence_node_level.shape[1] - 1:
                    end_ind = sequence_node_level.shape[1]
                    start_ind = end_ind - 1 - config.windows_size * 2
                else:
                    start_ind = min(0, ind + config.windows_size)
                    end_ind = start_ind + config.windows_size * 2 + 1
                y_pred = model(candidate_sequence_node_level[:, start_ind:end_ind, :])
            accept_prob = y_pred.cpu().data[:, 1]
            max_accept_prob, indices = torch.max(accept_prob, 0)
            if max_accept_prob > config.threshold:
                sequence_node_level = candidate_sequence_node_level[indices].cpu().numpy().reshape(1,
                                                                                                   len(key_nodes) + 1,
                                                                                                   config.d_model)
                node_sequence = np.concatenate(
                    [node_sequence[:ind], [sampling_nodes[indices]], node_sequence[ind:]], axis=0)
                key_nodes = candidate_key_nodes[indices]
            else:
                action = 2
        # deletion (action: 2)
        if action == 1:
            # avoid deleting key nodes
            if key_nodes[ind] == 1.0:
                continue
            if len(key_nodes) <= min_length:
                continue
            else:
                candidate_sequence_node_level = np.zeros((len(key_nodes), len(key_nodes) - 1, config.d_model))
                candidate_key_nodes = [[] for _ in range(len(key_nodes))]
                for k in range(len(key_nodes)):
                    candidate_key_nodes[k] = np.concatenate([key_nodes[:k], key_nodes[k + 1:]], axis=0)
                    candidate_sequence_node_level[k] = np.concatenate(
                        [sequence_node_level[0, :k], sequence_node_level[0, k + 1:]], axis=0)
                candidate_sequence_node_level = torch.from_numpy(candidate_sequence_node_level).to(model.device)
                y_pred = model(candidate_sequence_node_level)
                accept_prob = y_pred.cpu().data[:, 0]
                max_accept_prob, indices = torch.max(accept_prob, 0)
                if max_accept_prob > config.threshold and choose_action([0.5, 0.5]):
                    sequence_node_level = candidate_sequence_node_level[indices].cpu().numpy().reshape(1, len(
                        key_nodes) - 1, config.d_model)
                    key_nodes = candidate_key_nodes[indices]
                    node_sequence = np.concatenate([node_sequence[:indices], node_sequence[indices + 1:]], axis=0)
                else:
                    action = 2
        if action == 2:
            pos += 1
        action = choose_action(config.action_prob)
        if len(key_nodes) >= config.max_sen_len:
            break
        if len(key_nodes) <= min_length:
            break
    with open(
            '{}'.format(config.use_output_path[:-4] + '_{}_{}'.format(index_i, index_j) + config.use_output_path[-4:]),
            'a') as g:
        g.write(', '.join(map(str, node_sequence)) + '\n')
    with open('{}'.format(
            config.use_output_path_1[:-4] + '_{}_{}'.format(index_i, index_j) + config.use_output_path_1[-4:]),
              'a') as g:
        for j in range(len(node_sequence)):
            if j + 1 < len(node_sequence):
                g.write('{} {} 1\n'.format(node_sequence[j], node_sequence[j + 1]))


# Initialize the graph generation task
# 1. Loading the datasets
# 2. Initialize the Transformer-based FairGen
# 3. Pretrain FairGen for a few epochs to begin generate synthetic random walk sequences
def initialization(args, config, output_directory):
    train_dataset = data_loader(output_directory, args.embedding, 0, 0, 'train', True)
    train_iterator = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    val_dataset = data_loader(output_directory, args.embedding, 0, 0,  'val')
    val_iterator = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    model = Transformer(config).to(device)
    model = model.double()
    model.device = device
    model.learning_rate = 0.04
    run_epoch(train_iterator, val_iterator, config.max_epochs, model)
    print('Finish pretrain the model!')
    torch.save(model.state_dict(), '{}/model_epoch_{}.ckpt'.format(output_directory, config.max_epochs))


# Train the discriminator. The goal of discriminator is to determine whether the synthetic sequences
# follow the distribution of the raw graph
def train_discriminator(train_iterator, test_iterator, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate, momentum=0.95)
    epoch = 1000
    x = torch.tensor(train_iterator.dataset.data).to(model.device)
    y = torch.tensor(train_iterator.dataset.label).to(model.device)
    identity = torch.tensor(train_iterator.dataset.identity).to(model.device)
    print('Start to train the discriminator')
    for k in range(1, epoch):
        optimizer.zero_grad()
        loss = model(x, identity, y)
        loss.backward()
        optimizer.step()
    model.eval()  # eval mode (batch norm uses moving mean/variance instead of mini-batch mean/variance)
    x = torch.tensor(test_iterator.dataset.data).to(model.device)
    identity = torch.tensor(test_iterator.dataset.identity).to(model.device)
    y_pred = model(x, identity)
    return y_pred


# Update the training set at each pace
def update_files(node_indices, i, j, output_directory, embedding):
    node_level_emb = dict()
    with open(embedding, 'r') as f_emb:
        next(f_emb)
        for line in f_emb:
            line = line.split()
            node_level_emb[line[0]] = np.array(list(map(float, line[1:])))
    # update BLOG_label_0_0.mat file by adding sequence whose initial nodes are from node_indices list.
    data = loadmat(data_directory[:-4] + '_label_{}_{}.mat'.format(i, j))
    train_index = list(data['train_index'][0])
    test_index = data['test_index'][0]
    new_indices = list(test_index[node_indices])
    test_index = list(test_index)
    for idx in new_indices:
        train_index.append(idx)
        test_index.remove(idx)
    data['train_index'] = train_index
    data['test_index'] = test_index
    # update seq_negative sampling and positive sampling nodes
    # read generated sequence from file and add it to negative samples.
    data_1 = sio.loadmat('{}/train_{}_{}.mat'.format(output_directory, i, j))
    data_2 = sio.loadmat('{}/val_{}_{}.mat'.format(output_directory, i, j))
    sio.savemat('{}/val_{}_{}.mat'.format(output_directory, i, j+1), data_2)
    new_sequences = []
    label = []
    with open('{}'.format(config.use_output_path[:-4] + '_{}_{}'.format(i, j) + config.use_output_path[-4:]), 'r') as f:
        for lines in f:
            zeros = np.zeros((10, 80))
            lines = lines.rstrip("\n")
            lines = list(map(int, lines.split(',')))
            if len(lines) >= 10:
                length = 10
            else:
                length = len(lines)
            for k in range(length):
                zeros[k] = node_level_emb[str(lines[k])]
            new_sequences.append(zeros)
            label.append([0, 1])
    label = np.array(label)
    new_sequences = np.array(new_sequences)
    data_1['label'] = np.concatenate([data_1['label'], label])
    data_1['node'] = np.concatenate([data_1['node'], new_sequences])
    sio.savemat('{}/train_{}_{}.mat'.format(output_directory, i, j + 1), data_1)
    sio.savemat(data_directory[:-4] + '_label_{}_{}.mat'.format(i, j+1), data)
    print('save label file in '+ data_directory[:-4] + '_label_{}_{}.mat'.format(i, j+1))


# Main function
def main(args, config, output_directory):
    # initialization
    initialization(args, config, output_directory)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = Transformer(config).to(device)
    model = model.double()
    model.device = device
    model.learning_rate = 0.01
    new_model = torch.load('{}/model_epoch_{}.ckpt'.format(output_directory, config.max_epochs))
    model.load_state_dict(new_model)
    train_dataset = data_loader(output_directory, args.embedding, 0, 0, 'train')
    train_iterator = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=8)
    config.threshold = max(threshold(train_iterator, model), 0.95)
    # initialize discriminator
    data = loadmat(output_directory + '/{}_label_0_0.mat'.format(config.data))
    num = len(np.unique(data['label']))
    disc = Discriminator(args.emb_size, num).to(device).double()
    disc.device = device
    disc.learning_rate = 0.05
    iterations = 3
    Layer = 1
    final = False
    for i in range(Layer):
        # lam: a threshold to determine whether a sample should be added into the labeled set based on confidence score
        lam = 0.995
        for j in range(iterations):
            print('\n Pace {}:'.format(j + 1))
            if j == iterations - 1:
                final = True
                # generate synthetic sequences
                generate_sequence(config, output_directory, model, i, j, final)
            if not final:
                # generate synthetic sequences
                generate_sequence(config, output_directory, model, i, j, final)
                # generate synthetic graph based on the synthetic sequences
                os.system("python ./deepwalk/main.py --representation-size {} --input {} --output {}/{}_emb_layer_{}_iter_{}".format(
                    args.emb_size, data_directory[:-4] + '_new' + data_directory[-4:], output_directory, args.data, i, j))
                node_train = node_loader(output_directory, config, i, j, 'train')
                node_iterator = DataLoader(node_train, batch_size=50, shuffle=False, num_workers=8)
                node_test = node_loader(output_directory, config, i, j, 'test')
                node_test_iterator = DataLoader(node_test, batch_size=50, shuffle=False, num_workers=8)
                # curriculum learning to select the most k confident nodes
                pred_prob = train_discriminator(node_iterator, node_test_iterator, disc)
                # print(pred_prob)
                prob = torch.max(pred_prob, 1)[0]
                selected_nodes = []
                for k in range(len(prob)):
                    if prob[k] > lam:
                        selected_nodes.append(k)
                lam = lam - 0.01
                # # Update the training set
                update_files(selected_nodes, i, j, output_directory, config.embedding)
                # load new training set
                train_dataset = data_loader(output_directory, args.embedding, i, j+1, 'train')
                train_iterator = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=8)
                val_dataset = data_loader(output_directory, args.embedding, i, j+1,  'val')
                val_iterator = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
                # Fine-tune the transformer
                print('Finetune the transformer')
                run_epoch(train_iterator, val_iterator, 2, model)
                print('Finish Finetuning!')
                torch.save(model.state_dict(), '{}/model_epoch_{}.ckpt'.format(output_directory, config.max_epochs))
    filename = config.use_output_path_1[:-4] + '_{}_{}'.format(i, j) + config.use_output_path_1[-4:]
    print('The final random walk sequences are saved in directory: {}'.format(filename))
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FairNet", conflict_handler='resolve')
    parser.add_argument('-d', dest='data', type=str, default='FLICKR', help='data directory')
    parser.add_argument('-g', dest='gpu', type=str, default='0', help='the index of GPU')
    parser.add_argument('-b', dest='biased', action='store_true', help="biased or unbiased, default is unbiased")
    # parser.add_argument('-m', dest='mode', action='store_true', help='train or test')
    args = parser.parse_args()
    config = Config()
    biased = args.biased
    # path of data for training language model
    args.data_path = './data/{}/sequences.txt'.format(args.data)
    # data path of original sentences
    args.embedding = './data/{}/{}_emb'.format(args.data, args.data)
    config.embedding = './data/{}/{}_emb'.format(args.data, args.data)
    args.model_path = './model_{}/'.format(args.data)
    config.use_output_path = './data/{}/{}_output_sequences.txt'.format(args.data, args.data)
    output_directory = "./data/{}".format(args.data)
    data_directory = './data/{}/{}.txt'.format(args.data, args.data)
    config.data_directory = data_directory
    config.data = args.data
    config.use_output_path_1 = './data/{}/{}_output_edgelist.txt'.format(args.data, args.data)
    args.emb_size = config.d_model
    data_process(args, biased, data_directory=data_directory, output_directory=output_directory, directed=False)
    filename = main(args, config, output_directory)
    # filename = './data/FLICKR/FLICKR_output_edgelist_0_2.txt'
    # evaluating the performance
    os.system('python metrics.py -d {} -f {}'.format(args.data, filename))

