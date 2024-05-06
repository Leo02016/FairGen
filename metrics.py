import pickle
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import powerlaw
import argparse


def load_data(output_dir, filename, f_out):
    graph_attr = pickle.load(open(output_dir + '/graph.pickle', "rb"))
    original_network = graph_attr['graph']
    node_index = graph_attr['index']
    n = original_network.shape[0]
    generated_network = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        generated_network[i, i] = 1
    original_network = original_network + np.identity(n)
    original_network[original_network >= 2] = 1
    edge_count = int(np.sum(original_network))
    with open(filename, 'r+') as f:
        for line in f:
            line = line.rstrip("\n")
            nodes = line.split(', ')
            for i in range(len(nodes)-1):
                if i <= len(nodes) - 1:
                    index_i = node_index[nodes[i]]
                    index_j = node_index[nodes[i+1]]
                    r = np.random.uniform(low=0.8, high=1)
                    generated_network[index_i, index_j] += r
                    generated_network[index_j, index_i] += r
    DD = np.sort(generated_network.flatten())[::-1]
    threshold = DD[edge_count]
    idx = np.where(generated_network >= threshold)
    generated_network = np.zeros(generated_network.shape, dtype=np.int8)
    generated_network[idx] = 1
    with open(f_out, 'a') as f:
        aaa = compute_graph_statistics(np.array(generated_network), Z_obs=None)
        f.write('\nOurs:\n')
        write_dict(f, aaa)
    generated_network = nx.from_numpy_array(generated_network)
    original_network = nx.from_numpy_array(original_network)
    return generated_network, original_network


def squares(g):
    """
    Count the number of squares for each node
    Parameters
    ----------
    g: igraph Graph object
       The input graph.

    Returns
    -------
    List with N entries (N is number of nodes) that give the number of squares a node is part of.
    """

    cliques = g.cliques(min=4, max=4)
    result = [0] * g.vcount()
    for i, j, k, l in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
        result[l] += 1
    return result


def statistics_degrees(A_in):
    """
    Compute min, max, mean degree

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    d_max. d_min, d_mean
    """

    degrees = A_in.sum(axis=0)
    return np.max(degrees), np.min(degrees), np.mean(degrees)


def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC


def statistics_wedge_count(A_in):
    """
    Compute the wedge count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    The wedge count.
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([0.5 * x * (x - 1) for x in degrees])))


def statistics_claw_count(A_in):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Claw count
    """

    degrees = A_in.sum(axis=0)
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))


def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    A_graph = nx.from_numpy_array(A_in)
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)


def statistics_power_law_alpha(A_in):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """

    degrees = A_in.sum(axis=0)
    return powerlaw.Fit(degrees, xmin=max(np.min(degrees), 1)).power_law.alpha


def statistics_gini(A_in):
    """
    Compute the Gini coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Gini coefficient
    """

    n = A_in.shape[0]
    degrees = A_in.sum(axis=0)
    degrees_sorted = np.sort(degrees)
    G = (2 * np.sum(np.array([i * degrees_sorted[i] for i in range(len(degrees))]))) / (n * np.sum(degrees)) - (
            n + 1) / n
    return float(G)


def statistics_edge_distribution_entropy(A_in):
    """
    Compute the relative edge distribution entropy of the input graph.

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.

    Returns
    -------
    Rel. edge distribution entropy
    """

    degrees = A_in.sum(axis=0)
    m = 0.5 * np.sum(np.square(A_in))
    n = A_in.shape[0]

    H_er = 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees + .0001) / (2 * float(m))))
    return H_er


def statistics_cluster_props(A, Z_obs):
    def get_blocks(A_in, Z_obs, normalize=True):
        block = Z_obs.T.dot(A_in.dot(Z_obs))
        counts = np.sum(Z_obs, axis=0)
        blocks_outer = counts[:, None].dot(counts[None, :])
        if normalize:
            blocks_outer = np.multiply(block, 1 / blocks_outer)
        return blocks_outer

    in_blocks = get_blocks(A, Z_obs)
    diag_mean = np.multiply(in_blocks, np.eye(in_blocks.shape[0])).mean()
    offdiag_mean = np.multiply(in_blocks, 1 - np.eye(in_blocks.shape[0])).mean()
    return diag_mean, offdiag_mean


def statistics_compute_cpl(A):
    """Compute characteristic path length."""
    P = sp.csgraph.shortest_path(sp.csr_matrix(A))
    return P[((1 - np.isinf(P)) * (1 - np.eye(P.shape[0]))).astype(np.bool)].mean()


def compute_graph_statistics(A_in, Z_obs=None):
    """

    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
    Z_obs: np.matrix [N, K], where K is the number of classes.
          Matrix whose rows are one-hot vectors indicating the class membership of the respective node.

    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Size of the largest connected component (LCC)
             * Wedge count
             * Claw count
             * Triangle count
             * Square count
             * Power law exponent
             * Gini coefficient
             * Relative edge distribution entropy
             * Assortativity
             * Clustering coefficient
             * Number of connected components
             * Intra- and inter-community density (if Z_obs is passed)
             * Characteristic path length
    """
    A = A_in.copy()
    # assert ((A == A.T).all())
    A_graph = nx.from_numpy_array(A).to_undirected()
    statistics = {}
    d_max, d_min, d_mean = statistics_degrees(A)
    # Degree statistics
    statistics['d_max'] = d_max
    statistics['d_min'] = d_min
    statistics['d'] = d_mean
    # largest connected component
    LCC = statistics_LCC(A)
    statistics['LCC'] = LCC.shape[0]
    # wedge count
    statistics['wedge_count'] = statistics_wedge_count(A)
    # # claw count
    statistics['claw_count'] = statistics_claw_count(A)
    # # triangle count
    # statistics['triangle_count'] = statistics_triangle_count(A)
    # power law exponent
    statistics['power_law_exp'] = statistics_power_law_alpha(A)
    # gini coefficient
    statistics['gini'] = statistics_gini(A)
    # Relative edge distribution entropy
    statistics['rel_edge_distr_entropy'] = statistics_edge_distribution_entropy(A)
    # Assortativity
    statistics['assortativity'] = nx.degree_assortativity_coefficient(A_graph)
    # Clustering coefficient
    # statistics['clustering_coefficient'] = 3 * statistics['triangle_count'] / statistics['claw_count']
    # Number of connected components
    statistics['n_components'] = connected_components(A, directed=False)[0]
    if Z_obs is not None:
        # inter- and intra-community density
        intra, inter = statistics_cluster_props(A, Z_obs)
        statistics['intra_community_density'] = intra
        statistics['inter_community_density'] = inter
    # statistics['cpl'] = statistics_compute_cpl(A)
    return statistics


def write_dict(f, aaa):
    for item, key in aaa.items():
        f.write('{} = {}\n'.format(item, key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("deepwalk", conflict_handler='resolve')
    parser.add_argument('-d', dest='data', type=str, default='FLICKR', help='data directory')
    parser.add_argument('-f', dest='filename', type=str, default='./FLICKR_output_sequences_0_2.txt',
                        help='data directory')
    args = parser.parse_args()
    string = args.data
    output_dir = './data/{}'.format(args.data)
    filename = args.filename
    file = filename[:-4] + '_metric' + filename[-4:]
    print('Start evaluation')
    load_data(output_dir, filename, file)
