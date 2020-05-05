from cdt.causality.graph import LiNGAM, CAM, CCDr, GES, PC, GS, IAMB, MMPC
import copy
import os
import pickle
import json
from functools import singledispatch
import pandas as pd
import numpy as np
import argparse
import networkx as nx
from pgmpy.readwrite.BIF import BIFReader
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from CausalModel.BayesianNetwork import BayesianNetwork
from CausalModel.CMD import ODist, IDist
from cdt.metrics import SID, SHD

algorithms = {'LiNGAM': LiNGAM, 'CAM': CAM, 'CCDr': CCDr, 'GS': GS, 'GES': GES, 'PC': PC, 'IAMB': IAMB, 'MMPC': MMPC}


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


def load_networks(folder_path):
    networks = [(f.split('.')[0], os.path.join(folder_path, f)) for f in os.listdir(folder_path)]

    loaded_networks = {}
    for name, path in networks:
        reader = BIFReader(path)
        model = reader.get_model()
        loaded_networks[name] = model
    return loaded_networks


def generate_datasets(networks, folder, nb_samples=2000):
    for network in networks:
        dataset_out_path = os.path.join(folder, 'datasets', network + '.csv')
        inference = BayesianModelSampling(networks[network])
        samples = inference.forward_sample(size=nb_samples)

        samples.to_csv(dataset_out_path)


def run_algorithms(folder):
    dst_folder = os.path.join(folder, 'graphs')
    datasets_folder = os.path.join(folder, 'datasets')
    datasets = [(f.split('.')[0], os.path.join(datasets_folder, f)) for f in os.listdir(datasets_folder)]
    for dataset_name, path in datasets:
        print(dataset_name)
        # if dataset_name == 'andes' or dataset_name == 'diabetes' or dataset_name == 'win95pts':
        #     continue
        data = pd.read_csv(path)
        data = data.astype('float64')
        # data = data + 0.0001 * np.random.normal(0, 0.1, [data.shape[0], data.shape[1]])
        # print(data.head())
        # exit()
        for algo_name, algo in algorithms.items():
            if algo_name == 'CAM':
                continue
            print(algo.__name__)
            saving_file = algo_name + '_' + dataset_name
            saving_path = os.path.join(dst_folder, saving_file)

            al = algo()

            # ugraph = al.predict(data)
            ugraph = al.create_graph_from_data(data)
            with open(saving_path, 'wb') as f:
                f.write(pickle.dumps(ugraph))


def set_observe(graph):
    for n, node in graph.nodes(data=True):
        node['observed'] = True


def remove_bidirected_edges(edges):
    ed = {}
    for a, b in edges:
        ed[" ".join(sorted([a, b]))] = (a, b)
    return list(ed.values())


def evaluate_single_graph(df_samples, graph, bn_truth, nb_repeat=3):
    testing_graph = BayesianModel()
    testing_graph.add_nodes_from(bn_truth.causal_graph.nodes())
    for edge in remove_bidirected_edges(graph.edges()):
        try:
            testing_graph.add_edge(edge[0], edge[1])
        except Exception as e:
            try:
                testing_graph.add_edge(edge[1], edge[0])
            except Exception as e:
                print(e)
                continue

    testing_graph.fit(df_samples, estimator=BayesianEstimator)
    testing_graph.check_model()
    bn_test = BayesianNetwork(testing_graph)

    set_observe(bn_test.bn)
    set_observe(bn_truth.bn)

    bn_truth.set_state_names()
    bn_test.set_state_names()

    return {
        'SID': SID(bn_truth.causal_graph, bn_test.causal_graph),
        'SHD': SHD(bn_truth.causal_graph, bn_test.causal_graph),
        'OD': np.mean([ODist(bn_truth, bn_test, 1000, discrete=True) for _ in range(nb_repeat)]),
        'ID': np.mean([IDist(bn_truth, bn_test, 1000, discrete=True) for _ in range(nb_repeat)])
    }


def evaluate_graphs(folder):
    already_done = ['alarm']

    networks = load_networks(os.path.join(folder, 'BNs'))
    graph_folder = os.path.join(folder, 'graphs')
    datasets_folder = os.path.join(folder, 'datasets')
    datasets = dict([(f.split('.')[0], os.path.join(datasets_folder, f)) for f in os.listdir(datasets_folder)])
    algos = list(set([f.split('_')[0] for f in os.listdir(graph_folder)]))

    for network in networks:
        if network in already_done:
            continue
        bn_truth = BayesianNetwork(networks[network])
        df_samples = pd.read_csv(datasets[network])
        evaluation_points = []

        for algo in algos:
            print(algo)
            test_graph_path = os.path.join(graph_folder, '_'.join([algo, network]))
            with open(test_graph_path, 'rb') as f:
                test_graph = pickle.loads(f.read())

            if 'Unnamed: 0' in test_graph.nodes():
                test_graph.remove_node('Unnamed: 0')  # problem with pickling pandas

            # print(test_graph.nodes())
            if algo == 'GS' or algo == 'PC' or algo == 'GES' or algo == 'IAMB' or algo == 'MMPC':
                mapping = dict(zip(test_graph.nodes(), df_samples))
                # mapping = {}
                # for n in test_graph.nodes():
                #     mapping[n] = str(n)
                test_graph = nx.relabel_nodes(test_graph, mapping)

            # print(test_graph.edges())
            # print([a for a in nx.topological_sort(test_graph)])
            eval_point = {'network': network,
                          'algo': algo}

            eval_point.update(evaluate_single_graph(df_samples, test_graph, bn_truth))
            evaluation_points.append(eval_point)

        filename = os.path.join(folder, 'results', network + '_results.json')
        with open(filename, 'w') as f:
            f.write(json.dumps(evaluation_points, default=to_serializable))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', dest='task', choices=['generate-datasets', 'run-algorithms', 'evaluate'], help='task to execute')
    arguments = parser.parse_args()
    folder = 'real_networks_evaluation'

    if arguments.task == 'generate-datasets':
        data = load_networks(os.path.join(folder, 'BNs'))
        generate_datasets(data, folder, nb_samples=2000)
    if arguments.task == 'run-algorithms':
        # data = load_datasets(os.path.join(folder, 'BNs'))
        run_algorithms(folder)
    else:
        evaluate_graphs(folder)
