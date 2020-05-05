from CausalModel.Generator import Generator, DiscreteGenerator
from CausalModel.Mechanisms import LinearAdditive, GaussianProcess, GaussianProcessAdditive
from pomegranate.distributions import IndependentComponentsDistribution
from pomegranate.distributions import NormalDistribution, GammaDistribution
import numpy as np
import networkx as nx
from cdt.causality.graph import LiNGAM, CAM, CCDr, GES, PC, GS, IAMB, MMPC
import copy
import os
import pickle
import json
from functools import singledispatch
import pandas as pd
import argparse

algorithms = {'LiNGAM': LiNGAM, 'CAM': CAM, 'CCDr': CCDr, 'GS': GS, 'GES': GES, 'PC': PC, 'IAMB': IAMB, 'MMPC': MMPC}

##########################################
# DATASETS GENERATION
##########################################

default_structure = {
    'nb_nodes': 3,
    'density': 0.3,  # 0.4 / (n ** 1.25 / 10)
    'cycles': False,
    'fraction_observed': 1
}

default_mechanism = {
    'function': LinearAdditive,
    'parameters': None
}

default_GP_Additive = {
    'function': GaussianProcessAdditive,
    'parameters': {'nb_points': 1000,
                   'variance': 1}
}

default_GP = {
    'function': GaussianProcess,
    'parameters': {'nb_points': 1000,
                   'variance': 1}
}


def generate_discrete_model(structure=default_structure, nb_models=1):
    bns = []
    for _ in range(nb_models):
        d_gen = DiscreteGenerator(structure)
        bns.append(d_gen.generate_bm())

    if nb_models == 1:
        return bns[0]
    return bns


def default_gaussian_noise(n):
    return IndependentComponentsDistribution([NormalDistribution(0, np.random.rand()) for _ in range(n)])


def non_gaussian_noise(n):
    IndependentComponentsDistribution([GammaDistribution(alpha=np.random.rand(), beta=np.random.rand())
                                       for _ in range(n)])


def generate_continuous_model(structure_parameters=default_structure,
                              nb_models=1,
                              mechanims_parameters=default_mechanism,
                              noises=None):

    scms = []
    for _ in range(nb_models):
        gen = Generator(structure_parameters=structure_parameters,
                        mechanims_parameters=mechanims_parameters,
                        noise=noises)
        scms.append(gen.generate_scm())
    if nb_models == 1:
        return scms[0]
    return scms


def generate_linear_gaussian(structure=default_structure, nb_models=1):
    mechanism = default_mechanism
    noises = default_gaussian_noise(structure['nb_nodes'])

    return generate_continuous_model(structure_parameters=structure,
                                     nb_models=nb_models,
                                     mechanims_parameters=mechanism,
                                     noises=noises)


def generate_linear_non_gaussian(structure=default_structure, nb_models=1):
    mechanism = default_mechanism
    noises = non_gaussian_noise(structure['nb_nodes'])

    return generate_continuous_model(structure_parameters=structure,
                                     nb_models=nb_models,
                                     mechanims_parameters=mechanism,
                                     noises=noises)


def generate_GP_additive_non_gaussian(structure=default_structure, nb_models=1):
    mechanism = default_GP_Additive
    noises = non_gaussian_noise(structure['nb_nodes'])

    return generate_continuous_model(structure_parameters=structure,
                                     nb_models=nb_models,
                                     mechanims_parameters=mechanism,
                                     noises=noises)


def generate_GP(structure=default_structure, nb_models=1):
    mechanism = default_GP
    noises = non_gaussian_noise(structure['nb_nodes'])

    return generate_continuous_model(structure_parameters=structure,
                                     nb_models=nb_models,
                                     mechanims_parameters=mechanism,
                                     noises=noises)


def generate_dataset(model, nb_samples=500):
    X = model.sample(nb_samples)
    X = X.sample(frac=1).reset_index(drop=True)
    return X


def generate_random_dataset(generator, nb_samples=500):
    model = generator(nb_models=1)
    return generate_dataset(model, nb_samples)


def sample_datasets(list_generators, list_nb_nodes, list_nb_samples, dst_folder='baseline_evaluation/datasets/'):
    for nnodes in list_nb_nodes:
        structure_parameters = copy.deepcopy(default_structure)
        structure_parameters['nb_nodes'] = nnodes
        # structure_parameters['density'] = 0.4 / (nnodes ** 1.25 / 10)
        print(nnodes)
        for generator_name, generator in list_generators.items():
            print(' -- ' + generator_name)
            scm = generator(structure=structure_parameters, nb_models=1)
            scm_name = '_'.join([generator_name, str(nnodes)])
            scm_dst = dst_folder + 'SCMs/' + scm_name
            with open(scm_dst, 'wb') as f:
                f.write(pickle.dumps(scm))

            for n_samples in list_nb_samples:
                print(' ++++ ' + str(n_samples))
                X = scm.sample(n_samples)
                name = '_'.join([generator_name, str(nnodes), str(n_samples)])
                name += '.csv'
                dst = dst_folder + name
                X.to_csv(dst)


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


##########################################
# DATASETS -> GRAPHS
##########################################

def load_datasets(dst_folder):
    scms = {}
    scm_folder = dst_folder + 'SCMs/'
    scm_files = [f for f in os.listdir(scm_folder) if not f.startswith('.')]
    for scm_file in scm_files:
        filepath = dst_folder + 'SCMs/' + scm_file
        with open(filepath, 'rb') as f:
            scm = pickle.loads(f.read())

        scms[scm_file.split('.')[0]] = scm

    data = []
    datasets_files = [f for f in os.listdir(dst_folder) if not f.startswith('.')]
    for data_file in datasets_files:
        data_path = dst_folder + data_file
        if os.path.isdir(data_path):
            continue
        generator, nnodes, nb_samples = data_file.split('.')[0].split('_')
        # if nnode_filter:
        #     if int(nnodes) != int(nnode_filter):
        #         continue
        #     if int(nb_samples) < 101:
        #         continue

        X = pd.read_csv(data_path, index_col=0)

        data.append({'filename': data_file,
                     'dataset': X,
                     'SCM': copy.deepcopy(scms['_'.join([generator, nnodes])])})

    return data


def run_algorithms(data, dst_folder, filter_nnodes=None):
    for dataset in data:
        print(dataset['filename'])
        generator, nnodes, nb_samples = dataset['filename'].split('.')[0].split('_')
        if int(nnodes) == filter_nnodes:
            continue
        for algo_name, algo in algorithms.items():
            print(algo.__name__)
            saving_file = algo_name + '_' + dataset['filename']
            saving_path = dst_folder + saving_file
            if os.path.isfile(saving_path):
                continue
            al = algo()
            # print(dataset['dataset'].head())
            # exit()
            ugraph = al.predict(dataset['dataset'])
            with open(saving_path, 'wb') as f:
                f.write(pickle.dumps(ugraph))


##########################################
# GRAPHS -> EVALUATION
##########################################


from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from CausalModel.BayesianNetwork import BayesianNetwork
from CausalModel.CMD import ODist, IDist, Tau
from cdt.metrics import SID, SHD
import collections


def set_observe(graph):
    for n, node in graph.nodes(data=True):
        node['observed'] = True


def remove_bidirected_edges(edges):
    ed = {}
    for a, b in edges:
        ed[" ".join(sorted([a, b]))] = (a, b)
    return list(ed.values())


def evaluate_single_graph(df_samples, graph, bn_truth, nb_repeat=1):
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

    # print(bn_test.causal_graph.edges())
    # exit()

    # model_test = BayesianModel()
    # model_test.add_nodes_from(graph.nodes())
    # for a in graph_diff:
    #     model_test.add_node(a)
    # for edge in remove_bidirected_edges(graph.edges()):
    #     try:
    #         model_test.add_edge(edge)
    #     except Exception as e:
    #         rev_edge = (edge[1], edge[0])
    #         try:
    #             model_test.add_edge(rev_edge)
    #         except Exception as e:
    #             continue

    # model_test.fit(df_samples, estimator=BayesianEstimator)
    # model_test.check_model()
    # bn_test = BayesianNetwork(model_test)

    set_observe(bn_test.bn)
    set_observe(bn_truth.bn)

    bn_truth.set_state_names()
    bn_test.set_state_names()

    # mapping = dict((i, str(i)) for i in bn_truth.bn.nodes())
    # nx.relabel_nodes(bn_truth.bn.graph, mapping)
    # nx.relabel_nodes(bn_test.bn.graph, mapping)
    # print()

    # print(bn_truth.causal_graph.edges())
    # print(testing_graph.edges())

    # print(SID(bn_truth.causal_graph, bn_test.causal_graph))
    # print(SHD(bn_truth.causal_graph, bn_test.causal_graph))

    return {
        'SID': SID(bn_truth.causal_graph, bn_test.causal_graph),
        'SHD': SHD(bn_truth.causal_graph, bn_test.causal_graph),
        'Tau': Tau(bn_truth.causal_graph, bn_test.causal_graph),
        'OD': np.mean([ODist(bn_truth, bn_test, 1000, discrete=True) for _ in range(nb_repeat)]),
        'ID': np.mean([IDist(bn_truth, bn_test, 1000, discrete=True) for _ in range(nb_repeat)])
    }
    # Test if directed


def samples_to_df(data_samples, bins=5):
    def myfunc(x, col):
        return (x[col].right + x[col].left) / 2.

    df = pd.DataFrame()
    for col in data_samples.columns:
        df[col] = pd.cut(data_samples[col], bins)

    for col in data_samples.columns:
        df[col] = df.apply(lambda x: myfunc(x, col), axis=1)
    return df
    # return df.rename(int, axis='columns')


def evaluate_graphs(data, graph_folder, dst_folder, nnode_filter=40):
    evaluation_points = []
    for dataset in data:
        print(dataset['filename'])
        generator, nnodes, nb_samples = dataset['filename'].split('.')[0].split('_')

        if int(nnodes) != nnode_filter:
            continue

        df_samples = samples_to_df(dataset['dataset'], bins=3) # Used 5 bins when 5 nodes
        # print(df_samples.head())
        scm = dataset['SCM']

        ground_truth = BayesianModel()
        ground_truth.add_nodes_from([str(i) for i in scm.causal_graph.nodes()])
        ground_truth.add_edges_from([(str(a), str(b)) for a, b in scm.causal_graph.edges()])
        ground_truth.fit(df_samples, estimator=BayesianEstimator)
        ground_truth.check_model()
        bn_truth = BayesianNetwork(ground_truth)

        for algo_name, algo in algorithms.items():
            print(algo.__name__)
            # if algo.__name__ == 'CAM':
            #     continue

            algo_file = algo_name + '_' + dataset['filename']
            algo_path = graph_folder + algo_file

            if not os.path.isfile(algo_path):
                continue

            with open(algo_path, 'rb') as f:
                graph = pickle.loads(f.read())

            if 'Unnamed: 0' in graph.nodes():
                graph.remove_node('Unnamed: 0')  # problem with pickling pandas

            if algo.__name__ == 'GS' or algo.__name__ == 'IAMB' or algo.__name__ == 'MMPC':
                mapping = {}
                for n in graph.nodes():
                    mapping[n] = n.split('X')[1]
                graph = nx.relabel_nodes(graph, mapping)

            eval_point = {'generator': generator,
                          'nnodes': nnodes,
                          'nb_samples': nb_samples,
                          'algo': algo_name}

            eval_point.update(evaluate_single_graph(df_samples, graph, bn_truth))
            evaluation_points.append(eval_point)

    filename = os.path.join(dst_folder, 'results-n' + str(nnode_filter) + '.json')
    with open(filename, 'w') as f:
        f.write(json.dumps(evaluation_points, default=to_serializable))


# if __name__ == '__main__':
#     list_generators = {
#         # 'Discrete': generate_discrete_model,
#         'linGauss': generate_linear_gaussian,
#         'linNGauss': generate_linear_non_gaussian,
#         'GPAddit': generate_GP_additive_non_gaussian,
#         'GP': generate_GP}
#     list_nb_samples = [200, 500, 1000, 2000]
#     list_nb_nodes = [5, 10, 20, 40]
#     sample_datasets(list_generators, list_nb_nodes, list_nb_samples, dst_folder='baseline_evaluation/datasets/')


# if __name__ == '__main__':
#     data = load_datasets('baseline_evaluation/datasets/')
#     run_algorithms(data, 'baseline_evaluation/graphs/', 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', dest='task', choices=['generate-datasets', 'run-algorithms', 'evaluate'], help='task to execute')
    arguments = parser.parse_args()

    if arguments.task == 'generate-datasets':
        list_generators = {
            'Discrete': generate_discrete_model,
            'linGauss': generate_linear_gaussian,
            'linNGauss': generate_linear_non_gaussian,
            'GPAddit': generate_GP_additive_non_gaussian,
            'GP': generate_GP}
        list_nb_samples = [200, 500, 1000, 2000]
        list_nb_nodes = [5, 10, 20, 40]
        sample_datasets(list_generators, list_nb_nodes, list_nb_samples, dst_folder='baseline_evaluation/datasets/')
    elif arguments.task == 'run-algorithms':
        data = load_datasets('baseline_evaluation/datasets/')
        run_algorithms(data, 'baseline_evaluation/graphs/', 40)
    else:
        data = load_datasets('baseline_evaluation/datasets/')
        evaluate_graphs(data, 'baseline_evaluation/graphs/', 'baseline_evaluation/results/', nnode_filter=10)
