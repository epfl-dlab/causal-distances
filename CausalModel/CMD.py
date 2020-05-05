#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import pandas as pd
from pomegranate.distributions import IndependentComponentsDistribution
from pomegranate.distributions import UniformDistribution, NormalDistribution
from pomegranate.distributions import MultivariateDistribution, DiscreteDistribution
import networkx as nx

from itertools import chain, combinations

from pyemd import emd, emd_samples
import nuts.emcee_nuts as en
from scipy import stats
import numpy as np

from .SCM import SCM
from .BayesianNetwork import BayesianNetwork


def Tau(graph_a, graph_b):
    a = nx.topological_sort(graph_a)
    b = nx.topological_sort(graph_b)
    return stats.kendalltau(a, b)[0]


def discrete_distance(arr):
    nb_variables = len(arr)
    distance_matrix = np.ones((nb_variables, nb_variables))
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


def empirical_distribution_distance(model_a, model_b, nb_samples, discrete=True):
    samples_a = model_a.sample(nb_samples)
    samples_b = model_b.sample(nb_samples)
    if discrete:
        return emd_samples(samples_a, samples_b, distance=discrete_distance)
    # return emd_samples(samples_a, samples_b, bins=2 * len(samples_a))
    return emd_samples(samples_a, samples_b)


def ODist(scm_a, scm_b, nb_samples, discrete):
    nodes_a = set([n[0] for n in scm_a.causal_graph.nodes(data=True) if n[1]['observed']])
    nodes_b = set([n[0] for n in scm_b.causal_graph.nodes(data=True) if n[1]['observed']])
    assert nodes_a == nodes_b, \
        "The two causal graphs should have the same observed nodes"

    return empirical_distribution_distance(scm_a, scm_b, nb_samples, discrete)


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def IDist_multinodes(scm_a, scm_b, nb_samples, intervention_range=None):
    nodes_a = set([n[0] for n in scm_a.causal_graph.nodes(data=True) if n[1]['observed']])
    nodes_b = set([n[0] for n in scm_b.causal_graph.nodes(data=True) if n[1]['observed']])
    assert nodes_a == nodes_b, \
        "The two causal graphs should have the same observed nodes"

    results = 0.
    nb_interventions = 0
    for nodeset in powerset(nodes_a):
        try:
            intervention_scm_a = copy.deepcopy(scm_a)
            intervention_scm_b = copy.deepcopy(scm_b)
        except Exception:
            intervention_scm_a = SCM(scm_a.causal_graph,
                                     scm_a.mechanisms,
                                     scm_a.noises_distribution)
            intervention_scm_b = SCM(scm_b.causal_graph,
                                     scm_b.mechanisms,
                                     scm_b.noises_distribution)

        for node in nodeset:
            if intervention_range:
                intervention = UniformDistribution(intervention_range[node])
            else:
                intervention = UniformDistribution(0, 1)
                # intervention = NormalDistribution(0, 1)

            intervention_scm_a.intervene(node, intervention)
            intervention_scm_b.intervene(node, intervention)

        nb_interventions += 1
        results += empirical_distribution_distance(intervention_scm_a,
                                                   intervention_scm_b, nb_samples)

    return results / float(nb_interventions)


def default_intervention_noise(scm_a, scm_b, node, discrete):
    if discrete:
        v_card_a, states_a = scm_a.get_state_space(str(node))
        v_card_b, states_b = scm_b.get_state_space(str(node))

        # assert states_a == states_b, "The BayesianNetwork have different \
        #             state spaces for node {}".format(node)

        states = set.union(set(states_a), set(states_b))
        return DiscreteDistribution(dict([(s, 1. / len(states)) for s in states]))
    else:
        return UniformDistribution(-5, 5)


def IDist(scm_a, scm_b, nb_samples, l_samples=5, discrete=False, intervention_distributions=None, add_OD=False):
    nodes_a = set([n[0] for n in scm_a.causal_graph.nodes(data=True) if n[1]['observed']])
    nodes_b = set([n[0] for n in scm_b.causal_graph.nodes(data=True) if n[1]['observed']])
    assert nodes_a == nodes_b, \
        "The two causal graphs should have the same observed nodes"

    if add_OD:
        results = ODist(scm_a, scm_b, nb_samples, discrete)
        nb_terms = 1  # Was set to zero at some point
    else:
        results = 0
        nb_terms = 0  # Was set to zero at some point

    for node in nodes_a:
        if intervention_distributions:
            intervention_noise = intervention_distributions[node]
        else:
            intervention_noise = default_intervention_noise(scm_a, scm_b, node, discrete)

        for l in intervention_noise.sample(l_samples):
            # print('Intervention: node: {}, value: {}'.format(node, k))
            try:
                intervention_scm_a = copy.deepcopy(scm_a)
                intervention_scm_b = copy.deepcopy(scm_b)
            except Exception:
                print("copy error")
                if type(scm_a) == SCM:
                    intervention_scm_a = SCM(copy.deepcopy(scm_a.causal_graph),
                                             copy.deepcopy(scm_a.mechanisms),
                                             copy.deepcopy(scm_a.noises_distribution))
                    intervention_scm_b = SCM(copy.deepcopy(scm_b.causal_graph),
                                             copy.deepcopy(scm_b.mechanisms),
                                             copy.deepcopy(scm_b.noises_distribution))
                else:
                    intervention_scm_a = BayesianNetwork(copy.deepcopy(scm_a.bn),
                                                         copy.deepcopy(scm_a.causal_graph))
                    intervention_scm_b = BayesianNetwork(copy.deepcopy(scm_b.bn),
                                                         copy.deepcopy(scm_b.causal_graph))

            intervention_scm_a.intervene(node, l)
            intervention_scm_b.intervene(node, l)

            results += ODist(intervention_scm_a, intervention_scm_b, nb_samples, discrete)

            nb_terms += 1

    return results / float(nb_terms)


# class CounterfactualNoise(MultivariateDistribution):

class CounterfactualNoise():
    def __init__(self, samples, start, nb_samples_k, burn=50):
        self.start = start
        self.burn = burn
        # samples = self.cf.sample(self.start, M=nb_samples_k, Madapt=self.burn)
        pd_samples = {}
        for i in range(samples.shape[1]):
            pd_samples[i] = samples[:, i]
        self.samples = pd.DataFrame(pd_samples)
        self.i = 0
        # print(self.samples.head())

    def sample(self, n=None):
        # current_samples = []
        # for _ in range(n):
        x = self.samples.iloc[[self.i]]
        self.i = (self.i + 1) % len(self.samples)
        return x.values
        # current_samples.append(x)
        # return current_samples

    # def __reduce__(self):

        # def CDist_multinodes(scm_a, scm_b, nb_samples_x, nb_samples_k):
        #     nodes_a = scm_a.nodes()
        #     nodes_b = scm_b.nodes()
        #     assert nodes_a == nodes_b, \
        #         "The two causal graphs should have the same nodes"
        #     n = len(nodes_a)

        #     x_distrib = IndependentComponentsDistribution([UniformDistribution(0, 1) for _ in range(n)])
        #     scm_a.precompute_log_probabilities()
        #     scm_b.precompute_log_probabilities()

        #     results = IDist_multinodes(scm_a, scm_b, nb_samples_k)

        #     for x in x_distrib.sample(nb_samples_x):
        #         print('x: ' + str(x))
        #         cf_a = en.NUTSSampler(n, scm_a.log_probability)
        #         cf_b = en.NUTSSampler(n, scm_b.log_probability)

        #         counterfactual_scm_a = copy.deepcopy(scm_a)
        #         counterfactual_scm_b = copy.deepcopy(scm_b)

        #         #print("Creating noise")
        #         a_samples = cf_a.sample(x, M=nb_samples_k, Madapt=50)
        #         b_samples = cf_b.sample(x, M=nb_samples_k, Madapt=50)

        #         counterfactual_scm_a.noises_distribution = CounterfactualNoise(a_samples, x, nb_samples_k)
        #         counterfactual_scm_b.noises_distribution = CounterfactualNoise(b_samples, x, nb_samples_k)
        #         #print('End noise')

        #         #print("Run IDist")
        #         results += IDist_multinodes(counterfactual_scm_a, counterfactual_scm_b, nb_samples_k)
        #         #print("End IDist")

        #     return results / float(nb_samples_x + 1)


def CDist(scm_a, scm_b, nb_samples_x, nb_samples_k=5, discrete=False):
    nodes_a = scm_a.nodes()
    nodes_b = scm_b.nodes()
    assert nodes_a == nodes_b, \
        "The two causal graphs should have the same nodes"
    n = len(nodes_a)

    x_distrib = IndependentComponentsDistribution([UniformDistribution(0, 1) for _ in range(n)])
    scm_a.precompute_log_probabilities()
    scm_b.precompute_log_probabilities()

    results = IDist(scm_a, scm_b, nb_samples_k, discrete=discrete)

    for x in x_distrib.sample(nb_samples_x):
        # print('x: ' + str(x))
        cf_a = en.NUTSSampler(n, scm_a.log_probability)
        cf_b = en.NUTSSampler(n, scm_b.log_probability)

        counterfactual_scm_a = copy.deepcopy(scm_a)
        counterfactual_scm_b = copy.deepcopy(scm_b)

        # print("Creating noise")
        a_samples = cf_a.sample(x, M=nb_samples_k, Madapt=200)
        b_samples = cf_b.sample(x, M=nb_samples_k, Madapt=200)

        counterfactual_scm_a.noises_distribution = CounterfactualNoise(a_samples, x, nb_samples_k)
        counterfactual_scm_b.noises_distribution = CounterfactualNoise(b_samples, x, nb_samples_k)
        # print('End noise')

        # print("Run IDist")
        results += IDist(counterfactual_scm_a, counterfactual_scm_b, nb_samples_k, discrete=discrete)
        # print("E nd IDist")

    return results / float(nb_samples_x + 1)
