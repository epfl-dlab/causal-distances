#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
import random

from pomegranate.distributions import IndependentComponentsDistribution
from pomegranate.distributions import NormalDistribution

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np

from .Mechanisms import LinearAdditive
from .SCM import SCM
from .BayesianNetwork import BayesianNetwork

default_structure = {
    'nb_nodes': 6,
    'density': 0.3,  # 0.4 / (n ** 1.25 / 10)
    'cycles': False,
    'fraction_observed': 1
}

default_mechanism = {
    'function': LinearAdditive,
    'parameters': None
}


def default_noise(n):
    return IndependentComponentsDistribution([NormalDistribution(0, 0.1) for _ in range(n)])


class DiscreteGenerator(object):
    def __init__(self, structure_parameters=None):
        self.structure_parameters = structure_parameters if structure_parameters else default_structure

    def generate_bm(self):
        self.generate_structure()
        self.generate_cpds()
        self.bn = BayesianNetwork(self.model, self.graph)
        return self.bn

    def generate_structure(self):
        self.generate_dag()
        self.select_observable()

    def generate_cpds(self):
        model = BayesianModel([(str(a), str(b)) for a, b in self.graph.edges()])

        variable_cards = {}
        cpds = []
        for n in nx.topological_sort(self.graph):
            causes = sorted(self.graph.predecessors(n))
            variable_card = random.choice([2, 3, 4, 5])
            variable_cards[n] = variable_card
            if len(causes) == 0:
                values = np.random.rand(1, variable_card)
                values = values / np.sum(values)
                cpd = TabularCPD(variable=str(n), variable_card=variable_card,
                                 values=values)
                cpds.append(cpd)
            else:
                evidence_card = [variable_cards[i] for i in causes]
                values = np.random.rand(variable_card, np.prod(evidence_card))
                values = values / np.sum(values, axis=0)
                cpd = TabularCPD(variable=str(n), variable_card=variable_card,
                                 values=values,
                                 evidence=[str(a) for a in causes],
                                 evidence_card=evidence_card)
                cpds.append(cpd)

        model.add_cpds(*cpds)
        model.check_model()

        self.model = model

    def generate_dag(self):
        random_graph = nx.fast_gnp_random_graph(self.structure_parameters['nb_nodes'],
                                                self.structure_parameters['density'],
                                                directed=False)

        if nx.number_connected_components(random_graph) > 1:
            self.structure_parameters['density'] += 0.1 * self.structure_parameters['density']
            return self.generate_dag()

        self.graph = nx.DiGraph(
            [
                (u, v) for (u, v) in random_graph.edges() if u < v
            ]
        )

    def select_observable(self):
        candidates = [f for f in self.graph.nodes()]
        nb_observed = int(self.structure_parameters['fraction_observed'] * len(candidates))
        observed = random.sample(candidates, nb_observed)
        for i, node in self.graph.nodes(data=True):
            if i in observed:
                node['observed'] = True
            else:
                node['observed'] = False


class Generator(object):
    def __init__(self, structure_parameters=None, mechanims_parameters=None, noise=None):
        self.structure_parameters = structure_parameters if structure_parameters else default_structure
        self.mechanims_parameters = mechanims_parameters if mechanims_parameters else default_mechanism
        self.noise = noise if noise else default_noise(self.structure_parameters['nb_nodes'])

    def generate_scm(self):
        self.generate_structure()
        self.generate_mechanisms()
        return self.get_scm()

    def generate_structure(self):
        self.generate_dag()
        self.add_cycles()
        self.select_observable()

    def generate_mechanisms(self):
        self.mechanisms = [None] * len(self.graph.nodes())
        for node in self.graph.nodes():
            nb_causes = len(list(self.graph.predecessors(node)))
            function = self.mechanims_parameters['function']
            parameters = self.mechanims_parameters['parameters']
            if parameters:
                mechanism = function(nb_causes=nb_causes, parameters=parameters, generate_random=True)
            else:
                mechanism = function(nb_causes=nb_causes, generate_random=True)
            self.mechanisms[node] = mechanism

    def get_scm(self, noise=None):
        if noise:
            self.noise = noise
        return SCM(self.graph, self.mechanisms, self.noise)

    def generate_dag(self):
        random_graph = nx.fast_gnp_random_graph(self.structure_parameters['nb_nodes'],
                                                self.structure_parameters['density'],
                                                directed=False)

        if nx.number_connected_components(random_graph) > 1:
            return self.generate_dag()

        self.graph = nx.DiGraph(
            [
                (u, v) for (u, v) in random_graph.edges() if u < v
            ]
        )

    def add_cycles(self):
        if not self.structure_parameters['cycles']:
            return
        candidates = [f for f in self.graph.nodes() if self.G.in_degree(f) > 0]
        cycle_len = random.choice(range(2, 4))
        cycle_nodes = random.sample(candidates, cycle_len)
        self.graph.add_cycle(cycle_nodes)

    def select_observable(self):
        candidates = [f for f in self.graph.nodes()]
        nb_observed = int(self.structure_parameters['fraction_observed'] * len(candidates))
        observed = random.sample(candidates, nb_observed)
        for i, node in self.graph.nodes(data=True):
            if i in observed:
                node['observed'] = True
            else:
                node['observed'] = False
