#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import networkx as nx
from pomegranate.distributions import MultivariateDistribution

from .Mechanisms import Mechanism
from .CausalModel import CausalModel


class SCM(CausalModel):
    def __init__(self, causal_graph, mechanisms, noises_distribution):
        self.__check(causal_graph, mechanisms, noises_distribution)

        self.causal_graph = causal_graph
        self.mechanisms = mechanisms
        self.noises_distribution = noises_distribution

        self.init_nodes()

    def __check(self, causal_graph, mechanisms, noises_distribution):
        assert isinstance(causal_graph, nx.DiGraph), \
            "causal_graph should be of type nx.Digraph"

        assert isinstance(noises_distribution, MultivariateDistribution), \
            "The noise distribution should be a pomegranate MultivariateDistribution"

        for mechanism in mechanisms:
            assert isinstance(mechanism, Mechanism), \
                "The mechanisms should be of type Mechanism"

        # assert noises_distribution.d == len(causal_graph.nodes()), \
        #     "There should be one noise distribution for each node in the causal graph"

        assert len(mechanisms) == len(causal_graph.nodes()), \
            "There should one mechanism for each node in the causal graph"

        for n in causal_graph.nodes():
            assert mechanisms[n].nb_causes == len(list(causal_graph.predecessors(n))), \
                "nb_causes in the mechanism different from number of parents in the graph"

        for _, n in causal_graph.nodes(data=True):
            assert 'observed' in n, \
                "Each node should have a boolean 'observed' attribute"

    def nodes(self):
        return self.causal_graph.nodes()

    def init_nodes(self, nodes_to_init=None):
        if nodes_to_init is None:
            nodes_to_init = self.causal_graph.nodes(data=True)

        for n in nodes_to_init:
            n[1]['value'] = None

        self.has_definite_state = False

    # def intervene(self, intervention_node, noise_distribution=None):
    #     edges_to_remove = list(self.causal_graph.in_edges(intervention_node))
    #     self.causal_graph.remove_edges_from(edges_to_remove)
    #     self.mechanisms[intervention_node].nb_causes = 0
    #     if noise_distribution:
    #         self.causal_graph.nodes(data=True)[intervention_node]['intervention_distribution'] = noise_distribution
    #     else:
    #         # self.graph.nodes()[node]['intervention_distribution'] = Uniform()
    #         pass

    def intervene(self, intervention_node, intervention_value):
        edges_to_remove = list(self.causal_graph.in_edges(intervention_node))
        self.causal_graph.remove_edges_from(edges_to_remove)
        self.mechanisms[intervention_node].nb_causes = 0

        self.causal_graph.nodes(data=True)[intervention_node]['value'] = intervention_value

    def precompute_log_probabilities(self):
        for n, m in enumerate(self.mechanisms):
            m.precompute_log_probabilities(self.noises_distribution[n])

    def log_probability(self, x):
        log_l = 0
        for n in nx.topological_sort(self.causal_graph):
            causes = sorted(self.causal_graph.predecessors(n))
            if len(list(causes)) == 0:
                log_l += self.noises_distribution[n].log_probability(x[n])
                continue

            causes_values = [x[i] for i in causes]
            log_l += self.mechanisms[n].log_probability(x[n], causes_values)

        return log_l

    def compute_state(self, noises=None):
        if noises is None:
            noises = self.noises_distribution.sample(1)[0]

        nodes = self.causal_graph.nodes(data=True)
        # print(nodes)

        for n in nx.topological_sort(self.causal_graph):
            # print(nodes[n])
            if nodes[n]['value'] is not None:
                continue

            # print('computing')
            causes = sorted(self.causal_graph.predecessors(n))
            causes_values = np.array([nodes[pa_x]['value'] for pa_x in causes])

            if 'intervention_distribution' in nodes[n]:
                assert len(causes_values) == 0, "A node has been intervened on but still \
                                                has parents in the causal graph"

                noise = nodes[n]['intervention_distribution'].sample(1)[0]
            else:
                noise = noises[n]

            # print(noise)
            # print('{} :: {}'.format(causes, causes_values))
            nodes[n]['value'] = self.mechanisms[n](causes_values, noise)

        self.has_definite_state = True
        return np.array([nodes[n]['value'] for n in range(len(nodes))])

    def sample(self, nb_samples=1, init=False, observed=True):
        samples = []

        if init:
            self.init_nodes()

        nodes_to_init = [node for node in self.causal_graph.nodes(data=True) if node[1]['value'] is None]

        # if len(nodes_to_init) != len(list(self.causal_graph.nodes())):
        #     print("Some nodes already have fixed values. The sampling will only vary the values of other nodes")

        for _ in range(nb_samples):
            samples.append(self.compute_state())
            self.init_nodes(nodes_to_init)

        samples = np.array(samples)

        pd_samples = {}
        for i in range(len(self.causal_graph.nodes())):
            # print(self.causal_graph.nodes(data=True))
            if observed and not self.causal_graph.nodes(data=True)[i]['observed']:
                continue
            pd_samples[i] = samples[:, i]

        return pd.DataFrame(pd_samples)
