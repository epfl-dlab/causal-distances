#!/usr/bin/env python
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np

from .CausalModel import CausalModel

from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD


class BayesianNetwork(CausalModel):
    def __init__(self, bn, causal_graph=None):
        self.bn = bn
        if causal_graph:
            self.causal_graph = causal_graph
        else:
            self.causal_graph = nx.DiGraph()
            for n in self.bn.nodes():
                self.causal_graph.add_node(n)
            for e in self.bn.edges():
                self.causal_graph.add_edge(e[0], e[1])
            for i, node in self.causal_graph.nodes(data=True):
                node['observed'] = True

    def sample(self, nb_sample=1):
        # sampling of pgmpy samples the index of the values
        # Here we convert back this index to the actual value
        def convert(samples):
            for col in samples.columns:
                _, states = self.get_state_space(col)
                samples[col] = samples[col].apply(lambda x: states[x])
            return samples

        inference = BayesianModelSampling(self.bn)
        samples = inference.forward_sample(size=nb_sample)

        return convert(samples)

    def set_state_names(self):
        for cpd in self.bn.get_cpds():
            for i in cpd.state_names:
                cpd.state_names[i] = list(range(len(cpd.state_names[i])))
            # cpd.state_names[cpd.variable] = list(range(cpd.variable_card))
            # print(cpd.state_names)

    def nodes(self):
        return self.causal_graph.nodes()

    def get_state_space(self, i):
        for cpd in self.bn.get_cpds():
            if cpd.variable == i:
                return cpd.variable_card, cpd.state_names[i]

    def intervene(self, intervention_node, intervention_value=None):
        intervention_node = str(intervention_node)

        v_card, states = self.get_state_space(intervention_node)

        values = [float(s == intervention_value) for s in states]
        values = np.array([values])

        self.bn.remove_cpds(intervention_node)

        if np.sum(values) == 0:
            self.bn.remove_node(intervention_node)
            self.bn.add_node(intervention_node)
            cpd = TabularCPD(variable=intervention_node, variable_card=1, values=np.array([[1.]]), state_names={intervention_node: [intervention_value]})
            self.bn.add_cpds(cpd)
        else:
            cpd = TabularCPD(variable=intervention_node, variable_card=v_card, values=values)

            edges = [(e[0], e[1]) for e in self.bn.in_edges(intervention_node)]
            for n_in, n_out in edges:
                self.bn.remove_edge(n_in, n_out)

            self.bn.add_cpds(cpd)

        # print(self.bn.nodes())
        # for a in self.bn.get_cpds():
        #     print(a)
        # print(self.bn.edges())
        self.bn.check_model()

    def __instancecheck__(self, inst):
        return hasattr(inst, 'bn')
