#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

from sklearn.metrics.pairwise import euclidean_distances


def K(xs, ys, sigma):
    xnorm = np.power(euclidean_distances(xs, xs), 2)
    return (sigma ** 2) * np.exp(-xnorm / (2.0))


def m(nb_points):
    return np.zeros(nb_points)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def draw_graph(causal_graph):
    color_map = []
    for i, node in causal_graph.nodes(data=True):
        if node['observed']:
            color_map.append('blue')
        else:
            color_map.append('red')
    nx.draw_networkx(causal_graph, node_color=color_map, with_labels=True)
