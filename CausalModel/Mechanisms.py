#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.gaussian_process import GaussianProcessRegressor
from pomegranate.distributions import GaussianKernelDensity

from abc import ABC, abstractmethod
import numpy as np
import scipy
import copy

from .utils import m, K, sigmoid


class Mechanism(ABC):
    @abstractmethod
    def __call__(self):
        pass

    def precompute_log_probabilities(self, noise_distribution, l=5, k=50):  # 10, 200
        self.log_probabilities = []
        for i in range(l):
            causes = [np.random.rand()] * self.nb_causes
            fitting_conditional = GaussianKernelDensity()
            x = [self(causes, n) for n in noise_distribution.sample(k)]
            fitting_conditional.fit(x)
            self.log_probabilities.append((causes, fitting_conditional))

    def log_probability(self, y, causes):
        top_n = 3
        distances = [(i, scipy.spatial.distance.euclidean(causes, c[0])) for i, c in enumerate(self.log_probabilities)]
        closest_l_prob = sorted(distances, key=lambda t: t[1], reverse=True)[:top_n]
        res = 0.
        for l_p in closest_l_prob:
            idx = l_p[0]
            res += self.log_probabilities[idx][1].log_probability(y)
        return res / float(top_n)


class LinearAdditive(Mechanism):
    def __init__(self, nb_causes, generate_random=True):
        self.nb_causes = nb_causes
        self.coeffs = []

        if generate_random:
            self.generate_random()

    def generate_random(self):
        if self.nb_causes > 0:
            self.coeffs = np.random.rand(self.nb_causes)

    def perturbate(self, epsilon, mu=0.):
        if self.nb_causes > 0:
            self.coeffs = (1 - epsilon) * self.coeffs + epsilon * np.random.uniform(1, 10, size=self.nb_causes)
            # np.random.normal(mu, epsilon, size=self.nb_causes)

    def __call__(self, causes, noise):
        if self.nb_causes > 0:
            return np.dot(causes, self.coeffs) + noise
        return noise

    # def log_probability(self, y, causes, noise_distribution, k=1000):
    #     model = smp.Model()
    #     model.add(smp.normal())
    #     y = self(x, noise)
    #     model.add(smp.normal(y, mu=y, sig=0))
    #     if self.fitting_conditional:
    #         fitting_conditional = GaussianKernelDensity()
    #         x = [self(causes, n) for n in noise_distribution.sample(k)]
    #         fitting_conditional.fit(x)
    #     return fitting_conditional.log_probability(y)


class GaussianProcessAdditive(Mechanism):
    def __init__(self, nb_causes, parameters, generate_random=True):
        self.nb_causes = nb_causes
        self.nb_points = parameters['nb_points']
        self.variance = parameters['variance']

        if generate_random:
            self.generate_random()

    def generate_random(self):
        if self.nb_causes == 0:
            return

        X = []
        for _ in range(self.nb_causes):
            X.append(np.linspace(0, 1, self.nb_points))
        X = np.array(X).T

        ys = sigmoid(np.random.multivariate_normal(m(self.nb_points),
                                                   K(X, X, self.variance)))
        self.gpr = GaussianProcessRegressor()
        self.gpr.fit(X, ys)

    def __call__(self, causes, noise):
        if self.nb_causes == 0:
            return noise
        causes = causes.reshape(1, -1)  # Single example
        return self.gpr.predict(causes)[0] + noise


class GaussianProcess(Mechanism):
    def __init__(self, nb_causes, parameters, generate_random=True):
        self.nb_causes = nb_causes
        self.nb_points = parameters['nb_points']
        self.variance = parameters['variance']

        if generate_random:
            self.generate_random()

    def generate_random(self):
        if self.nb_causes == 0:
            return

        X = []
        for _ in range(self.nb_causes + 1):
            X.append(np.linspace(0, 1, self.nb_points))
        X = np.array(X).T

        ys = sigmoid(np.random.multivariate_normal(m(self.nb_points),
                                                   K(X, X, self.variance)))
        self.gpr = GaussianProcessRegressor()
        self.gpr.fit(X, ys)

    def __call__(self, causes, noise):
        if self.nb_causes == 0:
            return noise
        X = np.hstack((causes, noise))
        X = X.reshape(1, -1)
        return self.gpr.predict(X)[0]
        # causes = causes.reshape(1, -1)  # Single example
        # return self.gpr.predict(causes)[0] + noise
