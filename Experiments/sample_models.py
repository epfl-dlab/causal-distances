import numpy as np

from pomegranate.distributions import IndependentComponentsDistribution
from pomegranate.distributions import NormalDistribution
from pomegranate.distributions import GammaDistribution

from CausalModel.Generator import Generator
from CausalModel.Generator import DiscreteGenerator
from CausalModel.Mechanisms import LinearAdditive, GaussianProcess, GaussianProcessAdditive

default_structure = {
    'nb_nodes': 3,
    'density': 0.3,  # 0.4 / (n ** 1.25 / 10)
    'cycles': False,
    'fraction_observed': 1
}


########################################
# Discrete Causal Models
########################################

def generate_discrete_models(structure_parameters=default_structure, nb_models=1):
    bns = []
    for _ in range(nb_models):
        d_gen = DiscreteGenerator(structure_parameters)
        bns.append(d_gen.generate_bm())

    if nb_models == 1:
        return bns[0]
    return bns


########################################
# Continuous Causal Models
########################################

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


def default_gaussian_noise(n):
    return IndependentComponentsDistribution(
        [NormalDistribution(0, np.random.rand()) for _ in range(n)])


def non_gaussian_noise(n):
    IndependentComponentsDistribution(
        [GammaDistribution(alpha=np.random.rand(), beta=np.random.rand())
         for _ in range(n)])


def generate_linear_gaussian(structure=None, nb_models=1):
    if structure is None:
        structure = default_structure
    mechanism = default_mechanism
    noises = default_gaussian_noise(structure['nb_nodes'])

    return generate_continuous_model(structure_parameters=structure,
                                     nb_models=nb_models,
                                     mechanims_parameters=mechanism,
                                     noises=noises)


def generate_linear_non_gaussian(structure=None, nb_models=1):
    if structure is None:
        structure = default_structure
    mechanism = default_mechanism
    noises = non_gaussian_noise(structure['nb_nodes'])

    return generate_continuous_model(structure_parameters=structure,
                                     nb_models=nb_models,
                                     mechanims_parameters=mechanism,
                                     noises=noises)


def generate_GP_additive_gaussian(structure=None, nb_models=1):
    if structure is None:
        structure = default_structure
    mechanism = default_GP_Additive
    noises = default_gaussian_noise(structure['nb_nodes'])

    return generate_continuous_model(structure_parameters=structure,
                                     nb_models=nb_models,
                                     mechanims_parameters=mechanism,
                                     noises=noises)


def generate_GP_additive_non_gaussian(structure=None, nb_models=1):
    if structure is None:
        structure = default_structure
    mechanism = default_GP_Additive
    noises = non_gaussian_noise(structure['nb_nodes'])

    return generate_continuous_model(structure_parameters=structure,
                                     nb_models=nb_models,
                                     mechanims_parameters=mechanism,
                                     noises=noises)


def generate_GP(structure=None, nb_models=1):
    if structure is None:
        structure = default_structure
    mechanism = default_GP
    noises = non_gaussian_noise(structure['nb_nodes'])

    return generate_continuous_model(structure_parameters=structure,
                                     nb_models=nb_models,
                                     mechanims_parameters=mechanism,
                                     noises=noises)


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
