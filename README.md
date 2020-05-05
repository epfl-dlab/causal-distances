# Causal Distances

Causal discovery, the task of automatically constructing a causal model from data, is of major significance across the sciences.
Evaluating the performance of causal discovery algorithms should ideally involve comparing the inferred models to ground-truth models available for benchmark datasets, which in turn requires a notion of distance between causal models.
While such distances have been proposed previously, they are limited by focusing on graphical properties of the causal models being compared.
Here, we overcome this limitation by defining distances derived from the causal distributions induced by the models, rather than exclusively from their graphical structure.
Pearl and Mackenzie (2018) have arranged the properties of causal models in a hierarchy called the ''ladder of causation'' spanning three rungs: observational, interventional, and counterfactual. Following this organization, we introduce a hierarchy of three distances, one for each rung of the ladder.
These new definitions are intuitively appealing as well as efficient to compute approximately.

This repository contains the implementation of the causal distances and code to reproduce the experiments described in the paper.


----
# Installation
    virtualenv causal-distances
    source causal-distances/bin/activate
    pip3 install -r requirements.txt

The experiments rely on the causal-discovery-toolbox which requires the R implementations of causal discovery systems and metrics like SID and SHD.
Please check their [documentation](https://github.com/FenTechSolutions/CausalDiscoveryToolbox) to install them.

----
# Usage

* The implementation of causal distance is available at [Causal Model Distances](CausalModel/CMD.py)

* The replication of the case-study motivating the need for new causal distances is available: [Case-study](Experiments/Examples.ipynb)

* The properties of causal distances are studied: [Causal Distances Properties](Experiments/CausalDistances-Properties.ipynb)

* The evaluation of existing causal discovery systems is performed at:
[Causal Discovery Evaluation](Experiments/CausalDiscovery-comparison.ipynb). The evaluation data is available [here](https://drive.google.com/open?id=1KNdS11OFOPPv4tbVt39CQhjwCTsX-BAP)

## Contact
maxime.peyrard@epfl.ch


