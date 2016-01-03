`torch-dngo`
============

Quick and dirty torch implementation of the network featured in
**Scalable Bayesian optimization using deep neural networks**
by [Snoek et al. (ICML 2015)](http://arxiv.org/pdf/1502.05700v2.pdf).

Note on hyperparameters: the `alpha` and `beta` hypers are neither fit or
inferred here because the marginal likelihood (Eq. 8) and its gradient still
needs to be implemented.
