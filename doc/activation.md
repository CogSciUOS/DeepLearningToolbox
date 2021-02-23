# Activation Map Analysis

Analyzing activation maps in deep networks can help to gain a better
understanding of the features encoded by activation patterns.

# Univariate analysis

Univariate analysis refers to the analysis of activation pattern in a
single layer of a neural network. Here questions of redundancy and
entanglement arise.

## Principal Component Analysis (PCA)

PCA is a standard method, realized as an orthogonal
transformation. Principal components are ordered by the amount of
variance they capture, allowing for a simple dimensionality reduction
scheme.  This can also be used to estimate the number of relevant
dimensions of activations in a network layer.

* Harold Hotelling (1933): *Analysis of a complex of statistical
  variables into principal components*, Journal of Educational
  Psychology 24(6), 417-441,
  DOI: [10.1037/h0071325](https://doi.org/10.1037/h0071325)

# Bivariate analysis

Comparing activation patterns of two layers, for example:
* of the same network, at different times during training
* of the same network, trained with different initializations
* of different layers of the same network
* of layers in different networks (do different networks learn the same representation?)

Several approaches addressing such kind of questions have been brought
forward in literature. Some selection:

* Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John Hopcroft
  (2016): *Convergent Learning: Do different neural networks learn the
  same representations?* This paper applies different methods to map
  units or clusters of units for a pair of layers.

## Canonical Correlation Analysis (CCA)


* Harold Hotelling (1936): *Relations Between Two Sets of Variates*,
  Biometrika 28(3/4), 321-377.

### SVCCA

Raghu et al. (2017) determine the similarity between two layers by
first pruning the layers with a singular value decomposition
preprocessing step, and then applying canonical correlation analysis
(CCA) to the reduced layers. They assess the similarity of $L_1$ and
$L_2$ by the mean correlation coefficient.

* Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha
  Sohl-Dickstein (2017): *SVCCA: Singular Vector Canonical Correlation
  Analysis for Deep Learning Dynamics and Interpretability*


### Projection weighted CCA

* Ari S. Morcos, Maithra Raghu, and Samy Bengio (2018): [*Insights on
  Representational Similarity in Neural Networks with Canonical
  Correlation*](http://dl.acm.org/citation.cfm?id=3327345.3327475),
  in: Proceedings of the 32Nd International Conference on Neural
  Information Processing Systems (NIPS), 5732-5741.

## Centered Kernel Analysis (CKA)

CKA is used by Kornblith et al. (2019) to identify correspondences
between representations in networks trained from different
initializations.

* Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton (2019):
  [*Similarity of Neural Network Representations Revisited*](http://proceedings.mlr.press/v97/kornblith19a.html), in:
  Proceedings of the 36th International Conference on Machine Learning (ICML),
  3519-3529.

## The RV coefficient

The RV coefficient is a relatively old approach to linear mutlivariate
statistical analysis.

* P. Robert and Y. Escoufier (1976): *A Unifying Tool for Linear
  Multivariate Statistical Methods: The RV-Coefficient*, Journal of
  the Royal Statistical Society. Series {C} (Applied Statistics), 25(3),
  257-265, DOI: [10.2307/2347233](https://doi.org/10.2307/2347233).

### The modified RV coefficient (RV2)

The modified RV coefficient is introduced by Thompson et al. (2019) to
compare layers in networks trained in different settings.

* Jessica Thompson and Yoshua Bengio and Marc Schönwiesner (2019):
  *The effect of task and training on intermediate representations in
  convolutional neural networks revealed with modified RV similarity
  analysis*,
  DOI: [10.32470/CCN.2019.1300-0](https://doi.org/10.32470/CCN.2019.1300-0).
