# Spatial VaDE

This repository contains a modification of the model *Variational Deep Embedding* (VaDE).
We built upon the code basis from [https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch](https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch), which is a PyTorch implementation of the original model, for which the code was published at [https://github.com/slim1017/VaDE](https://github.com/slim1017/VaDE).

The idea was to leverage VaDE for hyperspectral image segmentation by informing the clustering about the spatial information given by the image domain.
To this end, we tried different things:

- adding the total variation (TV) to the objective function of VaDE to enforce neighboring pixels having the same class;
- using the learned latent space embeddings from the pretraining as input for the VaDE;
- adding a second input layer that gets the pixel indices as input to learn the right weighting between spectral and spatial information;
- implementing the SGVB estimator for numbers of Monte Carlo samplings $L > 1$ (the original implementation only used $L = 1$);
- input patches instead of raw spectra to also exploit the information given by the surroundings of the pixels.

The original VaDE model was published and explained in the article:

Zhuxi Jiang, Yin Zheng, Huachun Tan, Bangsheng Tang, Hanning Zhou. Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering. *Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence* (IJCAI-17), DOI: [10.24963/ijcai.2017/273](https://doi.org/10.24963/ijcai.2017/273), pp. 1965-1972, 2017.
