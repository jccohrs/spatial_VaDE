# %% [markdown]
# # Analyze VaDE and its results

# %% [markdown]
# ## With spatial information

# %% [markdown]
# ### One spectral, one spatial input layer

# %% [markdown]
# Idea: Use two input layers, one for spatial, one for spectral information. The outputs are then added and the result further processed to give similar weighting to spectral and spatial information.
#
# Default initialization is to draw random values from $$\mathcal{U}(-\sqrt{k}, \sqrt{k})$$ where $$k = \frac{1}{infeatures}$$.

# %%
import numpy as np
from math import sqrt
from vade import VaDE
from msiplib.io import read_image
import matplotlib.pyplot as plt

# %% [markdown]
# #### SalinasA

# %%
im = read_image("/Users/jan/sciebo/data/segmentation/SalinasA/SalinasA.nc")
vade = VaDE(204, 10, 6, [500, 500, 2000], [2000, 500, 500], im.shape[:-1], 1, 0, ".", 0, True)
vade.load_model(
    "/Users/jan/sciebo/output/segmentation/vade/pure/SalinasA/2022-7-1-17-40-58-ep1000_ldim10_reg0_spat1/vade_params-ep1000_ldim10_reg0_spat1.pt"
)
spatial_enc = vade.spatial_enc[0].weight.detach().numpy()
spectral_enc = vade.spectral_enc[0].weight.detach().numpy()
print(f"Min weight spatial encoder: {spatial_enc.min()}")
print(f"Max weight spatial encoder: {spatial_enc.max()}")
print(f"Mean weight spatial encoder: {spatial_enc.mean()}")
print(f"Std. dev. weights spatial encoder: {spatial_enc.std()}")
print(f"Initial weights of spatial encoder drawn uniformly from interval: ({-sqrt(1 / 2)}, {sqrt(1 / 2)}).")
print(f"Min weight spectral encoder: {spectral_enc.min()}")
print(f"Max weight spectral encoder: {spectral_enc.max()}")
print(f"Mean weight spectral encoder: {spectral_enc.mean()}")
print(f"Std. dev. weights spectral encoder: {spectral_enc.std()}")
print(
    f"Initial weights of spectral encoder drawn uniformly from interval: ({-sqrt(1 / im.shape[-1])}, {sqrt(1 / im.shape[-1])})."
)


# %%
plt.matshow(spatial_enc)

# %% [markdown]
# #### Indian Pines

# %% [markdown]
# Range of weights when no perimeter is used:

# %%
im = read_image("/Users/jan/sciebo/data/segmentation/IndianPines/IndianPines.nc")
vade = VaDE(im.shape[-1], 10, 16, [500, 500, 2000], [2000, 500, 500], im.shape[:-1], 1, 0, ".", 0, True)
vade.load_model(
    "/Users/jan/sciebo/output/segmentation/vade/pure/IndianPines/2022-7-4-14-39-13-ep1500_ldim10_reg0_spat1/vade_params-ep1500_ldim10_reg0_spat1.pt"
)
spatial_enc = vade.spatial_enc[0].weight.detach().numpy()
spectral_enc = vade.spectral_enc[0].weight.detach().numpy()
print(f"Min weight spatial encoder: {spatial_enc.min()}")
print(f"Max weight spatial encoder: {spatial_enc.max()}")
print(f"Mean weight spatial encoder: {spatial_enc.mean()}")
print(f"Std. dev. weights spatial encoder: {spatial_enc.std()}")
print(f"Initial weights of spatial encoder drawn uniformly from interval: ({-sqrt(1 / 2)}, {sqrt(1 / 2)}).")
print(f"Min weight spectral encoder: {spectral_enc.min()}")
print(f"Max weight spectral encoder: {spectral_enc.max()}")
print(f"Mean weight spectral encoder: {spectral_enc.mean()}")
print(f"Std. dev. weights spectral encoder: {spectral_enc.std()}")
print(
    f"Initial weights of spectral encoder drawn uniformly from interval: ({-sqrt(1 / im.shape[-1])}, {sqrt(1 / im.shape[-1])})."
)

# %% [markdown]
# Range of weights when perimeter with $$\lambda = 0.01$$ is used:

# %%
im = read_image("/Users/jan/sciebo/data/segmentation/IndianPines/IndianPines.nc")
vade = VaDE(im.shape[-1], 10, 16, [500, 500, 2000], [2000, 500, 500], im.shape[:-1], 1, 0, ".", 0, True)
vade.load_model(
    "/Users/jan/sciebo/output/segmentation/vade/tv/IndianPines/2022-7-11-17-21-27-ep1000_ldim10_reg0_01_spat1/vade_params-ep1000_ldim10_reg0_01_spat1.pt"
)
spatial_enc = vade.spatial_enc[0].weight.detach().numpy()
spectral_enc = vade.spectral_enc[0].weight.detach().numpy()
print(f"Min weight spatial encoder: {spatial_enc.min()}")
print(f"Max weight spatial encoder: {spatial_enc.max()}")
print(f"Mean weight spatial encoder: {spatial_enc.mean()}")
print(f"Std. dev. weights spatial encoder: {spatial_enc.std()}")
print(f"Initial weights of spatial encoder drawn uniformly from interval: ({-sqrt(1 / 2)}, {sqrt(1 / 2)}).")
print(f"Min weight spectral encoder: {spectral_enc.min()}")
print(f"Max weight spectral encoder: {spectral_enc.max()}")
print(f"Mean weight spectral encoder: {spectral_enc.mean()}")
print(f"Std. dev. weights spectral encoder: {spectral_enc.std()}")
print(
    f"Initial weights of spectral encoder drawn uniformly from interval: ({-sqrt(1 / im.shape[-1])}, {sqrt(1 / im.shape[-1])})."
)

# %% [markdown]
# Range of weights when perimeter with $$\lambda = 0.05$$ is used:

# %%
im = read_image("/Users/jan/sciebo/data/segmentation/IndianPines/IndianPines.nc")
vade = VaDE(im.shape[-1], 10, 16, [500, 500, 2000], [2000, 500, 500], im.shape[:-1], 1, 0, ".", 0, True)
vade.load_model(
    "/Users/jan/sciebo/output/segmentation/vade/tv/IndianPines/2022-7-12-9-14-56-ep1000_ldim10_reg0_05_spat1/vade_params-ep1000_ldim10_reg0_05_spat1.pt"
)
spatial_enc = vade.spatial_enc[0].weight.detach().numpy()
spectral_enc = vade.spectral_enc[0].weight.detach().numpy()
print(f"Min weight spatial encoder: {spatial_enc.min()}")
print(f"Max weight spatial encoder: {spatial_enc.max()}")
print(f"Mean weight spatial encoder: {spatial_enc.mean()}")
print(f"Std. dev. weights spatial encoder: {spatial_enc.std()}")
print(f"Initial weights of spatial encoder drawn uniformly from interval: ({-sqrt(1 / 2)}, {sqrt(1 / 2)}).")
print(f"Min weight spectral encoder: {spectral_enc.min()}")
print(f"Max weight spectral encoder: {spectral_enc.max()}")
print(f"Mean weight spectral encoder: {spectral_enc.mean()}")
print(f"Std. dev. weights spectral encoder: {spectral_enc.std()}")
print(
    f"Initial weights of spectral encoder drawn uniformly from interval: ({-sqrt(1 / im.shape[-1])}, {sqrt(1 / im.shape[-1])})."
)

# %% [markdown]
# Range of weights when perimeter with $$\lambda = 0.1$$ is used:

# %%
im = read_image("/Users/jan/sciebo/data/segmentation/IndianPines/IndianPines.nc")
vade = VaDE(im.shape[-1], 10, 16, [500, 500, 2000], [2000, 500, 500], im.shape[:-1], 1, 0, ".", 0, True)
vade.load_model(
    "/Users/jan/sciebo/output/segmentation/vade/tv/IndianPines/2022-7-11-14-15-4-ep1000_ldim10_reg0_1_spat1/vade_params-ep1000_ldim10_reg0_1_spat1.pt"
)
spatial_enc = vade.spatial_enc[0].weight.detach().numpy()
spectral_enc = vade.spectral_enc[0].weight.detach().numpy()
print(f"Min weight spatial encoder: {spatial_enc.min()}")
print(f"Max weight spatial encoder: {spatial_enc.max()}")
print(f"Mean weight spatial encoder: {spatial_enc.mean()}")
print(f"Std. dev. weights spatial encoder: {spatial_enc.std()}")
print(f"Initial weights of spatial encoder drawn uniformly from interval: ({-sqrt(1 / 2)}, {sqrt(1 / 2)}).")
print(f"Min weight spectral encoder: {spectral_enc.min()}")
print(f"Max weight spectral encoder: {spectral_enc.max()}")
print(f"Mean weight spectral encoder: {spectral_enc.mean()}")
print(f"Std. dev. weights spectral encoder: {spectral_enc.std()}")
print(
    f"Initial weights of spectral encoder drawn uniformly from interval: ({-sqrt(1 / im.shape[-1])}, {sqrt(1 / im.shape[-1])})."
)

# %%
