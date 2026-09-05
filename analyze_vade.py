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
import os
import numpy as np
from vade import VaDE
from msiplib.io import read_image
import matplotlib.pyplot as plt

# %%
im = read_image(os.path.expandvars('${DATA_DIR}') + '/segmentation/SalinasA/SalinasA.nc')
vade = VaDE(204, 10, 6, [500,500,2000], [2000,500,500], im.shape[:-1], 1, 0, '.', 0, True)
vade.load_model(os.path.expandvars('${OUTPUT_DIR}') + '/segmentation/vade/pure/SalinasA/2022-7-1-17-40-58-ep1000_ldim10_reg0_spat1/vade_params-ep1000_ldim10_reg0_spat1.pt')
spatial_enc = vade.spatial_enc[0].weight.detach().numpy()
spectral_enc = vade.spectral_enc[0].weight.detach().numpy()
print(f'Min weight spatial encoder: {spatial_enc.min()}')
print(f'Max weight spatial encoder: {spatial_enc.max()}')
print(f'Min weight spectral encoder: {spectral_enc.min()}')
print(f'Max weight spectral encoder: {spectral_enc.max()}')

# %%
plt.matshow(spectral_enc)

# %%



