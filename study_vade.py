import numpy as np
import train_vade

# seed segmentation
hyperseed = 1337
rng = np.random.default_rng(hyperseed)
low = 0
high = 1e6
seeds = rng.integers(low=low, high=high, size=100)

for s in seeds:
    train_vade.main(s)                                      