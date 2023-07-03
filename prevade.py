import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn.functional import softmax
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from msiplib.io import saveArrayAsNetCDF
from msiplib.segmentation import create_segmentation_colormap

def build_network(layers, activation="relu", dropout=0):
    '''
        function builds a network.
        Args:
            layers: array specifying number of nodes in each layer
            activation: activation function after each dense layer
    '''
    net = []
    for i in range(1, len(layers)):

        #--------------------------------------------------------------------------------#
        # create dense layers
        #--------------------------------------------------------------------------------#
        net.append(nn.Linear(layers[i - 1], layers[i]))
        #--------------------------------------------------------------------------------#
        # Batch Normalization
        #--------------------------------------------------------------------------------#
        #net.append(nn.BatchNorm1d(layers[i]))
        #--------------------------------------------------------------------------------#
        # add activation function
        #--------------------------------------------------------------------------------#
        if activation == "relu":
            # original implementation uses ReLU
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())

        #--------------------------------------------------------------------------------#
        # possibly add dropout layer
        #--------------------------------------------------------------------------------#
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    #--------------------------------------------------------------------------------#
    # return network
    #--------------------------------------------------------------------------------#
    return nn.Sequential(*net)

class PreVade(nn.Module):
    def __init__(self, input_dim=200, z_dim=10, n_centroids=10,
                encodeLayer=[500, 500,2000], decodeLayer=[2000, 500, 500],
                im_shape=(145, 145)):
        super(PreVade, self).__init__()
        self.z_dim = z_dim
        self.n_centroids = n_centroids
        self.im_shape = im_shape
        self.input_dim = input_dim

        self.encoder = build_network([input_dim] + encodeLayer)
        self.decoder = build_network([z_dim] + decodeLayer)

        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mu = nn.Linear(decodeLayer[-1], input_dim) 

    def forward(self, x):
        z = self.encode(x)
        reconstructed_x = self.decode(z)
        return z, reconstructed_x

    def encode(self, x):
        h = self.encoder(x)
        mu = self._enc_mu(h)
        return mu
    
    def decode(self, z):
        h = self.decoder(z)
        mu_x = self._dec_mu(h)
        return mu_x

    def save_model(self, path):
        ''' save current model parameters on disk '''
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        print("pretrained_dict", pretrained_dict)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def visualize_latent_space(self, z, y, epoch):
        ''' visualize latent space using t-SNE '''
        tsne = TSNE(n_components=2, init='random')
        z_tsne = tsne.fit_transform(z)
        fig, ax = plt.subplots()
        cmap = create_segmentation_colormap()
        for t in range(self.n_centroids):
            points = z_tsne[y == t]
            ax.scatter(points[:, 0], points[:, 1], color=cmap[t], label=str(t), s=1)
        ax.legend()
        fig.savefig(f'{self.directory}/latent_epoch{epoch}_reg{self.lambdas}.png')