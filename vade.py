#pylint: disable=no_member

# TODO: variablen Bezeichnungen vereinheitlichen


from numpy.core.numeric import NaN
import torch
from datetime import datetime
from torch.autograd.function import Function
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils import data
from torchvision.utils import save_image
from sklearn.manifold import TSNE

import numpy as np
import math
from sklearn.mixture import GaussianMixture

def cluster_acc(Y_pred, Y, ignore_label=None):
    if ignore_label is not None:
        index = Y!= ignore_label
        Y=Y[index]
        Y_pred=Y_pred[index]
    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return (w[ind[0], ind[1]]).sum() / Y_pred.size, w

def buildNetwork(layers, activation="relu", dropout=0):
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
            net.append(nn.ReLU())
            #net.append(nn.LeakyReLU())
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

def adjust_learning_rate(init_lr, optimizer, epoch):
    #--------------------------------------------------------------------------------#
    # function implements a learning rate scheduler
    #--------------------------------------------------------------------------------#
    lr = max(init_lr * (0.9 ** (epoch // 10)), 0.0002)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

log2pi = math.log(2 * math.pi)
def log_likelihood_samples_unit_gaussian(samples):
    return -0.5 * log2pi * samples.size()[1] - torch.sum(0.5 * (samples) ** 2, 1)

def log_likelihood_samplesImean_sigma(samples, mu, logvar):
    #--------------------------------------------------------------------------------#
    # computes the log of the joint probability of samples wrt to mean mu and variance (log of variance) logvar
    #--------------------------------------------------------------------------------#
    return -0.5 * log2pi * samples.size()[1] \
           - torch.sum(0.5 * (samples - mu) ** 2 / torch.exp(logvar) + 0.5 * logvar, 1)


class VaDE(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_centroids=10,
        encodeLayer=[500, 500,2000], decodeLayer=[2000, 500, 500],
        one_by_delta_x=86,regularization=100,
        directory='Without_batch_normal',ignore_label=None):
        super(VaDE, self).__init__()
        self.z_dim = z_dim
        self.n_centroids = n_centroids
        self.one_by_delta_x=one_by_delta_x
        self.directory=directory

        #--------------------------------------------------------------------------------#
        # initialize encoder and decoder network
        #--------------------------------------------------------------------------------#
        self.encoder = buildNetwork([input_dim] + encodeLayer)
        self.decoder = buildNetwork([z_dim] + decodeLayer)

        #--------------------------------------------------------------------------------#
        # add output layers of encoder
        #--------------------------------------------------------------------------------#
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._enc_log_sigma = nn.Linear(encodeLayer[-1], z_dim)

        #--------------------------------------------------------------------------------#
        # add output layers of decoder
        #--------------------------------------------------------------------------------#
        self._dec_mu = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_log_sigma = nn.Linear(decodeLayer[-1], input_dim)

        #--------------------------------------------------------------------------------#
        # add initial parameters for GMM
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # pi in paper: p(c), the prior over the Gaussians in the mixture model
        #--------------------------------------------------------------------------------#
        self.prior_init = nn.Parameter(torch.ones(n_centroids) / n_centroids)

        #--------------------------------------------------------------------------------#
        # mu_c in the paper, the initial guesses for the centroids of the mixtures of the Gaussians
        #--------------------------------------------------------------------------------#
        self.mean_init = nn.Parameter(torch.zeros(z_dim, n_centroids))

        #--------------------------------------------------------------------------------#
        # log_sigma_c^2 in the paper, the initial guesses for the variances of the mixtures of Gaussians
        #--------------------------------------------------------------------------------#
        self.log_var_init = nn.Parameter(torch.zeros(z_dim, n_centroids))

        #--------------------------------------------------------------------------------#
        #object for softmax
        #--------------------------------------------------------------------------------#
        self.soft_max=nn.Softmax(dim=0)

        #--------------------------------------------------------------------------------#
        #initialize regularizer parameter
        #--------------------------------------------------------------------------------#
        self.lambdas= regularization

        #--------------------------------------------------------------------------------#
        # Ignore label variable
        #--------------------------------------------------------------------------------#
        self.ignore_label = ignore_label
    
        
    def initialize_gmm(self, dataloader):
        #--------------------------------------------------------------------------------#
        # iniitialize GMM by applying the encoder and computing the z's that can then be clustered with the GMM
        #--------------------------------------------------------------------------------#
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        #--------------------------------------------------------------------------------#
        # put model into evaluation mode
        #--------------------------------------------------------------------------------#
        self.eval()
        data = []
        data_right =[]
        data_bottom =[]

        #--------------------------------------------------------------------------------#
        # take samples and map them to the latent space to fit the GMM there
        #--------------------------------------------------------------------------------#

        for _, (inputs, _) in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            data_dim = int(inputs.shape[1]/3)
            z, _, _, _, _ = self.forward(inputs[:,0:data_dim])

            data.append(z.data.cpu().numpy())

        data = np.concatenate(data)

        #--------------------------------------------------------------------------------#
        # fit the GMM
        #--------------------------------------------------------------------------------#

        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        gmm.fit(data)

        #--------------------------------------------------------------------------------#
        # copy learned data to parameters of the network
        #--------------------------------------------------------------------------------#

        self.prior_init.data.copy_(torch.from_numpy(gmm.weights_.T.astype(np.float32)))
        self.mean_init.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.log_var_init.data.copy_(torch.from_numpy(np.log(gmm.covariances_.T.astype(np.float32))))

    def reparameterize(self, mu, logvar):
        # TODO: Check for correctness. does it work with batches of data?
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        #--------------------------------------------------------------------------------#
        # apply decoder to a sample from the latent space
        #--------------------------------------------------------------------------------#
        h = self.decoder(z)
        mu_x = self._dec_mu(h)
        log_sigma_x = self._dec_log_sigma(h)
        return mu_x, log_sigma_x

    
    def get_gamma(self, z):
        Z = z.unsqueeze(2)

        p_c_z = torch.exp(torch.log(self.prior_init) - torch.sum(0.5*torch.log(2*math.pi*torch.exp(self.log_var_init))+\
            (Z-self.mean_init)**2/(2*torch.exp(self.log_var_init)), dim=1)) + 1e-10 # NxK

        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)
        return gamma
    
    def loss_function(self, inputs, mu_x, logvar_x, z, z_mean, z_log_var,z_right,z_bottom):
        Z=z.unsqueeze(2)

        Z_right=z_right.unsqueeze(2)
        Z_bottom=z_bottom.unsqueeze(2)

        z_mean_t=z_mean.unsqueeze(2)
        z_log_var_t=z_log_var.unsqueeze(2)
        gamma= self.get_gamma(z)

        gamma_right= self.get_gamma(z_right)
        gamma_bottom= self.get_gamma(z_bottom)
        x=inputs[:,0:int(inputs.shape[1]/3)]

        self.prior_init.data.copy_(self.soft_max(self.prior_init))
        #--------------------------------------------------------------------------------#
        # 1st term in equation 9 // also the 1st line in equation 12
        #--------------------------------------------------------------------------------#
        log_pxz = 0.5 * torch.sum(math.log(2 * math.pi) + logvar_x + (x - mu_x)**2 / torch.exp(logvar_x), dim=1)

        #--------------------------------------------------------------------------------#
        # 2nd term in equation 9  // also the 2nd line in equation 12
        #--------------------------------------------------------------------------------#

        logpzc = torch.sum(0.5 * gamma * torch.sum(self.log_var_init + \
            torch.exp(z_log_var_t) /torch.exp(self.log_var_init) + (z_mean_t - self.mean_init)**2 / torch.exp(self.log_var_init), dim=1), dim=1)

        #--------------------------------------------------------------------------------#
        # remaining 3 terms in equation is summed up in last line of equation 12
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # 3rd term in equation 9
        #--------------------------------------------------------------------------------#
        logpc = -torch.sum(torch.log(self.prior_init) * gamma, 1)

        #--------------------------------------------------------------------------------#
        # 4th term in equation 9
        #--------------------------------------------------------------------------------#
        qentropy = -0.5 * torch.sum(1 + z_log_var_t, 1)

        #--------------------------------------------------------------------------------#
        # 5th term in equation 9
        #--------------------------------------------------------------------------------#
        logqcx = torch.sum(torch.log(torch.clamp(gamma, min=1e-10)) * gamma, 1)

    
        #--------------------------------------------------------------------------------#
        # Regularize with gamma=q(c|x)
        #--------------------------------------------------------------------------------#
        delta_qcx_right = self.one_by_delta_x *(gamma_right -gamma)
        delta_qcx_bottom =self.one_by_delta_x *(gamma_bottom -gamma)
        
        square_sum_qcx = (delta_qcx_right)**2 + (delta_qcx_bottom)**2
        
        norm_qcx = (square_sum_qcx+0.01) ** (0.5)
        
        #--------------------------------------------------------------------------------#
        # final loss function 
        #--------------------------------------------------------------------------------#
        loss = torch.mean(log_pxz + logpzc + logpc + qentropy + logqcx) + self.lambdas * torch.mean(torch.sum(norm_qcx,1))
        
        return loss

    def forward(self, x):
        #--------------------------------------------------------------------------------#
        # compute a forward pass
        #--------------------------------------------------------------------------------#

        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        mu_x, logvar_x = self.decode(z)
        return z, mu, logvar, mu_x, logvar_x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        #--------------------------------------------------------------------------------#
        # load weights into memory
        #--------------------------------------------------------------------------------#
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)

        #--------------------------------------------------------------------------------#
        # get a state dictionary of the whole model, in particular, the parameters
        #--------------------------------------------------------------------------------#
        model_dict = self.state_dict()

        #--------------------------------------------------------------------------------#
        # extract all weights available in the loaded ones and also in the module itself
        #--------------------------------------------------------------------------------#
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        #--------------------------------------------------------------------------------#
        # copy weights to model
        #--------------------------------------------------------------------------------#
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def visualize_state(self, z, y, epoch):
        tsne = TSNE(n_components=2, init='pca')
        z_tsne = tsne.fit_transform(z)
        fig, ax = plt.subplots()
        cmap = plt.get_cmap("tab20")
        for t in range(self.n_centroids):
            points = z_tsne[y == t]
            ax.scatter(points[:, 0], points[:, 1], color=cmap(t), label=str(t),s=1)
        ax.legend()
        fig.savefig('{}/latent_epoch_{}_reg_{}.png'.format(self.directory,epoch,self.lambdas))
    def predict(self, dataloader):
        #--------------------------------------------------------------------------------#
        # apply model with current weights to a dataset
        #--------------------------------------------------------------------------------#

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        self.eval()
        valid_loss = 0.0
        Y_pred = []
        Y = []
        Z = []
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, mu, logvar, mu_x, logvar_x = self.forward(inputs)

            Z.append(z.detach().cpu().numpy())

            loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar)
            valid_loss += loss.data.item() * len(inputs)
            
            #--------------------------------------------------------------------------------#
            # compute gamma for whole evaluation batch
            #--------------------------------------------------------------------------------#
            gamma = self.get_gamma(z).data.cpu().numpy()
            Y.append(labels.numpy())

            #--------------------------------------------------------------------------------#
            # compute predicted labels by taking the argmax over gamma (q(c | z))
            #--------------------------------------------------------------------------------#
            Y_pred.append(np.argmax(gamma, axis=1))

        Y_pred = np.concatenate(Y_pred)
        Y = np.concatenate(Y)
        Z = np.concatenate(Z)
        acc = cluster_acc(Y_pred, Y,self.ignore_label)
        print('Accuracy: {}'.format(acc[0]))

        return Z, Y_pred


    def fit(self, trainloader, validloader, lr=0.002, num_epochs=10,
        visualize=True, anneal=False):

        #--------------------------------------------------------------------------------#
        # train the model
        #--------------------------------------------------------------------------------#
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        #--------------------------------------------------------------------------------#
        # initial optimizer. the filter function only passes the parameters to Adam that need gradient
        #--------------------------------------------------------------------------------#
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        #--------------------------------------------------------------------------------#
        # validate
        #--------------------------------------------------------------------------------#
        self.eval()
        valid_loss = 0.0
        Z = []
        Y = []
        for _, (inputs, labels) in enumerate(validloader):
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)

            data_dim = int(inputs.shape[1]/3)
            z, mu, logvar, mu_x, logvar_x = self.forward(inputs[:,0:data_dim])

            z_right, _, _, _,_ = self.forward(inputs[:,data_dim:2*data_dim])
            z_bottom, _, _, _,_= self.forward(inputs[:,2*data_dim:3*data_dim])

            Z.append(z.detach().cpu().numpy())
            Y.append(labels.numpy())
            
            loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar,z_right,z_bottom)
            
            valid_loss += loss.data.item() * len(inputs)
        
        Y = np.concatenate(Y)
        Z = np.concatenate(Z)
        self.visualize_state(Z, Y, epoch=-1)

        #--------------------------------------------------------------------------------#
        # Valid loss of initialized weights
        #--------------------------------------------------------------------------------#
        print("#Epoch -1: Valid Loss: %.5f" % (valid_loss / len(validloader.dataset)))
        
        #--------------------------------------------------------------------------------#
        # train num_epochs
        #--------------------------------------------------------------------------------#
    
        for epoch in range(num_epochs):
            self.train()
            #--------------------------------------------------------------------------------#
            # ensures that learning rate does not become smaller than 0.0002
            #--------------------------------------------------------------------------------#
            if anneal:
                epoch_lr = adjust_learning_rate(lr, optimizer, epoch)
            else:
                epoch_lr = optimizer.param_groups[0]['lr']

            #--------------------------------------------------------------------------------#
            # compute a gradient step for each training batch
            #--------------------------------------------------------------------------------#
            train_loss = 0
            a=1
            for _, (inputs, _) in enumerate(trainloader):
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                
                data_dim =int(inputs.shape[1]/3)
                z, mu, logvar, mu_x, logvar_x = self.forward(inputs[:,0:data_dim])
                z_right, _, _, _,_ = self.forward(inputs[:,data_dim:2*data_dim])
                z_bottom, _, _, _,_= self.forward(inputs[:,2*data_dim:3*data_dim])
                
                

                loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar,z_right,z_bottom)

                train_loss += loss.data.item() * len(inputs)
                loss.backward()
                optimizer.step()

            #--------------------------------------------------------------------------------#
            # validate current weights on validation set
            #--------------------------------------------------------------------------------#
            self.eval()
            valid_loss = 0.0
            Y = []
            Y_pred = []
            Z = []
            for batch_idx, (inputs, labels) in enumerate(validloader):
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)

                data_dim = int(inputs.shape[1]/3)
                z, mu, logvar, mu_x, logvar_x = self.forward(inputs[:,0:data_dim])
                z_right, _, _, _,_ = self.forward(inputs[:,data_dim:2*data_dim])
                z_bottom, _, _, _,_= self.forward(inputs[:,2*data_dim:3*data_dim])

                Z.append(z.detach().cpu().numpy())


                loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar,z_right,z_bottom)

                valid_loss += loss.data.item() * len(inputs)

                #--------------------------------------------------------------------------------#
                # compute gamma for whole evaluation batch
                #--------------------------------------------------------------------------------#
                gamma = self.get_gamma(z).data.cpu().numpy()
                Y.append(labels.numpy())
                
                #--------------------------------------------------------------------------------#
                # compute predicted labels by taking the argmax over gamma (q(c | z))
                #--------------------------------------------------------------------------------#
                Y_pred.append(np.argmax(gamma, axis=1))

            Y = np.concatenate(Y)
            Y_pred = np.concatenate(Y_pred)
            Z = np.concatenate(Z)
            acc = cluster_acc(Y_pred, Y,self.ignore_label)

            print("#Epoch %3d: lr: %.5f, Train Loss: %.5f, Valid Loss: %.5f, acc: %.5f" % (
                epoch, epoch_lr, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset), acc[0]))

            #if visualize and epoch % 50 == 0:
            if visualize and epoch == (num_epochs-1):
                print('Plot current latent space.')
                self.visualize_state(Z, Y_pred, epoch=epoch)

        return Y_pred

    def log_marginal_likelihood_estimate(self, x, num_samples):
        weight = torch.zeros(x.size(0))
        for _ in range(num_samples):
            z, mu, logvar, mu_x, _ = self.forward(x)
            zloglikelihood = log_likelihood_samples_unit_gaussian(z)
            dataloglikelihood = torch.sum(x*torch.log(torch.clamp(mu_x, min=1e-10))+
                (1-x)*torch.log(torch.clamp(1-mu_x, min=1e-10)), 1)
            log_qz = log_likelihood_samplesImean_sigma(z, mu, logvar)
            weight += torch.exp(dataloglikelihood + zloglikelihood - log_qz).data
        return torch.log(torch.clamp(weight/num_samples, min=1e-40))
