#pylint: disable=no-member
''' module defines the VaDE class and implements all necessary functions to use the model '''

import math

import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn.functional import softmax
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from msiplib.io import saveArrayAsNetCDF
from msiplib.segmentation import create_segmentation_colormap

from tsnecuda import TSNE as TSNE_GPU

def cluster_acc(y_pred, y, ignore_label=None):
    ''' compute overall accuracy of clustering '''
    if ignore_label is not None:
        index = y != ignore_label
        y = y[index]
        y_pred = y_pred[index]
    assert y_pred.size == y.size
    d = max(y_pred.max(), y.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return (w[ind[0], ind[1]]).sum() / y_pred.size, w

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

def adjust_learning_rate(init_lr, optimizer, epoch):
    ''' learning rate scheduler '''
    lr = max(init_lr * (0.9 ** (epoch // 10)), 0.0002)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

log2pi = math.log(2 * math.pi)
def log_likelihood_samples_unit_gaussian(samples):
    #TODO: derive formula and compare
    return -0.5 * log2pi * samples.size()[1] - torch.sum(0.5 * (samples)**2, 1)

def log_likelihood_samplesImean_sigma(samples, mu, logvar):
    #TODO: derive formula and compare
    #--------------------------------------------------------------------------------#
    # computes the log of the joint probability of samples wrt to mean mu and variance (log of variance) logvar
    #--------------------------------------------------------------------------------#
    return -0.5 * log2pi * samples.size()[1] \
           - torch.sum(0.5 * (samples - mu) ** 2 / torch.exp(logvar) + 0.5 * logvar, 1)

class VaDE(nn.Module):
    ''' class implements the VaDE model '''
    def __init__(self, input_dim=200, z_dim=10, n_centroids=10,
        encodeLayer=[500, 500,2000], decodeLayer=[2000, 500, 500],
        im_shape=(145, 145), one_by_delta_x=100, regularization=0.01,
        directory='.', ignore_label=None, spatial=False, L = 10):
        super(VaDE, self).__init__()
        self.z_dim = z_dim
        self.n_centroids = n_centroids
        self.directory = directory
        self.im_shape = im_shape
        self.one_by_delta_x = one_by_delta_x
        self.use_cuda = torch.cuda.is_available()
        self.input_dim = input_dim
        self.spatial = spatial
        self.L = L

        #--------------------------------------------------------------------------------#
        # initialize encoder and decoder network
        #--------------------------------------------------------------------------------#
        if spatial:
            # if spatial coordinates are concatenated, encode them using an extra linear layer
            self.spectral_enc = nn.Sequential(*[nn.Linear(input_dim, encodeLayer[0]), nn.ReLU()])
            self.spatial_enc = nn.Sequential(*[nn.Linear(2, encodeLayer[0]), nn.ReLU()])
            self.encoder = build_network(encodeLayer)
        else:
            self.encoder = build_network([input_dim] + encodeLayer)

        self.decoder = build_network([z_dim] + decodeLayer)

        #--------------------------------------------------------------------------------#
        # add output layers of encoder
        #--------------------------------------------------------------------------------#
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._enc_log_sigma = nn.Linear(encodeLayer[-1], z_dim)

        #--------------------------------------------------------------------------------#
        # add output layers of decoder
        #--------------------------------------------------------------------------------#
        # original implementation applies sigmoid activation for MNIST, no activation in all other cases
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
        #initialize regularizer parameter
        #--------------------------------------------------------------------------------#
        self.lambdas = regularization

        #--------------------------------------------------------------------------------#
        # Ignore label variable
        #--------------------------------------------------------------------------------#
        self.ignore_label = ignore_label


    def initialize_gmm(self, dataloader):
        '''
            after initialization of the network weights, map the samples to the latent space and fit a GMM
            to initialize the parameters of the GMM that are later tuned during training
        '''
        #--------------------------------------------------------------------------------#
        # iniitialize GMM by applying the encoder and computing the z's that can then be clustered with the GMM
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # put model into evaluation mode
        #--------------------------------------------------------------------------------#
        self.eval()
        data = []

        #--------------------------------------------------------------------------------#
        # take samples and map them to the latent space to fit the GMM there
        #--------------------------------------------------------------------------------#

        for _, (inputs, _) in enumerate(dataloader):
            inputs = Variable(inputs)
            if self.lambdas != 0:
                data_dim = int(inputs.shape[1] / 3)
                z, _, _, _, _ = self.forward(inputs[:, 0:data_dim])
            else:
                z, _, _, _, _ = self.forward(inputs)


            # GMM fitted on cpu, therefore moving data to cpu is okay.
            data.append(z.mean(dim=0).data.cpu().numpy())

        data = np.concatenate(data)

        #--------------------------------------------------------------------------------#
        # fit the GMM
        #--------------------------------------------------------------------------------#

        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        gmm.fit(data)

        #--------------------------------------------------------------------------------#
        # copy learned data to parameters of the network
        #--------------------------------------------------------------------------------#
        with torch.no_grad():
            self.prior_init.copy_(torch.from_numpy(gmm.weights_.T.astype(np.float32)))
            self.mean_init.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
            self.log_var_init.copy_(torch.from_numpy(np.log(gmm.covariances_.T.astype(np.float32))))

    def reparameterize(self, mu, logvar):
        ''' sample an auxiliary variable and reparameterize it '''
        # TODO: Check for correctness. does it work with batches of data?
        if self.training:
            std = logvar.mul(0.5).exp_() # [100, 10]
            eps = Variable(std.data.new(self.L, *std.size()).normal_()) # [L, 100, 10]
            ## Ex: L = 2, batch_size = 1, latent_dim = 3
            ## [[1, 2, 3]] std [1, 3]
            ## [[4, 5, 6]] mu [1, 3]
            ## [[1, 2, 3], [2, 4, 6]] eps [2, 3]
            ## [[1, 4, 9], [2, 8, 18]] eps*std [2, 3] + [[4, 5, 6], [4, 5, 6]] mu [2, 3]  = [[5, 9, 15], [8, 15, 24]]
            eps_l = eps.mul(std).add_(mu) # [L, 100, 10]
            return eps_l
        else:
            return mu.unsqueeze(0).expand(self.L, -1, -1)

    def decode(self, z):
        ''' decode a latent vector back to the original space '''
        #--------------------------------------------------------------------------------#
        # apply decoder to a sample from the latent space
        #--------------------------------------------------------------------------------#
        h = self.decoder(z)
        mu_x = self._dec_mu(h)
        log_sigma_x = self._dec_log_sigma(h)
        return mu_x, log_sigma_x


    def get_gamma(self, z):
        ''' compute p(c | z) as approximation to q(c | x) '''
        z_t = z.unsqueeze(3)

        p_c_z = torch.exp(torch.log(softmax(self.prior_init, dim=0)) \
            - torch.sum(0.5 * torch.log(2 * math.pi * torch.exp(self.log_var_init)) \
            + (z_t - self.mean_init)**2 / (2 * torch.exp(self.log_var_init)), dim=2)) + 1e-10 # NxK

        gamma_l = p_c_z / torch.sum(p_c_z, dim=2, keepdim=True)
        gamma_num = torch.exp(1- 1/self.L*torch.sum(torch.log(gamma_l), dim=0))
        gamma_denom = torch.sum(1/gamma_num, dim=1)
        gamma_denom = gamma_denom.unsqueeze(1)
        gamma = 1/(gamma_num*gamma_denom)
        
        return gamma 

    def loss_function(self, inputs, mu_x, logvar_x, z, z_mean, z_log_var, z_right=None, z_bottom=None):
        ''' Function computes the loss of the network '''
        z_mean_t = z_mean.unsqueeze(2)
        z_log_var_t = z_log_var.unsqueeze(2)
        gamma = self.get_gamma(z)

        if self.lambdas != 0:
            gamma_right = self.get_gamma(z_right)
            gamma_bottom = self.get_gamma(z_bottom)
            x = inputs[:, 0:int(inputs.shape[1] / 3)]
        else:
            x = inputs

        #-------------------------------------------------------------------------------#
        # 1st term in equation 9 // also the 1st line in equation 12
        #--------------------------------------------------------------------------------#
        if self.spatial:
            # if spatial coordinate is encoded, only reconstruct spectrum
            log_pxz = 0.5 * torch.sum(math.log(2 * math.pi) + 1/self.L * torch.sum(logvar_x, dim=0) + \
                    1/self.L*torch.sum((x[:, :-2] - mu_x)**2 / torch.exp(logvar_x), dim=0), dim=1)
            
        else:
            ## mu_x, logvar_x [100, 10] -> L=5 [100, 5, 10]
            ## sum(mu_x), sum(logvar_x) [100, 10]
            log_pxz = 0.5 * torch.sum(math.log(2 * math.pi) + 1/self.L * torch.sum(logvar_x, dim=0) + 1/self.L*torch.sum((x - mu_x)**2 / torch.exp(logvar_x), dim=0), dim=1)

        import pdb
        pdb.set_trace()
        #--------------------------------------------------------------------------------#
        # 2nd term in equation 9  // also the 2nd line in equation 12
        #--------------------------------------------------------------------------------#

        logpzc = torch.sum(0.5 * gamma * torch.sum(math.log(2 * math.pi) + self.log_var_init \
            + torch.exp(z_log_var_t) /torch.exp(self.log_var_init) \
            + (z_mean_t - self.mean_init)**2 / torch.exp(self.log_var_init), dim=1), dim=1)
        
        #--------------------------------------------------------------------------------#
        # remaining 3 terms in equation is summed up in last line of equation 12
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # 3rd term in equation 9
        #--------------------------------------------------------------------------------#
        logpc = -torch.sum(torch.log(softmax(self.prior_init, dim=0)) * gamma, 1)

        #--------------------------------------------------------------------------------#
        # 4th term in equation 9
        #--------------------------------------------------------------------------------#
        qentropy = 0.5 * torch.sum(1 + z_log_var_t, 1)
        qentropy = torch.squeeze(qentropy)

        #--------------------------------------------------------------------------------#
        # 5th term in equation 9
        #--------------------------------------------------------------------------------#
        logqcx = torch.sum(torch.log(torch.clamp(gamma, min=1e-10)) * gamma, 1)

        #--------------------------------------------------------------------------------#
        # Regularize with gamma=q(c|x) -> q(c|g(x)) where g(x) is our spectra
        #--------------------------------------------------------------------------------#
        if self.lambdas != 0:
            delta_qcx_right = self.one_by_delta_x * (gamma_right - gamma)
            delta_qcx_bottom = self.one_by_delta_x * (gamma_bottom - gamma)

            square_sum_qcx = (delta_qcx_right)**2 + (delta_qcx_bottom)**2

            norm_qcx = (square_sum_qcx + 0.01)**(0.5)

            #--------------------------------------------------------------------------------#
            # final loss function
            #--------------------------------------------------------------------------------#
            loss = torch.mean(log_pxz + logpzc + logpc + qentropy + logqcx) \
                + self.lambdas * torch.mean(torch.sum(norm_qcx, 1))
        else:
            loss = torch.mean(log_pxz + logpzc + logpc + qentropy + logqcx)

        return loss

    def forward(self, x):
        ''' compute forward pass through network '''
        if self.spatial:
            # if spatial coordinate is added, encode it in an extra feature vector
            h_spectral = self.spectral_enc(x[:, :-2])
            h_spatial = self.spatial_enc(x[:, -2:])
            h = self.encoder(h_spectral + h_spatial)
        else:
            h = self.encoder(x)

        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        mu_x, logvar_x = self.decode(z)
        return z, mu, logvar, mu_x, logvar_x

    def save_model(self, path):
        ''' save current model parameters on disk '''
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        ''' load pretrained weights into model '''
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
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} \
            # pylint: disable=unsupported-membership-test

        #--------------------------------------------------------------------------------#
        # copy weights to model
        #--------------------------------------------------------------------------------#
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)

    def visualize_latent_space_cpu(self, z, y, epoch):
        ''' visualize latent space using t-SNE in CPU'''
        tsne = TSNE(n_components=2, init='random')
        z_tsne = tsne.fit_transform(z)
        fig, ax = plt.subplots()
        cmap = create_segmentation_colormap()
        for t in range(self.n_centroids):
            points = z_tsne[y == t]
            ax.scatter(points[:, 0], points[:, 1], color=cmap[t], label=str(t), s=1)
        ax.legend()
        fig.savefig(f'{self.directory}/latent_epoch{epoch}_reg{self.lambdas}.png')

    def visualize_latent_space(self, z, y, epoch):
        ''' visualize latent space using t-SNE in GPU'''
        tsne = TSNE_GPU(n_components=2)
        z_tsne = tsne.fit_transform(z)
        fig, ax = plt.subplots()
        cmap = create_segmentation_colormap()
        for t in range(self.n_centroids):
            points = z_tsne[y == t]
            ax.scatter(points[:, 0], points[:, 1], color=cmap[t], label=str(t), s=1)
        ax.legend()
        fig.savefig(f'{self.directory}/latent_epoch{epoch}_reg{self.lambdas}.png')

    def predict(self, dataloader):
        ''' apply model with current weights to a dataset '''
        self.eval()
        valid_loss = 0.0
        y_pred = []
        y = []
        z_all = []
        gamma_all = []
        for _, (inputs, labels) in enumerate(dataloader):
            inputs = Variable(inputs)
            if self.lambdas != 0:
                data_dim = int(inputs.shape[1] / 3)
                z, mu, logvar, mu_x, logvar_x = self.forward(inputs[:, 0:data_dim])
                z_right, _, _, _, _ = self.forward(inputs[:, data_dim:2 * data_dim])
                z_bottom, _, _, _, _ = self.forward(inputs[:, 2 * data_dim:3 * data_dim])
                loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar, z_right, z_bottom)
            else:
                z, mu, logvar, mu_x, logvar_x = self.forward(inputs)
                loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar)

            z_all.append(z.detach().cpu().numpy())
            valid_loss += loss.data.item() * len(inputs)

            #--------------------------------------------------------------------------------#
            # compute gamma for whole evaluation batch
            #--------------------------------------------------------------------------------#
            gamma = self.get_gamma(z).detach()
            gamma_all.append(gamma.cpu().numpy())

            #--------------------------------------------------------------------------------#
            # compute predicted labels by taking the argmax over gamma (q(c | z))
            #--------------------------------------------------------------------------------#
            y.append(labels.cpu().numpy())
            y_pred.append(torch.argmax(gamma, dim=1).cpu().numpy())
        
        y_pred = np.concatenate(y_pred)
        y = np.concatenate(y)
        z_all = np.concatenate(z_all)
        gamma_all = np.concatenate(gamma_all)

        acc = cluster_acc(y_pred, y, self.ignore_label)
        print(f'Overall accuracy: {acc[0]:.5f}')

        return valid_loss, z_all, y_pred, gamma_all


    def fit(self, trainloader, validloader, lr=0.002, num_epochs=10,
        visualize=True, anneal=False, model=None, pretrain_model=None):
        ''' train model '''
        #--------------------------------------------------------------------------------#
        # initial optimizer. the filter function only passes the parameters to Adam that need gradient
        #--------------------------------------------------------------------------------#
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        #----------------------------------------       ----------------------------------------#
        # validate
        #--------------------------------------------------------------------------------#
        self.eval()
        valid_loss = 0.0
        z_all = []
        y = []
        denoised_input = torch.empty(0).to('cuda')
        
        for _, (inputs, labels) in enumerate(validloader):
            inputs = Variable(inputs)

            ## Replace current inputs with the denoise input
            if pretrain_model is not None:
                _, denoised_x = pretrain_model.forward(inputs)
                denoised_input = torch.cat((denoised_input, denoised_x), dim=0)
                # We need to add the spatial information back into the denoised_x
                last_two_columns = inputs[:, -2:]
                inputs = torch.cat((denoised_x, last_two_columns), dim=1)

            if self.lambdas != 0:
                data_dim = int(inputs.shape[1] / 3)
                z, mu, logvar, mu_x, logvar_x = self.forward(inputs[:, 0:data_dim])
                z_right, _, _, _, _ = self.forward(inputs[:, data_dim:2 * data_dim])
                z_bottom, _, _, _, _ = self.forward(inputs[:, 2 * data_dim:3 * data_dim])
                loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar, z_right, z_bottom)
            else:
                z, mu, logvar, mu_x, logvar_x = self.forward(inputs)
                loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar)

            z_all.append(z.detach().cpu().numpy())
            y.append(labels.cpu().numpy())

            valid_loss += loss.data.item() * len(inputs)

        # reshaped_denoised_input = denoised_input.view(145,145,200).to('cpu').detach().numpy()
        # saveArrayAsNetCDF(reshaped_denoised_input, f'{self.directory}/denoised_image.nc')

        if visualize:
            y = np.concatenate(y)
            z_all = np.concatenate(z_all, axis=1)
            self.visualize_latent_space(z_all, y, epoch=-1)

        #--------------------------------------------------------------------------------#
        # Valid loss of initialized weights computed over the whole dataset
        #--------------------------------------------------------------------------------#
        print(f'#Epoch -1: Valid Loss: {(valid_loss / len(validloader.dataset)):.5f}')

        #--------------------------------------------------------------------------------#
        # train num_epochs
        #--------------------------------------------------------------------------------#
        grad_norm_list = []
        grad_angle_list = []
        grad_angle_list_first = []
        previous_grad = None ## initialize dummy previous gradient to avoid error
        first_grad = []

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
            for idx, (inputs, _) in enumerate(trainloader):
                optimizer.zero_grad()
                inputs = Variable(inputs)

                ## Replace current inputs with the denoise input
                if pretrain_model is not None:
                    _, denoised_x = pretrain_model.forward(inputs)
                    last_two_columns = inputs[:, -2:]
                    inputs = torch.cat((denoised_x, last_two_columns), dim=1)
                    
                if self.lambdas != 0:
                    data_dim =int(inputs.shape[1] / 3)
                    z, mu, logvar, mu_x, logvar_x = self.forward(inputs[:, 0:data_dim])
                    z_right, _, _, _, _ = self.forward(inputs[:, data_dim:2 * data_dim])
                    z_bottom, _, _, _, _ = self.forward(inputs[:, 2 * data_dim:3 * data_dim])
                    loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar, z_right, z_bottom)
                else:
                    z, mu, logvar, mu_x, logvar_x = self.forward(inputs)
                    loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar)

                train_loss += loss.data.item() * len(inputs)
                loss.backward()
                optimizer.step()

                ############ LOG GRADIENT NORM AND ANGLES ############
                grad_norm = 0.0
                current_grad = []
                for param in self.parameters():
                    grad_norm += torch.norm(param.grad.data) ** 2
                    if param.grad is not None:
                        current_grad.append(param.grad.view(-1))
                current_grad = torch.cat(current_grad)
                if idx == 0:
                    first_grad = current_grad
                grad_norm = grad_norm ** 0.5
                ## calculate cur-to-prev gradient comparison
                if previous_grad is not None:
                    cosine_similarity = torch.dot(current_grad, previous_grad) / (torch.norm(current_grad) * torch.norm(previous_grad))
                    angle = np.arccos(cosine_similarity.cpu()) * 180 / np.pi
                    grad_angle_list.append(angle.item())
                previous_grad = current_grad.clone()
                ## calculate cur-to-first gradient comparison
                cosine_similarity = torch.dot(current_grad, first_grad) / (torch.norm(current_grad) * torch.norm(first_grad))
                angle = np.arccos(cosine_similarity.cpu()) * 180 / np.pi
                grad_angle_list_first.append(angle.item())
                grad_norm_list.append(grad_norm.item())
                #######################################################
            with open(self.directory+'/gradient_info.log', 'w') as file:
                file.write("Gradient Norm: " + str(grad_norm_list) + "\n")
                file.write("Gradient Angle (prev): " + str(grad_angle_list) + "\n")
                file.write("Gradient Angle (first): " + str(grad_angle_list_first) + "\n")

            #--------------------------------------------------------------------------------#
            # validate current weights on validation set
            #--------------------------------------------------------------------------------#
            self.eval()
            valid_loss = 0.0
            y = []
            y_pred = []
            z_all = []
            gamma_all = []
            for _, (inputs, labels) in enumerate(validloader):
                inputs = Variable(inputs)

                ## Replace current inputs with the denoise input
                if pretrain_model is not None:
                    _, denoised_x = pretrain_model.forward(inputs)
                    last_two_columns = inputs[:, -2:]
                    inputs = torch.cat((denoised_x, last_two_columns), dim=1)

                if self.lambdas != 0:
                    data_dim = int(inputs.shape[1] / 3)
                    z, mu, logvar, mu_x, logvar_x = self.forward(inputs[:, 0:data_dim])
                    z_right, _, _, _, _ = self.forward(inputs[:, data_dim:2 * data_dim])
                    z_bottom, _, _, _, _ = self.forward(inputs[:, 2 * data_dim:3 * data_dim])
                    loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar, z_right, z_bottom)
                else:
                    z, mu, logvar, mu_x, logvar_x = self.forward(inputs)
                    loss = self.loss_function(inputs, mu_x, logvar_x, z, mu, logvar)

                z_all.append(z.detach().cpu().numpy())

                valid_loss += loss.data.item() * len(inputs)

                #--------------------------------------------------------------------------------#
                # compute gamma for whole evaluation batch
                #--------------------------------------------------------------------------------#
                gamma = self.get_gamma(z).detach()
                gamma_all.append(gamma.cpu().numpy())

                #--------------------------------------------------------------------------------#
                # compute predicted labels by taking the argmax over gamma (q(c | z))
                #--------------------------------------------------------------------------------#
                y_pred.append(torch.argmax(gamma, dim=1).cpu().numpy())
                y.append(labels.cpu().numpy())

            y = np.concatenate(y)
            y_pred = np.concatenate(y_pred)
            z_all = np.concatenate(z_all, axis=1)
            gamma_all = np.concatenate(gamma_all)
       
            acc = cluster_acc(y_pred, y, self.ignore_label)

            print(f'#Epoch {epoch:3d}: Gradient Norm: {grad_norm.item()}, lr: {epoch_lr:.5f}, Train Loss: {train_loss / len(trainloader.dataset):.5f}, ' \
                  f'Valid Loss: {valid_loss / len(validloader.dataset):.5f}, acc: {acc[0]:.5f}')

            if visualize and (epoch % 50 == 0 or epoch == num_epochs - 1):
                print(f'Plot current latent space after {epoch} epochs.')
                self.visualize_latent_space(z_all, y, epoch=epoch)

            if epoch == num_epochs - 1:
                saveArrayAsNetCDF(gamma_all.reshape((self.im_shape[0], self.im_shape[1], gamma_all.shape[-1])),
                                  f'{self.directory}/gamma_epoch{epoch}_reg{self.lambdas}.nc')

        return y_pred

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
