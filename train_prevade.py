import os
import random
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from train_vade import HSIDataset, read_data, read_args

from prevade import PreVade

## Hyper Parameters
num_epochs = 20
lr = 0.001

## Initial Seeding for reproducibility
seed = 2021
random.seed(seed)   
torch.manual_seed(seed)
np.random.seed(seed)
g = torch.Generator()
g.manual_seed(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

def seed_worker(): 
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

## Load the hyperspectral image data
args = read_args()
data_path = f'./{args["image_file"]}/{args["image_file"]}.nc'
im, gt, _, args['gt_file'] = read_data(os.path.expandvars(data_path))
hyperspectral_data = HSIDataset(im, gt, None, args['device'])
dataloader = DataLoader(hyperspectral_data, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=False,
                                    worker_init_fn=seed_worker, generator=g)
args['im'] = im
args['gt'] = gt
args['data_dim'] = im.shape[2]
args['im_shape'] = im.shape

## Initialize an instance of autoencoder
prevade = PreVade(input_dim=args['data_dim'], z_dim=args['latent_dim'], n_centroids=args['n_cl'],
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], im_shape=args['im_shape'][:-1])
prevade.to(args['device'])

optimizer = optim.Adam(filter(lambda p: p.requires_grad, prevade.parameters()), lr=lr)
loss_fn = nn.MSELoss() 

## Training Loop
for epoch in range(num_epochs):
    train_loss = 0
    ## Loop through our batches of data
    for _, (inputs, _) in enumerate(dataloader):
        inputs = Variable(inputs)
        # 1. Forward pass
        z, mu_x = prevade.forward(inputs)
        # 2. Calculate the loss 
        loss = loss_fn(mu_x, inputs)
        train_loss += loss.data.item() * len(inputs)
        # 3. Optimize the zero-grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimize step
        optimizer.step()

    print(f'#Epoch {epoch:3d}: lr: {lr}, Train Loss: {train_loss / len(dataloader.dataset):.5f}, ')

        

    