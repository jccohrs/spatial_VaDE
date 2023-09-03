
#pylint: disable=no-member
''' script to train vade with hyperopt '''

# TODO: remove unnecessary imports
import os
import random
import re
import time
from typing import final
# import pdb    
from attr import field
import yaml
import argparse
from datetime import datetime
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset as ncDataset
from sklearn.decomposition import PCA
from spectral import calc_stats, mnf, noise_from_diffs
from torch import nn
from torch import optim
from torch.nn.functional import unfold
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision.transforms.functional import pad
from vade import VaDE
from prevade import PreVade

from skimage.exposure import rescale_intensity
from msiplib.io import saveArrayAsNetCDF, Tee
from msiplib.io.segmentation import (save_gt_colored_segmentation)
                                     #,visualize_correct_wrong_pixels)
from msiplib.metrics import segmentation_scores
from msiplib.segmentation import rebuild_segment_numbering

#------------------------------------------------------------------------------#
# seed the randomness, reduce non-determinism
#------------------------------------------------------------------------------#
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True) 

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    cos_theta = dot_product / (norm_v1 * norm_v2)
    radians = np.arccos(cos_theta)
    degrees = np.degrees(radians)
    
    return degrees

def filtered_yaml_dump(args, file):
    with open("./args.yaml", 'r') as f:
        data = yaml.safe_load(f)
    fields = list(data.keys())  
    filtered_args = {field: args[field] for field in fields if field in args}
    print(f"TRAIN_VADE: {filtered_args} in file {file}")
    yaml.dump(filtered_args, file) 
    return

def read_args():
    args = dict()
    
    parser = argparse.ArgumentParser(description='Args parsing from a specified path')
    parser.add_argument('-f', '--file', type=str, help='Path to .yaml file')
    cmd_args = parser.parse_args()

    if cmd_args.file is None:
        with open('args.yaml', 'r') as file:
            args = yaml.load(file, Loader=yaml.FullLoader)
            args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        return args
    
    with open(cmd_args.file, 'r') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
        args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def prepare_data(im, gt, args, seed_worker, generator):
    ''' function that prepares the data and dataloader for the training of VaDE '''

    args['fd_inv_stepsize'] = max(im.shape[1], im.shape[0])
    #------------------------------------------------------------------------------#
    # apply MNF for denoising
    #------------------------------------------------------------------------------#
    if args['use_mnf']:
        print('Denoise data with MNF...')
        sig = calc_stats(im)
        noise_dir = 'lowerright'
        noise = noise_from_diffs(im, direction=noise_dir)
        mnfr = mnf(sig, noise)
        im = mnfr.reduce(im, num=args['mnf_features'])
        args['im'] = rescale_intensity(im, out_range=(0.0, 1.0))
        print('Finished denoising of data with MNF.')
    #------------------------------------------------------------------------------#
    # principal component analysis
    #------------------------------------------------------------------------------#
    if args['use_pca']:
        print('Data with PCA ...')
        data_flat = im.reshape(im.shape[0] * im.shape[1], -1)
        pca = PCA(n_components=0.9999)
        data_pca = pca.fit_transform(data_flat)
        args['im'] = data_pca.reshape(im.shape[0], im.shape[1], data_pca.shape[1])
        print('Finished dimenionality reduction of data with PCA')

    args['im_shape'] = im.shape
    #------------------------------------------------------------------------------#
    # adding spatial value of pixel
    #------------------------------------------------------------------------------#
    if args['spatial_coord']:
        dummy_x = (np.arange(im.shape[0]) / (im.shape[0] - 1)).reshape((im.shape[0], 1))
        dummy_y = (np.arange(im.shape[1]) / (im.shape[1] - 1))
        dummy_zero = np.zeros((im.shape[0], im.shape[1]))
        norm_x_cor = dummy_zero + dummy_x
        norm_y_cor = dummy_zero + dummy_y
        im = np.dstack((im, norm_x_cor))
        im = np.dstack((im, norm_y_cor))

    # if training on patches, pad image and extend pixels by their suroundings
    if args['patches']:
        im_tensor = torch.permute(torch.from_numpy(im), (2, 0, 1))
        im_tensor_pad = pad(im_tensor, padding=args['patch_size'] // 2, fill=0, padding_mode='reflect')
        im = torch.permute(im_tensor_pad, (1, 2, 0)).numpy()

    args['im'] = im
    args['gt'] = gt
    args['data_dim'] = args['im_shape'].shape[-1] * args['patch_size']**2 if args['patches'] \
                       else args['im_shape'][-1]

    #-----------------------------------------------------------------------------#
    # add the right and bottom neighbor to each pixel if total variation regularization is used.

    # comment out the line with # for getting zero padding on right and bottom
    #-----------------------------------------------------------------------------#
    if args['regularizer'] != 0:
        im_with_right_and_below = np.zeros((im.shape[0] * im.shape[1], 3 * im.shape[2]))
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if (j < (im.shape[1] - 1)) and (i < (im.shape[0] - 1)):
                    im_with_right_and_below[i * im.shape[1] + j, 0:im.shape[2]] = im[i, j, :]
                    im_with_right_and_below[i * im.shape[1] + j, im.shape[2]:2 * im.shape[2]] = im[i, j + 1, :]
                    im_with_right_and_below[i * im.shape[1] + j, 2 * im.shape[2]:3 * im.shape[2]] = im[i + 1, j, :]
                elif (j == (im.shape[1] - 1)) and (i < (im.shape[0] - 1)):
                    im_with_right_and_below[i * im.shape[1] + j, 0:im.shape[2]] = im[i, j, :]
                    im_with_right_and_below[i * im.shape[1] + j, im.shape[2]:2 * im.shape[2]] = im[i, j, :] #
                    im_with_right_and_below[i * im.shape[1] + j, 2 * im.shape[2]:3 * im.shape[2]] = im[i + 1, j, :]
                elif (j < (im.shape[1] - 1)) and (i == (im.shape[0] - 1)):
                    im_with_right_and_below[i * im.shape[1] + j, 0:im.shape[2]] = im[i, j, :]
                    im_with_right_and_below[i * im.shape[1] + j, im.shape[2]:2 * im.shape[2]] = im[i, j + 1, :]
                    im_with_right_and_below[i * im.shape[1] + j, 2 * im.shape[2]:3 * im.shape[2]] = im[i, j, :] #
                else:
                    im_with_right_and_below[i * im.shape[1] + j, 0:im.shape[2]] = im[i, j, :]
                    im_with_right_and_below[i * im.shape[1] + j, im.shape[2]:2 * im.shape[2]] = im[i, j, :] #
                    im_with_right_and_below[i * im.shape[1] + j, 2 * im.shape[2]:3 * im.shape[2]] = im[i, j, :] #
        im = im_with_right_and_below

    #----------------------------------------------------------------------------------#
    # to avoid problems with segment numbering, rebuild it if ground truth is given
    #----------------------------------------------------------------------------------#
    if gt is not None:
        gt = rebuild_segment_numbering(gt)

    if args['patches']:
        ds = HSIDataset(im, gt, args['patch_size'], args['device'])
    else:
        ds = HSIDataset(im, gt, None, args['device'])

    args['dataloader'] = DataLoader(ds, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=False,
                                    worker_init_fn=seed_worker, generator=generator)
    args['validloader'] = DataLoader(ds, batch_size=args['batch_size'], shuffle=False, num_workers=0, pin_memory=False)
    print('Initialized dataset and dataloader.')

    return args

def save_segmentation(exp_dir, exp_spec_path, seg, args):
    ''' function to save segmentation results '''
    print(f'Labels in segmentation: {np.unique(seg)}')
    print(f'Number of labels in seg: {len(np.unique(seg))}')

    if args['use_mnf']:
        filename = f'seg-{exp_spec_path}_mnf_all_pxs'
    else:
        filename = f'seg-{exp_spec_path}_all_pxs'
    # save full segmentation
    save_gt_colored_segmentation(f'{exp_dir}/{filename}_full.png', seg, args['gt_file'],
                                args['ignore_label'], True, None, False, False)
    # save masked segmentation
    save_gt_colored_segmentation(f'{exp_dir}/{filename}_masked.png', seg, args['gt_file'],
                                args['ignore_label'], False, None, False, True)
    # save wrongly and correctly classified pixels
    #visualize_correct_wrong_pixels(f'{exp_dir}/{filename}.png', seg, args['gt'], args['gt_file'],
    #                           args['ignore_label'])
    # store found segmentation as netCDF file
    saveArrayAsNetCDF(seg, f'{exp_dir}/{filename}.nc')


def parse_log(path_to_log):
    """ Parse log file """

    regex_epoch = re.compile(r'#Epoch(\s+)(\d+)')
    regex_train_loss = re.compile(r'Train Loss: (-?\d+\.?\d*)')
    regex_valid_loss = re.compile(r'Valid Loss: (-?\d+\.?\d*)')
    regex_accuracy = re.compile(r'acc: (\d+\.?\d*)')

    df = []
    with open(path_to_log, 'r', encoding='utf-8') as f:
        for line in f:
            epoch_match = regex_epoch.search(line)
            train_loss_match = regex_train_loss.search(line)
            valid_loss_match = regex_valid_loss.search(line)
            accuracy_match = regex_accuracy.search(line)

            if epoch_match:
                epoch = int(epoch_match.group(2))
            if train_loss_match:
                train_loss = float(train_loss_match.group(1))
            if valid_loss_match:
                valid_loss = float(valid_loss_match.group(1))
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))

            if epoch_match and epoch != -1:
                df.append({'epoch': epoch, 'train loss': train_loss, 'valid loss': valid_loss, 'accuracy': accuracy})

    return pd.DataFrame(df)


def plot_stats(path, data):
    ''' plot training statistics '''
    averaging_width = 11
    ylim_lower = min(data['train loss'].min(), data['valid loss'].min())
    # plot train loss
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Training loss')
    ax.plot(data['epoch'], data['train loss'], label='Raw')
    ax.plot(data['epoch'], pd.Series(data['train loss']).rolling(averaging_width).mean(),
            label=f'Smoothed (width {averaging_width})')
    ax.set_ylim(bottom=ylim_lower, top=0)
    ax.set(xlabel='Epoch', ylabel='Train loss')
    ax.legend()
    fig.savefig(f'{path}-train-loss.png')

    # plot validation loss
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Validation loss')
    ax.plot(data['epoch'], data['valid loss'], label='Raw')
    ax.plot(data['epoch'], pd.Series(data['valid loss']).rolling(averaging_width).mean(),
            label=f'Smoothed (width {averaging_width})')
    ax.set_ylim(bottom=ylim_lower, top=0)
    ax.set(xlabel='Epoch', ylabel='Validation loss')
    ax.legend()
    fig.savefig(f'{path}-valid-loss.png')

    # plot accuracy
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Overall accuracy')
    ax.plot(data['epoch'], data['accuracy'], label='Raw')
    ax.plot(data['epoch'], pd.Series(data['accuracy']).rolling(averaging_width).mean(),
            label=f'Smoothed (width {averaging_width})')
    ax.set_ylim(bottom=0, top=1)
    ax.set(xlabel='Epoch', ylabel='Accuracy')
    ax.legend()
    fig.savefig(f'{path}-accuracy.png')


def read_data(filename):
    ''' read data: input image and ground truth '''
    #--------------------------------------------------------------------------------#
    # init dataset and data loader
    #--------------------------------------------------------------------------------#
    ncfile = ncDataset(filename, 'r')
    inputimage = ncfile['data'][:].data.astype(np.float32)

    gt_file = filename.replace('.nc', '_gt.png')

    #--------------------------------------------------------------------------------#
    # normalize image to have values between [0, 1]
    #--------------------------------------------------------------------------------#
    inputimage = rescale_intensity(inputimage, out_range=(0.0, 1.0))

    #--------------------------------------------------------------------------------#
    # if the image is grayscale, add a third dimension of size 1
    #--------------------------------------------------------------------------------#
    if inputimage.ndim == 2:
        inputimage = np.expand_dims(inputimage, axis=2)

    #--------------------------------------------------------------------------------#
    # load ground truth if available
    #--------------------------------------------------------------------------------#

    seg_gt = ncfile['groundtruth']['segmentation_mask'][:].data
    k_gt = ncfile['groundtruth']['num_segments'][:].data.item()

    return inputimage, seg_gt, k_gt, gt_file


class HSIDataset(Dataset):
    ''' define HSI dataset '''
    def __init__(self, im, labels, patch_size=None, device='cuda'):
        if patch_size is not None:
            # convert image into torch array and bring axes in the order (C, H, W)
            im_torch = torch.permute(torch.from_numpy(im), (2, 0, 1))
            # add a fourth leading dimension to match the requirements of unfold
            im_patches = im_torch.unsqueeze(dim=0)
            # extract patches from the image. the output of unfold is a tensor of size (C * "pixels in patch", H * W)
            # the function extracts the data of the patches from left to right and top to bottom.
            # hence, im_patches[:, i * W + j] contains all values of the patch of pixel (i, j) (images was padded such
            # that there is one patch for each pixel.
            # the values of the patch are stored in the one-dimensional array im_patches[:, i * W + j] and arranged
            # such that the array starts with the first row of the first channel of the patch, continues with the
            # second row, until the patch in the first channel was unrolled, then the second channel follows in the
            # the same way, row by row.
            im_patches = unfold(im_patches, patch_size, stride=1, padding=0, dilation=1).squeeze()
            # transpose im_patches to have the patches as rows
            self.data = im_patches.permute((1, 0)).float().to(device)
        else:
            self.data = torch.from_numpy(im.reshape((-1, im.shape[-1]))).float().to(device)

        self.labels = torch.from_numpy(np.ravel(labels)).int().to(device)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return self.data.shape[0]

def train_prevade(args, pretrain_logfile_path):
    #### Read hyperspectral data
    seed = int(args['seed'])
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    data_path = f'$DATA_DIR/segmentation/{args["image_file"]}/{args["image_file"]}.nc'
    im, gt, _, args['gt_file'] = read_data(os.path.expandvars(data_path))
    args = prepare_data(im, gt, args, seed_worker, g)

    dataloader = args['dataloader']

    ## Fetch relevant hyperparameters
    lr = args['lr']
    num_epochs = args['epochs_pretrain']

    ## Initialize an instance of autoencoder
    prevade = PreVade(spectral_dim=args['data_dim'], z_dim=args['latent_dim'], n_centroids=args['n_cl'],
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], im_shape=args['im_shape'][:-1], spatial=args['spatial_coord'])
    prevade.to(args['device'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, prevade.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    grad_norm_list = []
    grad_angle_list = []
    grad_angle_list_first = []
    previous_grad = None ## initialize dummy previous gradient to avoid error
    first_grad = []
    with Tee(pretrain_logfile_path):
        
        ## Training Loop
        print(f'Starting a pretraining loop for {num_epochs} epochs')
        for epoch in range(num_epochs):
            train_loss = 0
            ## Loop through our batches of data
            for idx, (inputs, _) in enumerate(dataloader):
                inputs = Variable(inputs)
                # 1. Forward pass
                z, mu_x = prevade.forward(inputs)
                # 2. Calculate the loss 
                if args['spatial_coord'] == True:
                    loss = loss_fn(mu_x, inputs[:, :-2])
                else:
                    loss = loss_fn(mu_x, inputs)
                train_loss += loss.data.item() * len(inputs)
                # 3. Optimize the zero-grad
                optimizer.zero_grad()
                # 4. Loss backward
                loss.backward()
                # 5. Optimize step
                optimizer.step()

                ############ LOG GRADIENT NORM AND ANGLES ############
                grad_norm = 0.0
                current_grad = []
                for param in prevade.parameters():
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
                    angle = np.arccos(cosine_similarity) * 180 / np.pi
                    grad_angle_list.append(angle.item())
                previous_grad = current_grad.clone()
                ## calculate cur-to-first gradient comparison
                cosine_similarity = torch.dot(current_grad, first_grad) / (torch.norm(current_grad) * torch.norm(first_grad))
                angle = np.arccos(cosine_similarity) * 180 / np.pi
                grad_angle_list_first.append(angle.item())
                grad_norm_list.append(grad_norm.item())
                #######################################################

            print(f'#Epoch {epoch:3d}: lr: {lr}, Train Loss: {train_loss}, Gradient Norm: {grad_norm}')
    with open(args['exp_dir']+'/gradient_info_pretrain.log', 'w') as file:
        file.write("Gradient Norm: " + str(grad_norm_list) + "\n")
        file.write("Gradient Angle (prev): " + str(grad_angle_list) + "\n")
        file.write("Gradient Angle (first): " + str(grad_angle_list_first) + "\n")

    ## Save Model
    pretrain_filename = f'pretrain_ep{num_epochs}_lr{lr}.pt'
    save_location = './' + pretrain_filename
    prevade.save_model(save_location)
    return save_location

# def pretrain_output(args, save_location, input):
#     prevade = PreVade(spectral_dim=args['data_dim'], z_dim=args['latent_dim'], n_centroids=args['n_cl'],
#                     encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], im_shape=args['im_shape'][:-1], spatial=args['spatial_coord'])
#     prevade.to(args['device'])
#     prevade.load_model(save_location)
#     z, reconstructed_x = prevade.forward(input)
#     return reconstructed_x

def train_vade(args):
    ''' main function '''
    # TODO: total variation plus patches not implemented yet!    
    # determinism for dataloader
    seed = args['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    def seed_worker(worker_id): # pylint:disable=unused-argument
        # torch.initial_seed() returns the initial seed for generating random numbers.
        # if torch.manual_seed(val) is used before, torch.initial_seed() returns val.
        worker_seed = torch.initial_seed() % 2**32

        # seed of random and numpy already set above
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    #--------------------------------------------------------------------------------#
    # check and create a folder to store all the data and figure
    #--------------------------------------------------------------------------------#

    directory =f'{args["image_file"]}'
    if args['patches'] and not args['regularizer'] == 0:
        current_directory = os.path.expandvars('${OUTPUT_DIR}/segmentation/vade/patch-tv')
    elif args['patches']:
        current_directory = os.path.expandvars('${OUTPUT_DIR}/segmentation/vade/patch')
    elif not args['regularizer'] == 0:
        current_directory = os.path.expandvars('${OUTPUT_DIR}/segmentation/vade/tv')
    else:
        current_directory = os.path.expandvars('${OUTPUT_DIR}/segmentation/vade/pure')
    final_directory = os.path.join(current_directory+'/', directory)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    args['final_directory'] = final_directory

    print(f"VADE: {args['final_directory'] + '/args.yaml'}")
    # create yaml file from arguments
    with open(args['final_directory'] + '/args.yaml', 'w') as file:
        filtered_yaml_dump(args, file)

    #--------------------------------------------------------------------------------#
    # read data from the source
    #--------------------------------------------------------------------------------#
    data_path = f'$DATA_DIR/segmentation/{args["image_file"]}/{args["image_file"]}.nc'
    im, gt, _, args['gt_file'] = read_data(os.path.expandvars(data_path))

    # prepare dataset and dataloader for training and validation
    args = prepare_data(im, gt, args, seed_worker, g)
    
    now = datetime.now()
    time_stamp = f'{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}'

    if args['patches']:
        exp_spec_path = f'ep{args["epochs"]}_ldim{args["latent_dim"]}_reg{str(args["regularizer"]).replace(".", "_")}' \
                        f'_spat{int(args["spatial_coord"])}_patch{args["patch_size"]}'
    else:
        exp_spec_path = f'ep{args["epochs"]}_ldim{args["latent_dim"]}_reg{str(args["regularizer"]).replace(".", "_")}' \
                        f'_spat{int(args["spatial_coord"])}'
    exp_dir = f'{args["final_directory"]}/{time_stamp}-{exp_spec_path}'
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    args['exp_dir'] = exp_dir   

    logfile_path = f'{exp_dir}/output-{exp_spec_path}.log'

    # if pretraining is enabled, run pre-training
    pretrain_logfile_path = f'{exp_dir}/output-pretrain-{exp_spec_path}.log'
    pretrain_location = None
    if args['pretrain'] == True:
        pretrain_location = train_prevade(args, pretrain_logfile_path)

    prevade = PreVade(spectral_dim=args['data_dim'], z_dim=args['latent_dim'], n_centroids=args['n_cl'],
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], im_shape=args['im_shape'][:-1], spatial=args['spatial_coord'])
    prevade.to(args['device'])
    prevade.load_model(pretrain_location)

    with Tee(logfile_path):
        print(f'Running training on {os.uname()[1]}')
        print(f'Working on image {args["image_file"]}...')
        print(f'Device: {args["device"]}')
        dataloader = args['dataloader']
        validloader = args['validloader']

        if args['spatial_coord']:
            print('Added spatial coordinates to spectra.')
            print('Spatial coordinates are input into a second input layer. '\
                  'Results of spectral and spatial input are added.')

        if args['patches']:
            print(f'Input patches of size {args["patch_size"]}.')

        # gt not involved in computation on GPU
        gt = args['gt']

        #--------------------------------------------------------------------------------#
        #calculate the initial time
        #--------------------------------------------------------------------------------#
        start_time = time.time()
        print(f'regularizer : {args["regularizer"]}')


        print(f'input_dim={args["data_dim"]}, im_shape={args["im_shape"][:-1]}')
        #------------------------------------------------------------------------------------#
        # init model
        #------------------------------------------------------------------------------------#
        vade = VaDE(input_dim=args['data_dim'], z_dim=args['latent_dim'], n_centroids=args['n_cl'],
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], im_shape=args['im_shape'][:-1],
                one_by_delta_x=args['fd_inv_stepsize'], regularization=args['regularizer'],
                directory=exp_dir, ignore_label=args['ignore_label'], spatial=args['spatial_coord'])
        print('Initialized VaDE model.')
        vade.to(args['device'])
        print(f'Pushed model to {args["device"]}.')

        #------------------------------------------------------------------------------------#
        # load pretrained models if provided
        #------------------------------------------------------------------------------------#

        #### DON"T LOAD PRETRAIN WEIGHTS INTO VADE MODEL, BUT USE IT AS DENOISING IN fit() function
        if pretrain_location is not None:
            # print(f"Load model from {pretrain_location}...")
            # vade.load_model(pretrain_location)
            # print('Loaded pretrained network weights.')
            shutil.move(pretrain_location, exp_dir + '/pretrain.pt')
        

        ### To check pretraining weights
        ### vade.load_state_dict(torch.load("/home/chagnaa/output/segmentation/vade/pure/IndianPines/Study_reg0_spat1_pretrain0/2023-7-26-11-25-29-ep500_ldim10_reg0_0_spat1/vade_params-ep500_ldim10_reg0_0_spat1.pt"))
        ### weights = vade.state_dict()
        ### pdb.set_trace()
        
        print("Initializing through GMM...")
        vade.initialize_gmm(dataloader)
        print("Initialized through GMM.")

        print('Fit the model...')
        y_pred = vade.fit(dataloader, validloader, lr=args['lr'], num_epochs=args['epochs'],
                          anneal=False, visualize=args['visualize'], model=vade, pretrain_model=prevade)

        print('Save found parameters...')
        if args['save']:
            save_location = f'{exp_dir}/vade_params-{exp_spec_path}.pt'
            print(f'Save weights at {save_location}')
            vade.save_model(save_location)

        print('Save found segmentation...')
        seg = y_pred.reshape(args['im_shape'][:-1]).astype(np.uint8)
        save_segmentation(exp_dir, exp_spec_path, seg, args)

        #---------------------------------------------------------------------------------------#
        # evaluate segmentation
        #---------------------------------------------------------------------------------------#

        print('Evaluate segmentation...')
        score = segmentation_scores(seg, gt, args['ignore_label'])
        print(score)
        #---------------------------------------------------------------------------------------#
        #print the total time of execution
        #---------------------------------------------------------------------------------------#

        print(f"--- {(time.time() - start_time)} seconds ---")

    # parse log file, store the extracted values and plot the stats
    df_log = parse_log(logfile_path)
    df_log.to_csv(f'{exp_dir}/exp_stats.csv', sep=',', header=True, index=False)
    plot_stats(f'{exp_dir}/plot_stats', df_log)

    return score['overallAcc']

def main(seed=32, args = None):
    """ main function """

    # parse arguments from command line and config file
    if args is None:
        args = read_args()
    args['seed'] = int(seed)

    # train VaDE with given hyperparameters
    train_vade(args)

if __name__ == "__main__":
    main()
