#pylint: disable=no-member
''' script to train vade with hyperopt '''

import argparse
import os
import random
import re
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hyperopt import Trials, fmin, hp, rand
from netCDF4 import Dataset as ncDataset
from sklearn.decomposition import PCA
from spectral import calc_stats, mnf, noise_from_diffs
from torch.nn.functional import unfold
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pad
from vade import VaDE

from msiplib.image.manipulators import normalize_image
from msiplib.io import saveArrayAsNetCDF, Tee
from msiplib.io.segmentation import (save_gt_colored_segmentation,
                                     visualize_correct_wrong_pixels)
from msiplib.metrics import segmentation_scores
from msiplib.segmentation import rebuild_segment_numbering

def parse_args():
    ''' function to parse arguments '''
    parser = argparse.ArgumentParser(description='VaDE: Variational Deep Embedding')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                        help='learning rate for training (default: 0.0001)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--pretrain', type=str, default="", metavar='N',
                        help='path to pretrained model')
    parser.add_argument('--regularizer',type=float, default=0.01, metavar='N',
                        help='value for reularization')
    parser.add_argument('--save', type=bool, default=True,
                        help='If true, save founds weights after training.')
    args = parser.parse_args()

    return vars(args)


def classify(output_path, image, weights, gt, n_cl, gt_file, ignore_label=None, add_spatial=False):
    ''' classify an image using a trained VaDE model '''

    im = normalize_image(image)
    args = {'spatial_coord': add_spatial}
    args['use_mnf'] = False
    args['use_pca'] = False
    args['ignore_label'] = ignore_label
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['batch_size'] = 100
    args['latent_dim'] = 10
    args['n_cl'] = n_cl
    args['regularizer'] = 0.002
    args['gt_file'] = gt_file

    args = prepare_data(im, gt, args, None, None)

    # init model
    vade = VaDE(input_dim=args['data_dim'], z_dim=args['latent_dim'], n_centroids=args['n_cl'],
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], im_shape=im.shape[:-1],
                one_by_delta_x=args['fd_inv_stepsize'], regularization=args['regularizer'],
                directory=output_path, ignore_label=args['ignore_label'])

    print('Initialized VaDE model.')
    vade.to(args['device'])
    print(f'Pushed model to {args["device"]}.')

    #------------------------------------------------------------------------------------#
    # load pretrained weights
    #------------------------------------------------------------------------------------#

    print(f"Load model from {weights}...")
    vade.load_model(weights)
    print('Loaded pretrained network weights.')

    print('Use model to classify the data...')
    loss, _, y_pred, gamma = vade.predict(args['validloader'])

    saveArrayAsNetCDF(gamma.reshape((args['im'].shape[0], args['im'].shape[1], gamma.shape[-1])),
                      f'{output_path}/gamma.nc')
    seg = y_pred.reshape((args['im'].shape[:-1]))

    print(f'Validation loss: {loss / len(args["validloader"])}')
    print(f'Segmentation scores: {segmentation_scores(seg, args["gt"], args["ignore_label_inside"])}')

    save_segmentation(output_path, f'prediction_spat{int(args["spatial_coord"])}', seg, args)


def prepare_data(im, gt, args, seed_worker, generator):
    ''' function that prepares the data and dataloader for the training of VaDE '''
    # TODO: convert data into torch array as soon as possible and work only in torch, possibly on GPU

    args['im_shape'] = im.shape
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
        args['im'] = normalize_image(im)
        print('Finished denoising of data with MNF.')
    #------------------------------------------------------------------------------#
    # principal component analysis
    #------------------------------------------------------------------------------#
    if args['use_pca']:
        print('Data with PCA ...')
        data_flat = im.reshape(im.shape[0]*im.shape[1],-1)
        pca = PCA(n_components=0.9999)
        data_pca = pca.fit_transform(data_flat)
        im = data_pca.reshape(im.shape[0],im.shape[1],data_pca.shape[1])
        print('Finished dimenionality reduction of data with PCA')
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
    args['data_dim'] = im.shape[-1] * args['patch_size']**2 if args['patches'] else im.shape[-1]
    args['fd_inv_stepsize'] = max(im.shape[1], im.shape[0])

    #-----------------------------------------------------------------------------#
    # add the right and bottom neighbor to each pixel if total variation regularization is used.

    # comment out the line with # for getting zero padding on right and bottom
    #-----------------------------------------------------------------------------#
    if not args['regularizer'] == 0:
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

    # args['dataloader'] = DataLoader(ds, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=False)
    args['dataloader'] = DataLoader(ds, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=False,
                                    worker_init_fn=seed_worker, generator=generator)
    # args['validloader'] = DataLoader(ds, batch_size=args['batch_size'], shuffle=False, num_workers=0, pin_memory=False)
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
    visualize_correct_wrong_pixels(f'{exp_dir}/{filename}.png', seg, args['gt'], args['gt_file'],
                                args['ignore_label'])
    # store found segmentation as netCDF file
    saveArrayAsNetCDF(seg, f'{exp_dir}/{filename}.nc')


def parse_log(path_to_log):
    """Parse log file
    """

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
    inputimage = normalize_image(inputimage)

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


def hyper_opt(args):
    ''' optimize over regularization parameter using hyperopt '''
    now = datetime.now()
    time_stamp = f'{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}'

    if args['patches']:
        exp_spec_path = f'ep{args["epochs"]}_ldim{args["latent_dim"]}_reg{str(args["regularizer"]).replace(".", "_")}_spat{int(args["spatial_coord"])}_patch{args["patch_size"]}'
    else:
        exp_spec_path = f'ep{args["epochs"]}_ldim{args["latent_dim"]}_reg{str(args["regularizer"]).replace(".", "_")}_spat{int(args["spatial_coord"])}'
    exp_dir = f'{args["final_directory"]}/{time_stamp}-{exp_spec_path}'
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logfile_path = f'{exp_dir}/output-{exp_spec_path}.log'
    with Tee(logfile_path):
        print(f'Running training on {os.uname()[1]}')
        print(f'Working on image {args["image_file"]}...')
        print(f'Device: {args["device"]}')
        dataloader = args['dataloader']
        validloader = args['validloader']

        if args['spatial_coord']:
            print('Added spatial coordinates to spectra.')

        if args['patches']:
            print(f'Input patches of size {args["patch_size"]}.')

        # gt not involved in computation on GPU
        gt = args['gt']

        #--------------------------------------------------------------------------------#
        #calculate the initial time
        #--------------------------------------------------------------------------------#
        start_time = time.time()
        print(f'regularizer : {args["regularizer"]}')
        #------------------------------------------------------------------------------------#
        # init model
        #------------------------------------------------------------------------------------#
        vade = VaDE(input_dim=args['data_dim'], z_dim=args['latent_dim'], n_centroids=args['n_cl'],
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], im_shape=args['im_shape'][:-1],
                one_by_delta_x=args['fd_inv_stepsize'], regularization=args['regularizer'],
                directory=exp_dir, ignore_label=args['ignore_label'])
        print('Initialized VaDE model.')
        vade.to(args['device'])
        print(f'Pushed model to {args["device"]}.')

        #------------------------------------------------------------------------------------#
        # load pretrained models if provided
        #------------------------------------------------------------------------------------#

        if args['pretrain'] != "":
            print(f"Load model from {args['pretrain']}...")
            vade.load_model(args['pretrain'])
            print('Loaded pretrained network weights.')
        print("Initializing through GMM...")
        vade.initialize_gmm(dataloader)
        print("Initialized through GMM.")

        print('Fit the model...')
        y_pred = vade.fit(dataloader, validloader, lr=args['lr'], num_epochs=args['epochs'],
                          anneal=False, visualize=args['visualize'])

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

    return -score['overallAcc']


def main():
    ''' main function '''
    # TODO: total variation plus patches not implemented yet!
    #------------------------------------------------------------------------------#
    # make argument compatible to hyperopt
    #------------------------------------------------------------------------------#
    args = parse_args()
    lam_lower = 0
    lam_upper = 0.025
    # args['regularizer'] = hp.uniform('regularizer', lam_lower, lam_upper)
    args['regularizer'] = 0
    max_evals = 1

    #------------------------------------------------------------------------------#
    # seed the randomness, reduce non-determinism
    #------------------------------------------------------------------------------#
    seed = 2021
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_hyp = seed
    g = torch.Generator()
    g.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # determinism for dataloader
    def seed_worker(worker_id): # pylint:disable=unused-argument
        # torch.initial_seed() returns the initial seed for generating random numbers.
        # if torch.manual_seed(val) is used before, torch.initial_seed() returns val.
        worker_seed = torch.initial_seed() % 2**32

        # seed of random and numpy already set above
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    #------------------------------------------------------------------------------#
    # Read the input image file
    #------------------------------------------------------------------------------#
    args['image_file'] = 'SalinasA'

    #------------------------------------------------------------------------------#
    # Specify the features
    #------------------------------------------------------------------------------#
    args['n_cl'] = 6
    args['visualize'] = False
    args['mnf_features'] = 16 #6
    args['latent_dim'] = 10
    args['ignore_label'] = 0
    args['patches'] = False
    args['patch_size'] = 5 # only odd values
    args['use_mnf'] = False
    args['use_pca'] = False
    args['spatial_coord'] = True
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    #--------------------------------------------------------------------------------#
    # read data from the source
    #--------------------------------------------------------------------------------#
    data_path = f'$DATA_DIR/segmentation/{args["image_file"]}/{args["image_file"]}.nc'
    im, gt, _, args['gt_file'] = read_data(os.path.expandvars(data_path))

    # prepare dataset and dataloader for training and validation
    args = prepare_data(im, gt, args, seed_worker, g)

    # run hyperopt to do hyperparameter tuning
    trials = Trials()
    best = fmin(fn=hyper_opt, space=args, algo=rand.suggest, max_evals=max_evals,
                rstate=np.random.default_rng(seed_hyp), trials=trials)

    with open(f'{final_directory}/hyperopt_best.txt', 'a', encoding='utf-8') as f:
        f.write(str(best))

    df = []
    if type(args['regularizer']) is not type(hp.uniform('dummy', 0, 1)):
        # handle case when lambda is set to a fixed value
        df_dict = {'regularizer': args['regularizer']}
        df_dict['loss']= -trials.results[0]['loss']
        df.append(df_dict)
    else:
        for it in range(max_evals):
            df_dict = {}
            for key, value in trials.vals.items():
                df_dict[key] = value[it]
            df_dict['loss']= -trials.results[it]['loss']
            df.append(df_dict)

    df = pd.DataFrame(df)

    # append results of hyperopt run to csv file
    output_hyperopt = f'{final_directory}/hyp_{args["image_file"]}.txt'
    if os.path.exists(output_hyperopt):
        # if file already exists, append to it
        df.to_csv(output_hyperopt, sep=" ", index=False, mode='a', header=False)
    else:
        # it file does not exist, create a new one
        df.to_csv(output_hyperopt, sep=" ", index=False, mode='w', header=True)

    # read the full hyperopt results file and update the plot
    df = pd.read_csv(output_hyperopt, sep=' ', header=0)
    fig, ax = plt.subplots(len(df.columns) - 1)
    for i in range(len(df.columns) - 1):
        if len(df.columns) - 1 == 1:
            ax.scatter(df[df.columns[i]], df['loss'])
            ax.set(xlabel=df.columns[i], ylabel='OA')
        else:
            ax[i].scatter(df[df.columns[i]], df['loss'])
            ax[i].set(xlabel=df.columns[i], ylabel='OA')

    fig.savefig(f'{final_directory}/lambda_vs_OA.png')

if __name__ == "__main__":
    main()
