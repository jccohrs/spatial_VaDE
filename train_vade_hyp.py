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
from torch.utils.data import DataLoader, Dataset
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
    ylim_lower = -1300
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
    def __init__(self, pts, labels, device='cuda'):
        self.data = torch.from_numpy(pts).float().to(device)
        self.labels = torch.from_numpy(labels).int().to(device)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return self.data.shape[0]


def hyper_opt(args):
    ''' optimize over regularization parameter using hyperopt '''
    now = datetime.now()
    time_stamp = f'{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}'

    exp_spec_path = f'epochs{args["epochs"]}_ldim{args["latent_dim"]}_reg{str(args["regularizer"]).replace(".", "_")}'
    exp_dir = f'{args["final_directory"]}/{time_stamp}-{exp_spec_path}'
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logfile_path = f'{exp_dir}/output-{exp_spec_path}.log'
    with Tee(logfile_path):
        print(f'Working on image {args["image_file"]}...')
        print(f'Device: {args["device"]}')
        dataloader = args['dataloader']
        validloader = args['validloader']
        
        if args['spatial_coord']:
            print(f'Added spatial coordinates to spectra.')

        # im and gt not involved in computation on GPU
        im = args['im']
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
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], im_shape=im.shape[:-1],
                one_by_delta_x=args['fd_inv_stepsize'], regularization=args['regularizer'],
                directory=exp_dir, ignore_label=args['ignore_label_inside'])
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

        seg = y_pred.reshape(im.shape[:-1]).astype(np.uint8)
        print(f'Labels in segmentation: {np.unique(seg)}')
        print(f'Number of labels in seg: {len(np.unique(seg))}')

        if args['ignore_label'] is None:
            if args['use_mnf']:
                filename = f'seg-{exp_spec_path}_mnf_all_pxs'
            else:
                filename = f'seg-{exp_spec_path}_all_pxs'
            # save full segmentation
            save_gt_colored_segmentation(f'{exp_dir}/{filename}_full.png', seg, args['gt_file'],
                                         args['ignore_label_inside'], True, None, False, False)
            # save masked segmentation
            save_gt_colored_segmentation(f'{exp_dir}/{filename}_masked.png', seg, args['gt_file'],
                                         args['ignore_label_inside'], False, None, False, True)
            # save wrongly and correctly classified pixels
            visualize_correct_wrong_pixels(f'{exp_dir}/{filename}.png', seg, gt, args['gt_file'],
                                           args['ignore_label_inside'])
            # store found segmentation as netCDF file
            saveArrayAsNetCDF(seg, f'{exp_dir}/{filename}.nc')
        else:
            # TODO: code ignoring pixels in computations outdated!
            # if args['use_mnf']:
            #     filename = f'seg_{args["image_file"]}_ncl{args["n_cl"]}_ldim{args["latent_dim"]}\\
            # _reg{args["regularizer"]}_mnf_ignore_pxs'
            # else:
            #     filename = f'seg_{args["image_file"]}_ncl{args["n_cl"]}_ldim{args["latent_dim"]}\\
            # _reg{args["regularizer"]}_ignore_pxs'
            # seg[gt != ignore_label] = y_pred + 1
            # saveColoredSegmentation(seg, f'"final_directory"/{filename}.png', None, args['ignore_label_inside'],
            #                         True, gt)
            saveArrayAsNetCDF(seg, f'{exp_dir}/{filename}.nc')

        #---------------------------------------------------------------------------------------#
        # evaluate segmentation
        #---------------------------------------------------------------------------------------#

        print('Evaluate segmentation...')
        score = segmentation_scores(seg, gt, args['ignore_label_inside'])
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
    #------------------------------------------------------------------------------#
    # make argument compatible to hyperopt
    #------------------------------------------------------------------------------#
    args = parse_args()
    lam_lower = 0
    lam_upper = 0.025
    # args['regularizer'] = hp.uniform('regularizer', lam_lower, lam_upper)
    args['regularizer'] = 0.0025

    #------------------------------------------------------------------------------#
    # seed the randomness
    #------------------------------------------------------------------------------#
    seed = 2021
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_hyp = seed
    max_evals = 1

    #------------------------------------------------------------------------------#
    # Read the input image file
    #------------------------------------------------------------------------------#
    args['image_file'] = 'IndianPines'

    #------------------------------------------------------------------------------#
    # Specify the features
    #------------------------------------------------------------------------------#
    args['n_cl'] = 16
    args['visualize'] = False
    args['mnf_features'] = 16 #6
    args['latent_dim'] = 10
    args['ignore_label_inside'] = 0
    args['ignore_label'] = None
    args['use_mnf'] = False
    args['use_pca'] = False
    args['spatial_coord'] = False
    #--------------------------------------------------------------------------------#
    # check and create a folder to store all the data and figure
    #--------------------------------------------------------------------------------#

    directory =f'{args["image_file"]}'
    current_directory = os.path.expandvars('${OUTPUT_DIR}/segmentation/vade-tv')
    final_directory = os.path.join(current_directory+'/', directory)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    args['final_directory'] = final_directory

    #--------------------------------------------------------------------------------#
    # read data from the source
    #--------------------------------------------------------------------------------#
    data_path = f'$DATA_DIR/segmentation/{args["image_file"]}/{args["image_file"]}.nc'
    im, gt, _, args['gt_file'] = read_data(os.path.expandvars(data_path))


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
        im=np.dstack((im, norm_x_cor))
        im=np.dstack((im, norm_y_cor))

    args['im'] = im
    args['gt'] = gt
    args['data_dim'] = im.shape[-1]
    args['fd_inv_stepsize'] = max(im.shape[1],im.shape[0])
    #-----------------------------------------------------------------------------#
    # add the right and bottom part of the image to create a new data strucutre
    # so that we can use the variational method

    #comment out the line with # for getting zero padding on right and bottom
    #-----------------------------------------------------------------------------#
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
                im_with_right_and_below[i * im.shape[1] + j, 2*im.shape[2]:3 * im.shape[2]] = im[i, j, :] #
            else:
                im_with_right_and_below[i * im.shape[1] + j, 0:im.shape[2]] = im[i, j, :]
                im_with_right_and_below[i * im.shape[1] + j, im.shape[2]:2 * im.shape[2]] = im[i, j, :] #
                im_with_right_and_below[i * im.shape[1] + j, 2 * im.shape[2]:3 * im.shape[2]] = im[i, j, :] #

    #--------------------------------------------------------------------------------#
    # remove pixels with ignore label
    #--------------------------------------------------------------------------------#

    if args['ignore_label'] is None:
        #pxs = im.reshape((-1, im.shape[-1]))
        pxs = im_with_right_and_below
        lbs = np.ravel(gt)
    else:
        #pxs = im[gt != args['ignore_label']]
        #lbs = gt[gt != args['ignore_label']]
        pxs = im_with_right_and_below
        lbs = np.ravel(gt)
        index = lbs!= args['ignore_label']
        lbs = lbs[index]
        pxs = pxs[index]
    #----------------------------------------------------------------------------------#
    # to avoid problems with segment numbering, rebuild it if ground truth is given
    #----------------------------------------------------------------------------------#
    if lbs is not None:
        lbs = rebuild_segment_numbering(lbs)

    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = HSIDataset(pxs, lbs, args['device'])

    args['dataloader'] = DataLoader(ds, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=False)
    args['validloader'] = DataLoader(ds, batch_size=args['batch_size'], shuffle=False, num_workers=0, pin_memory=False)
    print('Initialized dataset and dataloader.')

    trials=Trials()
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
