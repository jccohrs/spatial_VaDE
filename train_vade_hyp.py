#pylint: disable=no-member
''' script to train vade with hyperopt '''

import argparse
import os
import random
import time

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
from msiplib.io import saveArrayAsNetCDF
from msiplib.io.segmentation import (save_gt_colored_segmentation,
                                     saveColoredSegmentation)
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
    parser.add_argument('--save', type=str, default="model_params/trained_vade.pt", metavar='N',
                        help='path to location where to store the found weights')
    args = parser.parse_args()

    return vars(args)


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
    def __init__(self, pts, labels):
        self.data = torch.from_numpy(pts).float()
        self.labels = torch.from_numpy(labels).int()

    def __getitem__(self, item):
        return self.data[item], self.labels[item]

    def __len__(self):
        return self.data.shape[0]


def main():
    ''' main function '''
    #------------------------------------------------------------------------------#
    # make argument compatible to hyperopt
    #------------------------------------------------------------------------------#
    args = parse_args()
    args['regularizer'] = hp.uniform('regularizer', 0, 0.025),
    args['save'] = os.path.expandvars('${OUTPUT_DIR}/segmentation/vade-tv/trained_vade.pt')

    #------------------------------------------------------------------------------#
    # seed the randomness
    #------------------------------------------------------------------------------#
    seed = 2021
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_hyp = seed
    max_evals = 25
    #------------------------------------------------------------------------------#
    # Read the input image file
    #------------------------------------------------------------------------------#
    image_file = 'Salinas'
    #groundtruth_file ='SalinasA'


    #------------------------------------------------------------------------------#
    # Specify the features
    #------------------------------------------------------------------------------#
    n_cl =16 #6 #7
    mnf_features =16 #6
    latent_dim = 10
    ignore_label = None
    ignore_label_inside = 0
    use_mnf = False
    use_pca = False
    use_pixel_space_cord = False
    #--------------------------------------------------------------------------------#
    # check and create a folder to store all the data and figure
    #--------------------------------------------------------------------------------#

    directory ='Salinas_0_0025'
    current_directory = os.path.expandvars('${OUTPUT_DIR}/segmentation/vade-tv')
    # current_directory = os.getcwd()
    final_directory = os.path.join(current_directory+'/', directory)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    #--------------------------------------------------------------------------------#
    # read data from the source
    #--------------------------------------------------------------------------------#
    im, gt, _, gt_file = read_data(os.path.expandvars(f'$DATA_DIR/segmentation/{image_file}/{image_file}.nc'))
    #im, _, _, _ = read_data(os.path.expandvars('${DATA_DIR}/segmentation/{0}/{0}.nc').format(image_file))
    #_, gt, _, gt_file = read_data(os.path.expandvars('${DATA_DIR}/segmentation/{0}/{0}.nc').format(groundtruth_file))


    #------------------------------------------------------------------------------#
    # apply MNF for denoising
    #------------------------------------------------------------------------------#
    if use_mnf:
        print('Denoise data with MNF...')
        sig = calc_stats(im)
        noise_dir = 'lowerright'
        noise = noise_from_diffs(im, direction=noise_dir)
        mnfr = mnf(sig, noise)
        im = mnfr.reduce(im, num=mnf_features)
        im = normalize_image(im)
        print('Finished denoising of data with MNF.')
    #------------------------------------------------------------------------------#
    # principal component analysi
    #------------------------------------------------------------------------------#
    if use_pca:
        print('Data with PCA ...')
        data_flat=im.reshape(im.shape[0]*im.shape[1],-1)
        pca =PCA(n_components=0.9999)
        data_pca=pca.fit_transform(data_flat)
        im=data_pca.reshape(im.shape[0],im.shape[1],data_pca.shape[1])
        print('Finished dimenionality reduction of data with PCA')
    #------------------------------------------------------------------------------#
    # adding spatial value of pixel
    #------------------------------------------------------------------------------#
    if use_pixel_space_cord:
        dummy_x=(np.arange(im.shape[0])/(im.shape[0]-1)).reshape((im.shape[0],1))
        dummy_y=(np.arange(im.shape[1])/(im.shape[1]-1))
        dummy_zero=np.zeros((im.shape[0],im.shape[1]))
        norm_x_cor = dummy_zero + dummy_x
        norm_y_cor = dummy_zero + dummy_y
        im=np.dstack((im,norm_x_cor))
        im=np.dstack((im,norm_y_cor))

    data_dim = im.shape[-1]
    one_by_delta_x=max(im.shape[1],im.shape[0])
    #-----------------------------------------------------------------------------#
    # add the right and bottom part of the image to create a new data strucutre
    # so that we can use the variational method

    #comment out the line with # for getting zero padding on right and bottom
    #-----------------------------------------------------------------------------#
    im_with_right_and_below = np.zeros((im.shape[0] * im.shape[1], 3 * im.shape[2]))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if (j<(im.shape[1] -1)) and (i<(im.shape[0] -1)):
                im_with_right_and_below[i*im.shape[1]+j,0:im.shape[2]]=im[i,j,:]
                im_with_right_and_below[i*im.shape[1]+j,im.shape[2]:2*im.shape[2]]=im[i,j+1,:]
                im_with_right_and_below[i*im.shape[1]+j,2*im.shape[2]:3*im.shape[2]]=im[i+1,j,:]
            elif (j==(im.shape[1] -1)) and (i<(im.shape[0] -1)):
                im_with_right_and_below[i*im.shape[1]+j,0:im.shape[2]]=im[i,j,:]
                im_with_right_and_below[i*im.shape[1]+j,im.shape[2]:2*im.shape[2]]=im[i,j,:] #
                im_with_right_and_below[i*im.shape[1]+j,2*im.shape[2]:3*im.shape[2]]=im[i+1,j,:]
            elif (j<(im.shape[1] -1)) and (i==(im.shape[0] -1)):
                im_with_right_and_below[i*im.shape[1]+j,0:im.shape[2]]=im[i,j,:]
                im_with_right_and_below[i*im.shape[1]+j,im.shape[2]:2*im.shape[2]]=im[i,j+1,:]
                im_with_right_and_below[i*im.shape[1]+j,2*im.shape[2]:3*im.shape[2]]=im[i,j,:] #
            else:
                im_with_right_and_below[i*im.shape[1]+j,0:im.shape[2]]=im[i,j,:]
                im_with_right_and_below[i*im.shape[1]+j,im.shape[2]:2*im.shape[2]]=im[i,j,:] #
                im_with_right_and_below[i*im.shape[1]+j,2*im.shape[2]:3*im.shape[2]]=im[i,j,:] #

    #--------------------------------------------------------------------------------#
    # remove pixels with ignore label
    #--------------------------------------------------------------------------------#

    if ignore_label is None:
        #pxs = im.reshape((-1, im.shape[-1]))
        pxs = im_with_right_and_below
        lbs = np.ravel(gt)
    else:
        #pxs = im[gt != ignore_label]
        #lbs = gt[gt != ignore_label]
        lbs = np.ravel(gt)
        pxs = im_with_right_and_below
        index = lbs!= ignore_label
        lbs = lbs[index]
        pxs = pxs[index]
    #----------------------------------------------------------------------------------#
    # to avoid problems with segment numbering, rebuild it if ground truth is given
    #----------------------------------------------------------------------------------#
    if lbs is not None:
        lbs = rebuild_segment_numbering(lbs)

    ds = HSIDataset(pxs, lbs)

    dataloader = DataLoader(ds, batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(ds, batch_size=args['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    print('Initialized dataset and dataloader.')

    def hyper_opt(args):
        #--------------------------------------------------------------------------------#
        #calculate the initial time
        #--------------------------------------------------------------------------------#
        start_time = time.time()
        print('regularizer :', (args['regularizer'][0]))
        #------------------------------------------------------------------------------------#
        # init model
        #------------------------------------------------------------------------------------#
        vade = VaDE(input_dim=data_dim, z_dim=latent_dim, n_centroids=n_cl,
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500],
                one_by_delta_x=one_by_delta_x, regularization=args['regularizer'][0],
                directory=final_directory, ignore_label=ignore_label_inside)
        print('Initialized VaDE model.')

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
        y_pred = vade.fit(dataloader, valid_loader, lr=args['lr'], num_epochs=args['epochs'],
                          anneal=False, visualize=False)

        print('Save found parameters...')
        if args['save'] != "":
            save_location = args['save'].replace('.pt', f"_lam{args['regularizer'][0]}.pt")
            print(f"Save weights at {save_location}")
            vade.save_model(save_location)

        print('Save found segmentation...')
        # colour_map = plt.get_cmap("tab20")
        # cmap_tensor = np.zeros((n_cl+1,3))
        # for i in range(n_cl+1):
            # cmap_tensor[i] = np.array(colour_map(i)[0:3])

        seg = np.zeros(shape=im.shape[:-1], dtype=np.uint8)
        print("seg shape",seg.shape)
        if ignore_label is None:
            if use_mnf:
                filename = f'seg_{image_file}_ncl{n_cl}_ldim{latent_dim}_reg{args["regularizer"][0]}_mnf_all_pxs'
            else:
                filename = f'seg_{image_file}_ncl{n_cl}_ldim{latent_dim}_reg{args["regularizer"][0]}_all_pxs'
            seg = y_pred.reshape(im.shape[:-1])
            print(seg.shape)
            save_gt_colored_segmentation(f'{final_directory}/{filename}_full.png', seg, gt_file, ignore_label_inside,
                                         True, None, False, False)
            save_gt_colored_segmentation(f'{final_directory}/{filename}_full.png', seg, gt_file, ignore_label_inside,
                                         False, None, False, True)
            # saveColoredSegmentation(seg, f'{final_directory}/{filename}.png', cmap_tensor, ignore_label_inside,
                                    # True, gt)
            # saveColoredSegmentation('{}.png'.format(filename),
                                        #  seg, gt_file, None, True)
            saveArrayAsNetCDF(seg, f'{final_directory}/{filename}.nc')
        else:
            if use_mnf:
                filename = f'seg_{image_file}_ncl{n_cl}_ldim{latent_dim}_reg{args["regularizer"][0]}_mnf_ignore_pxs'
            else:
                filename = f'seg_{image_file}_ncl{n_cl}_ldim{latent_dim}_reg{args["regularizer"][0]}_ignore_pxs'
            seg[gt != ignore_label] = y_pred + 1
            saveColoredSegmentation(seg, f'{final_directory}/{filename}.png', cmap_tensor, ignore_label_inside,
                                    True, gt)
            # save_gt_colored_segmentation('{}.png'.format(filename),
                                        #  seg, gt_file, None, True)
            saveArrayAsNetCDF(seg, f'{final_directory}/{filename}.nc')

        #---------------------------------------------------------------------------------------#
        # evaluate segmentation
        #---------------------------------------------------------------------------------------#

        print('Evaluate segmentation...')
        score = segmentation_scores(seg, gt, ignore_label_inside)
        print(score)
        #---------------------------------------------------------------------------------------#
        #print the total time of execution
        #---------------------------------------------------------------------------------------#

        print(f"--- {(time.time() - start_time)} seconds ---")

        return -score['overallAcc']

    trials=Trials()
    best = fmin(fn=hyper_opt, space=args, algo=rand.suggest, max_evals=max_evals,
                rstate=np.random.default_rng(seed_hyp), trials=trials)

    print(best)

    df = []
    for it in range(max_evals):
        df_dict = {}
        df_dict["loss"]= -trials.results[it]['loss']
        for key, value in trials.vals.items():
            df_dict[key] = value[it]
        df.append(df_dict)

    df = pd.DataFrame(df)
    if os.path.exists(final_directory + "/hyp_Salinas.txt"):
        os.remove(final_directory + "/hyp_Salinas.txt")
    else:
        print("The file does not exist")
    df.to_csv(final_directory + '/hyp_Salinas.txt', sep=" ", index=False, mode='a')
    fig, ax = plt.subplots(len(df.columns) - 1)
    for i in range(1, len(df.columns)):
        if len(df.columns) - 1 == 1:
            ax.scatter(df[df.columns[i]], df['loss'])
            ax.set(xlabel=df.columns[i], ylabel='OA')
        else:
            ax[i - 1].scatter(df[df.columns[i]], df['loss'])
            ax[i - 1].set(xlabel=df.columns[i], ylabel='OA')
    
    fig.savefig(final_directory + '/loss vs regularizer')
    fig.show()

if __name__ == "__main__":
    main()
