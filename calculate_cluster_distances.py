from train_vade import *
import os
from msiplib.io import readArrayFromNetCDF
from msiplib.io.segmentation import load_seg_mask_from_image_file
from sklearn.metrics import pairwise_distances
from skimage.exposure import rescale_intensity
import numpy as np

def read_hsi():
    args = read_args()
    image_path = os.path.expandvars(f'$DATA_DIR/segmentation/{args["image_file"]}/{args["image_file"]}_gt.png')
    data_path = os.path.expandvars(f'$DATA_DIR/segmentation/{args["image_file"]}/{args["image_file"]}.nc')

    gt = load_seg_mask_from_image_file(image_path)
    im = readArrayFromNetCDF(data_path)
    im = rescale_intensity(im, out_range=(0.0, 1.0))
    return im, gt

def read_denoised():
    args = read_args()
    image_path = os.path.expandvars(f'$DATA_DIR/segmentation/{args["image_file"]}/{args["image_file"]}_gt.png')
    data_path = os.path.expandvars(f'$OUTPUT_DIR/segmentation/vade/pure/{args["image_file"]}/Denoised_input/denoised_image.nc')

    gt = load_seg_mask_from_image_file(image_path)
    im = readArrayFromNetCDF(data_path)
    return im, gt

def calculate_intra_cl_dist(im, gt):
    n_cl = np.max(gt)
    intra_cl_dist = np.zeros(n_cl+1)
    for i in range(0, n_cl+1):
        cl = im[gt == i]
        dist = pairwise_distances(cl, n_jobs=-1)
        intra_cl_dist[i] = np.max(dist)
    return intra_cl_dist

def calculate_inter_cl_dist(im, gt):
    n_cl = np.max(gt)
    inter_cl_dist = []
    for i in range(0, n_cl+1):
        cl = im[gt == i]
        inter_cl_dist.append(np.array(np.mean(cl, axis=0)))
    dist = pairwise_distances(inter_cl_dist, n_jobs=-1)
    # print(f'dist: {dist}')
    ans = 999999999
    for i in range(0, n_cl+1):
        for j in range(0, n_cl+1):
            if i is not j and ans > dist[i][j]:
                ans = dist[i][j]
    return ans

im, gt = read_hsi()
intracl_dist1 = calculate_intra_cl_dist(im, gt)
intercl_dist1 = calculate_inter_cl_dist(im, gt)
print(intracl_dist1,intercl_dist1)

im, gt = read_denoised()
intracl_dist2 = calculate_intra_cl_dist(im, gt)
intercl_dist2 = calculate_inter_cl_dist(im, gt)
print(intracl_dist2,intercl_dist2)
print(f'Difference between intracluster distances (raw) and (denoised): {intracl_dist1-intracl_dist2}')
# import pdb
# pdb.set_trace()
# print(intracl_dist)

# im, gt, _, args['gt_file'] = read_data(os.path.expandvars(data_path))
# args = prepare_data(im, gt, args, seed_worker, g)
# dataloader = args['dataloader']

# print(len(dataloader.dataset))
# for idx, (inputs, _) in enumerate(dataloader):
#     inputs = Variable(inputs)
#     print(inputs[0].shape)
#     # print(inputs.shape)

## intra-cluster distance
## for 1 to 16 number of clusters:
##    // assuming we have 16 classes and pixels are distributed evenly across classes, 
##    // we have 1300*1299 = 1,688,700 pairs to calculate distances
##    for (di) 1 to 1300 data points in a cluster (on average):
##      for (dj) 1 to 1299 data points:
##          // calculating L2 distance will take (for 1 to 200:)
##          distance = calculate_L2_distance(di-dj) 
##      