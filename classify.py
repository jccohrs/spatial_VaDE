# coding: utf-8
from train_vade_hyp import classify
from netCDF4 import Dataset
from msiplib.io import read_image

im = read_image("/Users/jan/sciebo/data/segmentation/IndianPines/IndianPines.nc")
gt_file = "/Users/jan/sciebo/data/segmentation/IndianPines/IndianPines_gt.png"
gt = Dataset("/Users/jan/sciebo/data/segmentation/IndianPines/IndianPines.nc", "r")["groundtruth"]["segmentation_mask"][
    :
].data
weights = "/Users/jan/sciebo/output/segmentation/vade-tv/IndianPines/2022-6-9-8-44-52-epochs1000_ldim10_reg0_002/vade_params-epochs1000_ldim10_reg0_002.pt"
classify("/Users/jan/sciebo/output/segmentation/vade-tv/IndianPines/classify", im, weights, gt, 16, gt_file, 0)
