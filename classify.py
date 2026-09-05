# coding: utf-8
import os
from train_vade_hyp import classify
from netCDF4 import Dataset
from msiplib.io import read_image

im = read_image(os.path.expandvars("${DATA_DIR}") + "/segmentation/IndianPines/IndianPines.nc")
gt_file = os.path.expandvars("${DATA_DIR}") + "/segmentation/IndianPines/IndianPines_gt.png"
gt = Dataset(os.path.expandvars("${DATA_DIR}") + "/segmentation/IndianPines/IndianPines.nc", "r")["groundtruth"]["segmentation_mask"][
    :
].data
weights = os.path.expandvars("${OUTPUT_DIR}") + "/segmentation/vade-tv/IndianPines/2022-6-9-8-44-52-epochs1000_ldim10_reg0_002/vade_params-epochs1000_ldim10_reg0_002.pt"
classify(os.path.expandvars("${OUTPUT_DIR}") + "/segmentation/vade-tv/IndianPines/classify", im, weights, gt, 16, gt_file, 0)
