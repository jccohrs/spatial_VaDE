import numpy 
import os
import matplotlib.pyplot as plt

def collect_last_line(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1].strip()
        numbers = last_line.split(',')
        return [float(num) for num in numbers]

def collect_statistics(directory):
    experiment_number = []
    valid_loss = []
    accuracy = []
    for root, dirs, files in os.walk(directory):
        for idx, folder in enumerate(dirs):
            folder_path = os.path.join(root, folder)
            file_path = os.path.join(folder_path, 'exp_stats.csv')
            if os.path.isfile(file_path):
                numbers = collect_last_line(file_path)
                experiment_number.append(idx)
                valid_loss.append(numbers[2])
                accuracy.append(numbers[3])
    print(experiment_number)
    print(valid_loss)
    return 


# Specify the directory to search for folders
directory = '/home/chagnaa/output/segmentation/vade/pure/IndianPines/Study_reg0_spat1_pretrain0'
collect_statistics(directory)




