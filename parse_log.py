#!/usr/bin/env python

"""
Parse training log

Evolved from parse_log.sh
"""

import re
import pandas as pd
import matplotlib.pyplot as plt


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
    for i in range(1, len(data.columns)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data[data.columns[0]], data[data.columns[i]])
        ax.plot(data[data.columns[0]], pd.Series(data[data.columns[i]]).rolling(11).mean())
        ax.set(xlabel=data.columns[0], ylabel=data.columns[i])

        fig.savefig(f'{path}-{data.columns[i].replace(" ","-")}.png')
