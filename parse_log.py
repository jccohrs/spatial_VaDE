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
    averaging_width = 11
    # plot train loss
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Training loss')
    ax.plot(data['epoch'], data['train loss'], label='Raw')
    ax.plot(data['epoch'], pd.Series(data['train loss']).rolling(averaging_width).mean(),
            label=f'Smoothed (width {averaging_width})')
    ax.set_ylim(bottom=-1200, top=0)
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
    ax.set_ylim(bottom=-1200, top=0)
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


def main():
    d = parse_log('/Users/jan/sciebo/output/segmentation/vade-tv/SalinasA_0_0_025/2022-6-7-14-58-20-ldim10_reg0/output-ldim10_reg0.log')
    plot_stats('/Users/jan/Desktop/plot_stats', d)

if __name__ == "__main__":
    main()
