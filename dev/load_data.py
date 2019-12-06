""" load_data
Parse data from ../eval/all_data_samples.csv and creates a table for every fold.
"""
import os
import numpy as np
import pandas as pd

def get_fold(filename, fold):
    lines_in_fold = []
    df = pd.read_csv(filename, delimiter = ';')
    lines_in_fold = df[df.columns[fold-1]]
    return lines_in_fold

def groundtruth_in_array(filename):
    df = pd.read_csv(filename, delimiter = ';')
    return df
        
def get_data(SELECT_SET, FOLD):
    GT_FOLDS = os.getcwd()[:-len('dev')] + 'eval/cv_' + SELECT_SET + '_sample_ids.csv'
    lines_in_fold = get_fold(GT_FOLDS, FOLD)

    ALL_DATA = os.getcwd()[:-len('dev')] + 'eval/all_data_samples.csv'
    df = groundtruth_in_array(ALL_DATA)

    data_from_fold = df.iloc[lines_in_fold]
    return data_from_fold

