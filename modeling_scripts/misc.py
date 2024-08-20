# System
import os

# Externals
import random
import pandas as pd

# Locals
import config



def partition(list_in, n):
    '''
    Randomly partition list of years into n train and validation sets for n-fold cross validation.
    '''
    random.seed(27106)
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def make_dirs(root_dir):
    '''
    Make directories.
    '''
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(root_dir+'train/swe_eval/aggregate_scheme/', exist_ok=True)
    os.makedirs(root_dir+'val/swe_eval/aggregate_scheme/', exist_ok=True)
    os.makedirs(root_dir+'train/swe_eval/climatological_scheme/', exist_ok=True)
    os.makedirs(root_dir+'val/swe_eval/climatological_scheme/', exist_ok=True)
    os.makedirs(root_dir+'train/swe_eval/site_year_scheme/train/', exist_ok=True)
    os.makedirs(root_dir+'val/swe_eval/site_year_scheme/val/', exist_ok=True)
    