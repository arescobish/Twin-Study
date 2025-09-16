## Finding Neural Similarity/differences from two vectorized functional connectivity

#import relevant libraries
import sys; sys.path
import pandas as pd
import numpy as np 
import os
#import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import scipy.stats
import pickle
from datetime import datetime

from sklearn.metrics import explained_variance_score, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

aa_MZ = pd.read_csv(r'..\Desktop\MZTwins.csv')
aa_MZ1 = pd.read_csv(r'..\vectorized_mergedFZ_MZDZ_278.csv', header = 0)
aa_list = aa_MZ[['S1','S2']].apply(tuple, axis = 1).tolist()

dist_dict = {}
for aa in aa_list:
    
    subj1 = aa[0]
    subj2 = aa[1]
    
    subj1_vec = np.asarray(aa_MZ1[aa_MZ1['Subject'] == subj1]).ravel()[1:]
    subj2_vec = np.asarray(aa_MZ1[aa_MZ1['Subject'] == subj2]).ravel()[1:]
    
    this_dist = np.abs(subj1_vec - subj2_vec)
    
    dist_dict[aa] = this_dist

dist_df = pd.DataFrame(dist_dict).transpose()

dist_df.to_csv(r'..\neural_similarity_differences_MZ.csv', index = False)
