## Data generated for estabilishing null distribution; 5000 permuted data


#import relevant libraries
import sys; sys.path
import pandas as pd
import numpy as np 
import os
#import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import scipy.stats
import random

from sklearn.metrics import explained_variance_score, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

aa_MZ = pd.read_csv(r'..\MZPairs_random.csv')

aa_MZ1 = pd.read_csv(r'..\vectorized_upper_RoiSub_278_MZ.csv', header = 0)

fmri_subjects = aa_MZ['Subject2'].tolist()
iteration = 5000

for itre in range (iteration):
    
    print(itre)
    
    dist_dict = {}
    shuffled_ids = random.sample(fmri_subjects, len(fmri_subjects))
    aa_MZ['Subject2'] = shuffled_ids
    aa_list = aa_MZ[['Subject1','Subject2']].apply(tuple, axis = 1).tolist()
    
    for aa in aa_list:
    
        subj1 = aa[0]
        subj2 = aa[1]
    
        subj1_vec = np.asarray(aa_MZ1[aa_MZ1['Subject'] == subj1]).ravel()[1:]
        subj2_vec = np.asarray(aa_MZ1[aa_MZ1['Subject'] == subj2]).ravel()[1:]
    
        this_dist = np.abs(subj1_vec - subj2_vec)
    
        dist_dict[aa] = -1 * this_dist

    dist_df = pd.DataFrame(dist_dict).transpose()

    dist_df.to_csv(f'..\\neural_similarity_differences_ROI_MZ' + str(itre+1) + '.csv', index = False)