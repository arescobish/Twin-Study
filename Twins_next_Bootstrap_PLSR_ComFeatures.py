# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 21:47:01 2023

@author: Administrator
"""
## First Section of Results
## Prediction of G-score within twins using functional connectomes
## FIND COMMON FEATURES MONOZYGOTIC AND DIZYGOTIC FUNCTIONAL CONNECTIVITY FROM THE ALL SUBJECTS C0NNECTIVITY FILE


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

from sklearn.metrics import explained_variance_score, r2_score
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.model_selection import GroupKFold, permutation_test_score, cross_validate
from sklearn.feature_selection import f_regression

start=datetime.now()
# Data Preparation
fc = pd.read_csv(r'..\vectorized_upper_278_twins.csv', header=None)
T = pd.read_csv(r'D:..\prepraredata_278_twins.csv', header= 0)
T1 = pd.read_csv(r'D:..\prepraredata_278_twins_MZ.csv', header= 0)

extracted_col = T["Subject"]
fc.insert(0, "s1", extracted_col)

d1 = fc[fc.s1.isin(T1.Subject.unique())]
d12 = d1.drop(d1.columns[0],axis=1)
fc = d12.values
T = T1

# Actual Start 
subj = T.Subject.values
group = T.Group.values

G = T.g.values
cog_metric = np.transpose(np.asarray([G]))

regr = PLSRegression()
paramGrid ={'plsr__n_components': [ i for i in range(1,101)],}
grid_pipe = Pipeline(steps = [('scaler', StandardScaler()),('plsr', PLSRegression())])

X = fc
Y = cog_metric

iter_count = 1000
n = int(group.shape[0]/2)
choices = ['12','21']
group_iterations = []

para_r = []
para_p = []

while len(group_iterations) < iter_count:
    
    rand = np.random.choice(choices,n)
    iter_lst = rand.tolist()
    iter_string = ''.join(iter_lst) 
    iter_str_lst = [str(i) for i in iter_string]
    if not iter_str_lst in group_iterations:
        group_iterations.append(iter_str_lst)
    else:
        pass

scores = []
perms_for_p = []
r2 = []
best_params = []
coeff = []
frozen_count = 1

group_kfold = GroupKFold(n_splits = 2)


common_features_all = []
output = {}

def SavePickle(infile, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(infile, f, protocol = 2)
        
for j, boot in enumerate(group_iterations):
    strap = np.array(boot)
    common_features = None
    print ('iteration: ' + str(frozen_count))
    print ('iteration: ' + str(frozen_count))
    
    for i, (train_index, test_index) in enumerate(group_kfold.split(X, Y, groups = strap)):
        print(f"Fold {i}:")
        x_tr, y_tr = X[train_index,:], Y[train_index]
        x_tt, y_tt = X[test_index], Y[test_index]
        print(f"  Train: index={train_index}, group={strap[train_index]}")
        print ("X.............")
        
        f_values, p_values = f_regression(x_tr, y_tr)
        p_value_threshold = 0.05
        
        # Select features based on the p-value threshold
        selected_indices = np.where(p_values < p_value_threshold)[0]

        if common_features is None:
            common_features = set(selected_indices)
        else:
            common_features = common_features.intersection(selected_indices)
            
        
        frozen_count = frozen_count + 1
    
    common_features = sorted(list(common_features))
    common_features_all.append(common_features)

output['selected_edges'] = common_features_all

SavePickle(output, f'..//g_MZ_112_CommonFeatures_1000bs.pkl')

print (datetime.now()-start)