# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:07:12 2023

@author: Administrator
"""

## First Section of Results
## Prediction of G-score within twins using functional connectomes
## MONOZYGOTIC AND DIZYGOTIC FUNCTIONAL CONNECTIVITY FROM THE ALL SUBJECTS C0NNECTIVITY FILE (i.e. vectorized_upper_278_twins) 


#import relevant libraries
import sys; sys.path
import pandas as pd
import numpy as np 
import os
#import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import scipy.stats

from sklearn.metrics import explained_variance_score, r2_score
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.model_selection import GroupKFold, permutation_test_score, cross_validate

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

# Input subjects
subj = T.Subject.values
# Input Groups (one group for each twin pairs)
group = T.iloc[:,19].values

# G-score values
G = T.g.values

cognition = ['g']
cog_metric = np.transpose(np.asarray([G]))

regr = PLSRegression()
paramGrid ={'plsr__n_components': [ i for i in range(1,101)],}
grid_pipe = Pipeline(steps = [('scaler', StandardScaler()),('plsr', PLSRegression())])
n_cog = np.size(cognition)
X = fc
Y = cog_metric
perm = 1
#correlation between true and predicted (aka prediction accuracy)
corr = np.zeros([perm,n_cog])


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
best_params = []
coeff = []
frozen_count = 1

group_kfold = GroupKFold(n_splits = 2)

for j, boot in enumerate(group_iterations):
    strap = np.array(boot)
    best_para = [] 
    print ('iteration: ' + str(frozen_count))
    print ('iteration: ' + str(frozen_count))
    
    for i, (train_index, test_index) in enumerate(group_kfold.split(X, Y, groups = strap)):
        print(f"Fold {i}:")
        x_tr, y_tr = X[train_index,:], Y[train_index]
        x_tt, y_tt = X[test_index], Y[test_index]
        print(f"  Train: index={train_index}, group={strap[train_index]}")
        print ("X.............")
        
        # Initialize GridSearchCV with Linear Regression, an empty hyperparameter grid, and the Pearson correlation scorer
        gridSearch = GridSearchCV(grid_pipe, param_grid=paramGrid, n_jobs = 1)           
        gridSearch.fit(x_tr, y_tr, groups= strap[train_index]) 
        # Get the best model from GridSearch
        best_model = gridSearch.best_estimator_
        best_n_comp = gridSearch.best_params_['plsr__n_components']
        
        [r,p] = scipy.stats.pearsonr(best_model.predict(x_tt).ravel(), y_tt.ravel())
        para_r.append(r)
        para_p.append(p)
        best_para.append(best_n_comp)
        best_params.append(best_n_comp)
        
        frozen_count = frozen_count + 1
     
    # For finding optimum alpha parameter
    opt_para_regr2 =  best_para[0]
    def pearson_corr_score(y_true, y_pred):
            corr, _ = pearsonr(y_true, y_pred)
            return corr
    # Make the custom scoring function usable with GridSearchCV
    pearson_scorer = make_scorer(pearson_corr_score)
        
    regr_2 = PLSRegression(n_components = opt_para_regr2, scale = True)
    score_true, perms, pval_true = permutation_test_score(regr_2, X, Y.flatten(), groups=group, scoring = pearson_scorer, cv=group_kfold, n_permutations=100,)
    perms_for_p.append(perms)


# For finding p_value between true correlation and chances of correlation (from correlation coeff)
chance_scores = np.concatenate(perms_for_p)
score = np.mean(para_r)
C = np.sum(chance_scores > score)
n_perms = chance_scores.size
pvalue = (C+1)/float(n_perms + 1)

np.savetxt('..\\g_DZ_corr.txt', para_r, delimiter=',')
np.savetxt('..\\g_DZ_pvalue.txt', para_p, delimiter=',')
np.savetxt('..\\g_DZ_best_params.txt', best_params, delimiter=',')

np.savetxt('..\\g_DZ_chance_scores.txt', chance_scores, delimiter=',')


with open("..\\g_DZ_parameters.txt",'a') as file:
    file.writelines("meanScore = " +repr(str(score))+ "\n" +"pvalue = " +repr(str(pvalue)))
    #file.writelines(str(opt_para_regr2))
file.close

print (datetime.now()-start)