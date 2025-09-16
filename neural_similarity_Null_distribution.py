
## Saving results of 5000 permutations to establish a null distribution 


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
import pickle
import pandas as pd
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cross_decomposition import PLSRegression

from sklearn.metrics import explained_variance_score, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from scipy.stats import linregress

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def LoadPickle(infile):
    with open(infile, 'rb') as f:
        outfile = pickle.load(f)
        return outfile
    
def run_plsr(df, n_components):

    X_data = df
    y_data = G

    actual_vals = []
    pred_vals = []

    pipe = Pipeline(steps = [('scaler', StandardScaler()),
                        ('filter', SelectKBest(f_regression, k = 400)),
                        ('plsr', PLSRegression(n_components = n_components))])
    kf = KFold(n_splits = 10, shuffle = False)
    cvsplit = kf.split(X_data)

    for train, test in cvsplit:
        train_X = X_data.values[train]
        train_y = y_data[train]

        test_X = X_data.values[test]
        test_y = y_data[test]

        pipe.fit(train_X, train_y)

        pred_vals.append(np.concatenate(pipe.predict(test_X).reshape(1,-1)))
        actual_vals.append(test_y)

    results = list(linregress(np.concatenate(actual_vals), np.concatenate(pred_vals, axis = 0)))
    output = []
    output.append(results)
    output_df = pd.DataFrame(output, columns = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr'])

    if not os.path.exists(f'..\\permutations_MZ_58_700_5000.csv'):
        output_df.to_csv(f'..\\permutations_MZ_58_700_5000.csv', index = False)
    else:
        output_df.to_csv(f'..\\permutations_MZ_58_700_5000.csv', mode = 'a', index = False, header = False)

true_model_output = LoadPickle(r'..\\pred_model_StdScaler_700_diff_g_MZ_58.pkl')
n_components = max(set(true_model_output['selected_n_components']), key=true_model_output['selected_n_components'].count)

dirName = r'..\Neural_similarity_permutation_58';
# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)

aa_MZ = pd.read_csv(f'..\\MZPairs_random_58.csv')
G = abs(aa_MZ.diff_g.values)

for i, file in enumerate(listOfFiles):
    
    print("No of Permutation is: {}".format(i))
    
    df = pd.read_csv(file)
    
    run_plsr(df, n_components)