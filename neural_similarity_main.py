
## Second section of Results
## Predicting G-score differences using functional connectomes differences in twin pairs

# Import Packages

import pickle
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from scipy.stats import pearsonr, percentileofscore, norm

start=datetime.now()

def SavePickle(infile, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(infile, f, protocol = 2)

def exclude_cohabitants(df):
    df = df[~(df['phys_dist'] == 0)]
    df = df.reset_index()
    return df

def run_plsr(df, G, analysis):
    roi_cols = [i for i in list(df.columns) if 'edge' in i]

    X_data = df
    y_data = G

    output = {}
    coefs = []
    actual_vals = []
    pred_vals = []
    selected_n_components = []
    selected_edges = []
    pls_components = []

    param_grid = {'plsr__n_components': [i for i in range(1, 101)]}
    grid_pipe = Pipeline(steps = [('scaler', StandardScaler()),
                        ('filter', SelectKBest(f_regression, k = 500)),
                        ('plsr', PLSRegression())])
    kf = KFold(n_splits = 10)
    cvsplit = kf.split(X_data)
    grid = GridSearchCV(grid_pipe, param_grid = param_grid, cv = 10, n_jobs = 1, verbose=0)

    for train, test in cvsplit:
        train_X = X_data.values[train]
        train_y = y_data[train]

        test_X = X_data.values[test]
        test_y = y_data[test]

        grid.fit(train_X, train_y)
        predicted_y = grid.best_estimator_.predict(test_X)
        selected_n_components.append(grid.best_params_['plsr__n_components'])
        coefs.append(grid.best_estimator_.named_steps['plsr'].coef_[:,0])
        selected_edges.append(X_data.columns[grid.best_estimator_.named_steps['filter'].get_support(indices=True)].tolist())
        pred_vals.append(predicted_y)
        actual_vals.append(test_y)

    output['actual_vals'] = actual_vals
    output['pred_vals'] = pred_vals
    output['coefs'] = coefs
    output['selected_n_components'] = selected_n_components
    output['selected_edges'] = selected_edges
    output['pls_components'] = pls_components
    
    return output
 
analysis = 'g';

# Input G-score values
aa_MZ = pd.read_csv(r'..\\DZPairs_random.csv')
# Input Neural similaity differences functional connectivity data
df = pd.read_csv(r'..\\neural_similarity_differences.csv', header = 0)
G = abs(aa_MZ.diff_g.values)

# Run PLSR function
out = run_plsr(df, G, analysis)

print (datetime.now()-start)



