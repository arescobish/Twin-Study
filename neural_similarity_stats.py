## Results obtained


import pandas as pd
import pickle
import pandas as pd
import numpy as np
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

data_path = '../'

def LoadPickle(infile):
    with open(infile, 'rb') as f:
        outfile = pickle.load(f)
        return outfile
from scipy.stats import pearsonr, percentileofscore, norm

def pearsonr_ci(x,y,alpha=0.05):
    r, p = pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

true_model_output = LoadPickle(f'..//pred_model_StdScaler_500_diff_g_MZ.pkl')


coeff = true_model_output['coefs']
coeff_df = pd.DataFrame(coeff).transpose()

actual_vals = np.concatenate(true_model_output['actual_vals'])
pred_vals = np.concatenate(true_model_output['pred_vals'])
mode_pls_components = max(set(true_model_output['selected_n_components']), key=true_model_output['selected_n_components'].count)
true_rval, raw_pval, lo, hi = pearsonr_ci(actual_vals, pred_vals.ravel())

## Correlation Plot

new_vis = pd.DataFrame([actual_vals, pred_vals.flatten()]).transpose()
new_vis.columns = ['Actual-F4_mem', 'Predicted-F4_mem']

import pingouin as pg
pg.corr(x=new_vis['Actual-F4_mem'], y=new_vis['Predicted-F4_mem'])

import seaborn as sns
sns.set_theme(style="darkgrid")

g = sns.jointplot(x="Actual-F4_mem", y="Predicted-F4_mem", data=new_vis,
                  kind="reg", truncate=False,
                  xlim=(-0.5, 2), ylim=(0.5, 0.9),
                  color="m", height=7)   


