## Different ways to find commonedges in first section of analysis


# Import packages

import pickle
import pandas as pd
import numpy as np
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import random

systems_Index1 = pd.read_csv(r'..\ROIs_neworks2.csv', header = None, index_col=0 )
systems_Index1.rename(columns={1: 'name'}, inplace=True)
systems_flat = systems_Index1.values.flatten()

def LoadPickle(infile):
    with open(infile, 'rb') as f:
        outfile = pickle.load(f)
        return outfile

true_model = LoadPickle(r'..\\g_MZ_112_CommonFeatures.pkl')
true_model_bs = LoadPickle(r'..\\g_MZ_112_CommonFeatures_1000bs.pkl')

edgeSele = true_model['selected_edges']
edgeSel = true_model_bs['selected_edges']

from collections import Counter

# Flatten the list of lists into a single list
flat_list = [item for sublist in edgeSel for item in sublist]

# Use Counter to count the occurrences of each element
element_counts = Counter(flat_list)
counts = {}
    
for element in element_counts:
    counts[element] = element_counts[element]

data_rows1 = []
data_rows2 = []
for key, values in element_counts.items():
        data_rows1.append((key))
        data_rows2.append(values)
df = pd.DataFrame({'number':data_rows1, 'strength':data_rows2})

edges = [str(i) for i in range(1, 35779)]
edge2systems_dict = {k: v for k, v in zip(edges, systems_flat)}

for key, value in edge2systems_dict.items():
    first = value.split('|')[0]
    second = value.split('|')[1]
    if first == second:
        continue
    else:
        alpha_sorted = sorted([first, second])
        edge2systems_dict[key] = f'{alpha_sorted[0]}|{alpha_sorted[1]}'
# for unique pairs 

x = pd.DataFrame(list(edge2systems_dict.items()), columns=['Key', 'name']).drop(columns=['Key'])

# Replace non-matching values with zeros
x['strength'] = x.index.map(df.set_index('number')['strength']).fillna(0)

## values greater than 800 are set to 1, and other values are set to 0
x['strength'] = (x['strength'] > 800).astype(int)

x.set_index(x.columns[0], inplace=True)
systems_Index1_tran = x.T

system_pairs_unique = set(list(edge2systems_dict.values()))

# Get a list of within-system pair names
within_systems = []
for sys_pair in system_pairs_unique:
    sys1 = sys_pair.split('|')[0]
    sys2 = sys_pair.split('|')[1]
    if sys1 == sys2:
        within_systems.append(sys_pair)
        
# Get a list of between-system pair names
between_systems = [i for i in system_pairs_unique if not i in within_systems]

within = systems_Index1_tran[within_systems].groupby(level=0, axis=1).sum()
between = systems_Index1_tran[between_systems].groupby(level=0, axis=1).sum()

wb_df1 = between.join(within, how = 'outer').transpose()

wb_df1.insert(0,"Networks",wb_df1.index, True)
wb_df1[['Net1','Net2']] = wb_df1.Networks.str.split("|", 1, expand= True )
wb_df1.rename(columns = {list(wb_df1)[1]:'value'}, inplace = True)

new_wb_df1 = pd.DataFrame([wb_df1.Net1.values, wb_df1.Net2.values, wb_df1.value.values]).transpose()
new_wb_df1.columns = ['Net1', 'Net2', 'value']
## Make edge-edge-value format to insert in OriginLabPro 
new_wb_df1.to_csv(r'..\OriginPro_ChordNetworks_MZ_morethan800.csv', index= False)

## make symmetric matrix to pass the values to matlab code to visualize Chord diagram 
import numpy as np
s = new_wb_df1.pivot(*new_wb_df1)
s1 = new_wb_df1.pivot(*new_wb_df1)
np.fill_diagonal(s1.values,0)
ret = s.add(s1.T, fill_value=0)
ret.to_csv(r'..\ChordNetworks_MZ_morethan800.csv', index= True)

## Make output format such that the file fits into Bioimage connviewer
system = x.values.flatten()
tri = np.zeros((268, 268))
idx, col = np.triu_indices(268, k=1)
tri[idx, col] = system
X = tri + tri.T - np.diag(np.diag(tri))

pd.DataFrame(X).to_csv(r'..\BioimageOut_PLSR_StdScaler_MZ_morethan800.csv', header = None, index = False)





