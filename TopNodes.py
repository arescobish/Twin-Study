# Secons Section of Results
# For finding top nodes that are connected 


import numpy as np
import pandas as pd

def find_top_indices(matrix):
    row_sums = np.sum(matrix, axis=1)  # Calculate row sums
    num_rows = len(row_sums)
    
    top_indices_count = 20  # Calculate number of indices for top 20%
    sorted_indices = np.argsort(row_sums)[::-1]  # Sort indices based on row sums in descending order
    
    top_indices = sorted_indices[:top_indices_count]  # Get indices for top 20% rows
    return top_indices+1

# In the below code, X is obtained from code ..\Commonedges_in_bootstrapping.py

top_indices = find_top_indices(X)
print("Indices of Top 20% of Row Sums:", top_indices)

df_mni = pd.read_csv("..\shen268_MNIcoords.csv", header=0)
df_indices = df_mni[df_mni['NodeNo'].isin(top_indices)]

df_indices['NodeNo'] = pd.Categorical(df_indices['NodeNo'], categories=top_indices, ordered=True)
# Sort the DataFrame based on the categorical column
df_topnodes = df_indices.sort_values(by='NodeNo').reset_index(drop=True)
df_topnodes.to_csv(r'..\Top20nodes_PLSR_StdScaler_500_abs_g_MZ_112.csv', header = 1, index = False)
