# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 04:10:57 2023

@author: Administrator
"""

## Plot Permutation distribution graph 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

corr = pd.read_csv('..\g_MZ_chance_scores.txt', header = None).values
corr_true = 0.31
pvalue = 8.19E-06

# Create a histogram of the data using seaborn
plt.figure(figsize=(8, 5))
sns.histplot(corr, bins=50, kde=False, color='green', edgecolor='black', alpha = 0.5, label='Freq Plot')

# Plot a vertical line for the true mean value
plt.axvline(x=corr_true, color='red', linestyle='dashed', linewidth=2, label='True Score')

# Add labels and legend
plt.xlabel('Similarity Index 'r'', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
#plt.title('Histogram with True Value', fontsize=12)
plt.legend()
plt.savefig('D:\HCP Behav data\Twins\Twin data\Results\PermutationPlot_MZ58.png', dpi=2000)
plt.show()

