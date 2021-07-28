import numpy as np
import os,sys

Output_Data_Path = "/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/Data/100bp_12_7_Data_20_July_2020/" 
x = np.load(sys.argv[1])
y = np.mean(np.concatenate([x, np.asarray([0]*(4-x.shape[0]%4))]).reshape(-1, 4), axis=1) 
np.save(Output_Data_Path + sys.argv[1], y)
