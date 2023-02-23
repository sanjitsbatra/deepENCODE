import sys, os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import random
import itertools
import warnings
import argparse
from tqdm import tqdm

from tensorflow.keras.models import load_model
import sys
from chrmt_inference import perturb_x
from tensorflow.keras import backend as K
  

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--transcript')
    args = parser.parse_args()

    cell_types = ["T{0:0=2d}".format(x) for x in [1,2,4,5,6,7,8,9,10,11,12,13]]
    special_cell_types = [cell_types[0], cell_types[1], cell_types[3]]

    assays = ['H3K36me3', 'H3K27me3', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K9me3', 'MNase']
    max_dict = {'H3K36me3':2, 'H3K27me3':10, 'H3K27ac':100, 'H3K4me1':10, 'H3K4me3':200, 'H3K9me3':3}

    num_layers_dict = {9:2, 13:2, 17:2, 21:2, 25:2,
                       31:3, 41:3, 51:3, 61:3, 71:3,
                       81:4, 101:4, 121:4, 141:4, 161:4, 181:4,
                       201:4, 221:4, 241:4, 261:5, 281:5, 301:5,
                       321:5, 341:5, 361:5, 381:5, 401:5}

    fold_change_dict = {}

    for idx, perturbed_assay_index in enumerate(range(6)):

        cell_type_index_dict = {0:-13, 1:-12, 2:-9}
        
        for jjj, cell_type in enumerate(special_cell_types):
            
            for iiii, model_type in enumerate(["maxpool"]):

                inserted_peak_width = 6
                gaussian_bandwidth = None
                inserted_scalar = 2500 
                maximum_inserted_minuslog10_p_value = max_dict[assays[perturbed_assay_index]]

                nucleosome_lambda = None
                stranded_MNase_offset = 0

                ise_radius = 40
                bin_wrt_tss_choices = list(range(-ise_radius, ise_radius)) # region where we perform ISE

                epigenetic_features = np.load("../../Data/saved_npy_arrays/." + args.transcript + ".CT_"+str(cell_type_index_dict[jjj])+".npy")
                TPM = (np.power(10, np.load("../../Data/saved_npy_arrays/." + args.transcript + ".CT_"+str(cell_type_index_dict[jjj])+".TPM.npy")) - 1)[0][0]    

                for context_size in [401]: #  81, 161, 241, 321, list(num_layers_dict.keys())[16:]:
                    half_window = context_size // 2

                    num_layers = num_layers_dict[context_size]

                    native_TPM_list = []

                    for replicate in tqdm(range(1, 101, 1), position=0, leave=True):

                        K.clear_session() # 2022 and what an insane bug in Keras! Time to switch to PyTorch
                        trained_model = load_model("../../Models/manuscript.R13.stochastic_chromosomes.replicate_"+str(replicate)+"_0_transcriptome_"+str(context_size)+"_"+str(num_layers)+"_32_"+str(model_type)+"_mse_1_-1.hdf5", compile=False)

                        key = (cell_type, perturbed_assay_index)

                        fold_change_list = []

                        predicted_native_TPM = np.power(10, trained_model.predict(epigenetic_features[:, 200-half_window:200+half_window+1, :-1])[0][0]) -1 # This is 200 because of the size of the loaded epigenetic features numpy array

                        for bin_wrt_tss in bin_wrt_tss_choices:
                            perturbed_epigenetic_features = perturb_x(epigenetic_features, bin_wrt_tss, inserted_peak_width, inserted_scalar, stranded_MNase_offset, maximum_inserted_minuslog10_p_value, gaussian_bandwidth, nucleosome_lambda, 0, 1, "False", "False", "False", "True", perturbed_assay_index)
                            predicted_log10p1_TPM = trained_model.predict(perturbed_epigenetic_features[:, 200-half_window:200+half_window+1, :-1])[0][0] # Same reason for 200, as above
                            predicted_TPM = np.power(10, predicted_log10p1_TPM) - 1
                            fold_change_list.append(predicted_TPM / predicted_native_TPM)

                            if(key in fold_change_dict):
                                fold_change_dict[key].append(fold_change_list)
                            else:
                                fold_change_dict[key] = [fold_change_list]

    for key in fold_change_dict.keys():

        mean_fold_change_list = np.mean(fold_change_dict[key], axis=0)

        (cell_type, perturbed_assay_index) = key

        np.save("../../Logs/" + args.transcript + "." + cell_type + "." + str(perturbed_assay_index) + ".npy", mean_fold_change_list)

