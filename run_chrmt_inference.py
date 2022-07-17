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
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
import sys
# sys.path.insert(1, '../../Code/deepENCODE/')
from chrmt_inference import perturb_x
from tensorflow.keras import backend as K


if __name__ == '__main__':

    RESOLUTION = 25
    window_size = 401

    ALL_CELL_TYPES = ["T{0:0=2d}".format(x) for x in [1,2,3,4,5,6,7,8,9,10,11,12,13]]
    assays = ['H3K36me3', 'H3K27me3', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K9me3', 'MNase']

    model_type_rename_dict = {"linear":"ridge", "maxpool":"cnn"}

    parser = argparse.ArgumentParser()
    parser.add_argument('--inserted_scalar')
    parser.add_argument('--maximum_inserted_minuslog10_p_value')
    parser.add_argument('--gaussian_bandwidth')
    parser.add_argument('--nucleosome_lambda')
    parser.add_argument('--output_prefix')
    parser.add_argument('--H3K27me3_flag', type=bool)
    parser.add_argument('--dCas9_binding_flag', type=bool)
    parser.add_argument('--MNase_scaling_flag', type=bool)
    parser.add_argument('--maximum_flag', type=bool)
    args = parser.parse_args()

    output_file_name = "../../Logs/" + args.output_prefix + "_" + args.inserted_scalar + "_" + args.maximum_inserted_minuslog10_p_value + "_" + args.gaussian_bandwidth + "_" + args.nucleosome_lambda + "_" + str(args.H3K27me3_flag) + "_" + str(args.dCas9_binding_flag) + "_" + str(args.MNase_scaling_flag) + "_" + str(args.maximum_flag) + ".csv"

    # First read Alan's data from the Google spreadsheet
    sheet_name = "p300_Alan_Cabrera_epigenome_editing_dataset"
    sheet_ID = "1rgbxJHon8Z6-Ke9hAK_uUzuR27SIZx8J5cZyz_jN2a8"
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df_p300 = pd.read_csv(sheet_url)

    # Load Alan's data into a dict
    p300_dict = {}

    for gene in list(set(list(df_p300["p300 target gene"]))):
        p300_dict[gene] = {}
        for i in range(len(df_p300)):
            if(df_p300.iloc[i, 0] == gene):
                transcript = df_p300.iloc[i, 1]
                gRNA_ID = df_p300.iloc[i, 5]
                gRNA_bin_wrt_tss = int(df_p300.iloc[i, 14]) // RESOLUTION            
                measured_fold_change = float(df_p300.iloc[i, 6])
                
                # Only consider bins within +-250bp of the TSS ########################################## CAREFUL
                if(abs(gRNA_bin_wrt_tss) > 10):
                    continue
                
                if((transcript, gRNA_ID, gRNA_bin_wrt_tss) in p300_dict[gene]):
                    p300_dict[gene][(transcript, gRNA_ID, gRNA_bin_wrt_tss)].append(measured_fold_change)
                else:
                    p300_dict[gene][(transcript, gRNA_ID, gRNA_bin_wrt_tss)] = [measured_fold_change]            

    # Initialize some dicts
    transcript_list = [("ENST00000241393.3", "CXCR4"),
                       ("ENST00000374994.8", "TGFBR1"),
                       ("ENST00000380392.3", "C2CD4B"),
                       ("ENST00000221972.7", "CD79A"),
                       ("ENST00000369887.3", "CYP17A1"),
                       ("ENST00000258787.11", "MYO1G"),
                       ("ENST00000296498.3", "PRSS12"),
                       ("ENST00000322002.4", "SOX11")]


    MNase_dict = {}
    for transcript_index in np.arange(len(transcript_list)):

        transcript = transcript_list[transcript_index][0]
        gene = transcript_list[transcript_index][1]
        epigenetic_features = np.load("../../Data/saved_npy_arrays/." + transcript + ".CT_"+str(-1)+".npy")
    
        MNase = np.expm1(epigenetic_features[0, :, -1]) # after the expm1, the MNase is now coverage
    
        min_MNase = np.min(MNase)
        max_MNase = np.max(MNase)
    
        MNase_dict[gene] = (min_MNase, max_MNase)


    num_layers_dict = {9:2, 13:2, 17:2, 21:2, 25:2,
                       31:3, 41:3, 51:3, 61:3, 71:3,
                       81:4, 101:4, 121:4, 141:4, 161:4, 181:4, 201:4, 221:4, 241:4,
                       261:5, 281:5, 301:5, 321:5, 341:5, 361:5, 381:5, 401:5}

    ise_mean_fold_change_dict = {}
    ise_stdev_fold_change_dict = {}

    results_list = []

    # Then, for a particular transcript, we compute the ISE scores mean-ed across many models and compare with Alan's data
    for transcript_index in range(len(transcript_list)):
        
        transcript = transcript_list[transcript_index][0]
        gene = transcript_list[transcript_index][1]

        inserted_scalar = float(args.inserted_scalar) 
        maximum_inserted_minuslog10_p_value = float(args.maximum_inserted_minuslog10_p_value)

        inserted_peak_width = 10
        gaussian_bandwidth = float(args.gaussian_bandwidth)

        nucleosome_lambda = float(args.nucleosome_lambda)

        stranded_MNase_offset = 0

        ise_radius = 10
        bin_wrt_tss_choices = list(range(-ise_radius, ise_radius)) # region where we perform ISE

        epigenetic_features = np.load("../../Data/saved_npy_arrays/." + transcript + ".CT_"+str(-1)+".npy")
        TPM = (np.power(10, np.load("../../Data/saved_npy_arrays/." + transcript + ".CT_"+str(-1)+".TPM.npy")) - 1)[0][0]    

        min_MNase = MNase_dict[gene][0]
        max_MNase = MNase_dict[gene][1]

        for model_type in ["maxpool"]:

            for context_size in [401]: #81, 161, 241, 321, 401]:
                half_window = context_size // 2

                num_layers = num_layers_dict[context_size]

                fold_change_dict = {}

                for replicate in tqdm(range(1, 101)):

                    K.clear_session() # 2022 and what an insane bug in Keras! Time to switch to PyTorch
                    trained_model = load_model("../../Models/manuscript.R13.stochastic_chromosomes.replicate_"+str(replicate)+"_0_transcriptome_"+str(context_size)+"_"+str(num_layers)+"_32_"+str(model_type)+"_mse_1_-1.hdf5", compile=False)    

                    key = (inserted_scalar, 
                           maximum_inserted_minuslog10_p_value,
                           gaussian_bandwidth,
                           nucleosome_lambda)

                    fold_change_list = []

                    for bin_wrt_tss in bin_wrt_tss_choices:

                        perturbed_epigenetic_features = perturb_x(epigenetic_features, bin_wrt_tss, inserted_peak_width, inserted_scalar, stranded_MNase_offset, maximum_inserted_minuslog10_p_value, gaussian_bandwidth, nucleosome_lambda, min_MNase, max_MNase, args.H3K27me3_flag, args.dCas9_binding_flag, args.MNase_scaling_flag, args.maximum_flag)

                        predicted_log10p1_TPM = trained_model.predict(perturbed_epigenetic_features[:, 200-half_window:200+half_window+1, :-1]) # This is 200 because of the size of the loaded epigenetic features numpy array

                        predicted_TPM = (np.power(10, predicted_log10p1_TPM) - 1)[0][0]

                        predicted_native_TPM = np.power(10, trained_model.predict(epigenetic_features[:, 200-half_window:200+half_window+1, :-1])[0][0]) -1 # Same reason for 200, as above

                        fold_change_list.append(predicted_TPM / predicted_native_TPM)

                    if(key in fold_change_dict):
                        fold_change_dict[key].append(fold_change_list)
                    else:
                        fold_change_dict[key] = [fold_change_list]

                mean_fold_change_list = np.mean(fold_change_dict[key], axis=0)
                stdev_fold_change_list = np.std(fold_change_dict[key], axis=0)

            ise_mean_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)] = mean_fold_change_list
            ise_stdev_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)] = stdev_fold_change_list

            # Now we compare the model predictions w.r.t Alan's data
            scatter_dict = {}                
            scatter_dict[gene] = {}
            scatter_dict[gene][(model_type_rename_dict[model_type], context_size)] = []

            for (transcript, gRNA, pos) in p300_dict[gene].keys():

                # If the gRNA_bin_wrt_TSS which is named pos, is not within the ise_radius, then ignore
                if(abs(pos) > ise_radius):
                    print(ise_radius, pos, file=sys.stderr)
                    continue

                fold_change_list = p300_dict[gene][(transcript, gRNA, pos)]
                mean_fold_change = np.mean(fold_change_list)
                stdev_fold_change = np.std(fold_change_list)

                scatter_dict[gene][(model_type_rename_dict[model_type], context_size)].append((mean_fold_change, ise_mean_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)][ise_radius + pos]))

            # mean_predictions = ise_mean_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)]
            # stdev_predictions = ise_stdev_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)]

            scatter_x = [x[0] for x in scatter_dict[gene][(model_type_rename_dict[model_type], context_size)]]
            scatter_y = [x[1] for x in scatter_dict[gene][(model_type_rename_dict[model_type], context_size)]]

            sc, sp = spearmanr(scatter_x, scatter_y)
            results_list.append([transcript, gene, model_type_rename_dict[model_type], context_size, round(key[0], 4), round(key[1], 4), round(key[2], 4), round(key[3], 4), round(sc, 3), round(sp, 3)])

    with open(output_file_name, 'w') as f_output:
        for result in results_list:
            f_output.write(f"{result}\n")

