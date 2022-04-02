import os
import sys
import seaborn as sns
import numpy as np
import pandas as pd
from chrmt_generator import TranscriptomeGenerator, TranscriptomePredictor
from chrmt_train import maximum_likelihood_loss
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random
import itertools
import warnings
warnings.filterwarnings('ignore')
from chrmt_generator import RESOLUTION, EPS
import argparse
from tqdm import tqdm


# Perform in silico epi-mutagenesis
def ise(trained_model, x, y, bin_wrt_tss, inserted_peak_width, inserted_lnp1_minuslog10_p_value):

    x_modified = np.copy(x)
    for p in range(bin_wrt_tss - inserted_peak_width // 2, 
                   bin_wrt_tss + inserted_peak_width // 2 + 1):
        if( (p >= 0) and (p < x.shape[1]) ):
            # Modify the H3K27ac peak NOTE: this is ln( -log10(transformed p-value) + 1)
            x_modified[:, p, 3] += (x_modified[:, p, -1] * inserted_lnp1_minuslog10_p_value)
   
    yPred = trained_model.predict(x[:, :, :-1])[0][0]
    yPred_perturbed = trained_model.predict(x_modified[:, :, :-1])[0][0]
    
    model_prediction_fold_change = (np.power(10, yPred_perturbed) - 1) / (np.power(10, yPred + EPS) - 1) 
    
    return model_prediction_fold_change


# Transform p-value back from log-space
def p_value_mapping(inserted_lnp1_minuslog10_p_value):
    minuslog10_p_value = np.expm1(inserted_lnp1_minuslog10_p_value)
    p_value = np.power(10, -1 * minuslog10_p_value)
    return round(minuslog10_p_value, 4)


# Visualize true vs predicted fold change
def visualize_fold_change(axis_dict, ise_results):

    yTrue_gRNA = {}
    yPred_gRNA = {}
    gene_list = ['CXCR4', 'TGFBR1']    
    for gene in gene_list:
        yTrue_gRNA[gene] = {}
        yPred_gRNA[gene] = {}

    for e in ise_results:
        
        gene, peak_width, inserted_lnp1_minuslog10_p_value, gRNA_ID, bin_wrt_tss, CRISPRa_qPCR_fold_change, model_prediction_fold_change = e        
        
        if(gRNA_ID in yTrue_gRNA[gene]):
            yTrue_gRNA[gene][gRNA_ID].append(CRISPRa_qPCR_fold_change)
            yPred_gRNA[gene][gRNA_ID].append(model_prediction_fold_change)
        else:
            yTrue_gRNA[gene][gRNA_ID] = [CRISPRa_qPCR_fold_change]
            yPred_gRNA[gene][gRNA_ID] = [model_prediction_fold_change]
    
    yTrue_mean = {}
    yPred_mean = {}
    for gene in gene_list:
        yTrue_mean[gene] = [np.mean(yTrue_gRNA[gene][gRNA_ID]) for gRNA_ID in list(yTrue_gRNA[gene].keys())]
        yPred_mean[gene] = [np.mean(yPred_gRNA[gene][gRNA_ID]) for gRNA_ID in list(yPred_gRNA[gene].keys())]
    
    # Create a scatter plot of the means vs predictions
    for gene in gene_list:    
        pc, pp = pearsonr(yTrue_mean[gene], yPred_mean[gene])
        sc, sp = spearmanr(yTrue_mean[gene], yPred_mean[gene])

        axis_dict[gene].plot(yTrue_mean[gene], yPred_mean[gene], 'o', markersize=30, color="#FF1493")
        axis_dict[gene].set_xlim(-1, 10)
        axis_dict[gene].set_ylim(-1, 10)
        axis_dict[gene].tick_params(axis='both', which='major', labelsize=40)
        axis_dict[gene].tick_params(axis='both', which='minor', labelsize=40)
        axis_dict[gene].set_xlabel("CRISPRa qPCR fold change", size=60)
        axis_dict[gene].set_ylabel("Model prediction's fold change", size=45)
        axis_dict[gene].set_title("Gene: "+gene+
                                  " peak_width = "+str(peak_width)+
                                  " inserted -log10(p-value) = "+str(p_value_mapping(inserted_lnp1_minuslog10_p_value))+
                                  "\nCorrelation between experimental and "+
                                  "predicted fold change\nPearson = "+
                                  str(round(pc, 2))+
                                  " ("+str(round(pp, 3))+
                                  ") Spearman = "+
                                  str(round(sc, 2))+
                                  " ("+str(round(sp, 3))+
                                  ")", size=40)

    return None


if __name__ == '__main__': 
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name')
    parser.add_argument('--trained_model')
    parser.add_argument('--window_size', type=int)
    args = parser.parse_args()

    trained_model = load_model(args.trained_model)

    # Generate data vectors for CXCR4 and TGFBR1
    cell_type_choice = -1

    CHROM = {'CXCR4':'chr2', 'TGFBR1':'chr9'}
    TSS = {'CXCR4':136118149, 'TGFBR1':99105113}
    STRAND = {'CXCR4':'-','TGFBR1':'+'}

    xInference = {}
    yInference = {}

    gene_list = ["CXCR4", "TGFBR1"]
    for gene in gene_list:
        
        prediction_generator = TranscriptomePredictor(args.window_size,
                               1,
                               shuffle=False,
                               mode='inference',
                               masking_probability=0.0,
                               chrom=CHROM[gene], 
                               start=int(TSS[gene]),
                               strand=STRAND[gene],
                               cell_type=cell_type_choice)

        for i in range(1):
            X, Y = prediction_generator.__getitem__(i)
            print(X.shape, Y.shape)

            xInference[gene] = X
            yInference[gene] = Y

            # np.save("../../Data/" + args.run_name + "." + gene + ".CT_" + str(cell_type_choice + 1) + ".npy", X)
            # np.save("../../Data/" + args.run_name + "." +  gene + ".CT_" + str(cell_type_choice + 1) + ".TPM.npy", Y)


    # assay_names = ['H3K36me3', 'H3K27me3', 'H3K27ac',
    #                'H3K4me1', 'H3K4me3', 'H3K9me3', 'MNase']

    # assay_colors = ['red', 'green', 'blue',
    #                 'cyan', 'pink', 'brown', 'purple']

    # Now load CRISPRa data from the Hilton Lab
    df = pd.read_csv("../../Data/p300_epigenome_editing_dataset.tsv", sep="\t")

    # For each position we have data for, perturb the epigenetic data and compute model predictions
    peak_width_choices = [6, 8]
    inserted_lnp1_minuslog10_p_value_choices = [1.5] # corresponds to p-value = 0.0003    

    fig, axs = plt.subplots(len(peak_width_choices) * len(inserted_lnp1_minuslog10_p_value_choices), 2)
    fig.set_size_inches(60, 60)

    plot_number = 0
    for peak_width in peak_width_choices:
        for inserted_lnp1_minuslog10_p_value in inserted_lnp1_minuslog10_p_value_choices:
            
            axis_dict = {}
            axis_dict['CXCR4'] = axs[plot_number, 0]
            axis_dict['TGFBR1'] = axs[plot_number, 1]
            plot_number += 1
            ise_results = []

            for index in tqdm(range(len(df))):

                gene = df.iloc[index, 0]
                if(gene not in gene_list):
                    continue

                chrom = CHROM[gene]
                tss = TSS[gene]
                strand = STRAND[gene]

                gRNA_ID = df.iloc[index, 5]
                bin_wrt_tss = df.iloc[index, 14] // RESOLUTION

                CRISPRa_qPCR_fold_change = df.iloc[index, 6]

                model_prediction_fold_change = ise(trained_model,
                                                   xInference[gene],
                                                   yInference[gene],
                                                   bin_wrt_tss,
                                                   peak_width,
                                                   inserted_lnp1_minuslog10_p_value)
        
                ise_results.append([gene, peak_width, inserted_lnp1_minuslog10_p_value, gRNA_ID, bin_wrt_tss, CRISPRa_qPCR_fold_change, model_prediction_fold_change])

            visualize_fold_change(axis_dict, ise_results)
                
    fig.savefig("../../Results/" + args.run_name + ".inference.pdf")

