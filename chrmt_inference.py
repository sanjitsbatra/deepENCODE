import os
import sys
import seaborn as sns
import numpy as np
import pandas as pd
from chrmt_generator import TranscriptomeGenerator, TranscriptomePredictor
# from chrmt_train import maximum_likelihood_loss
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages        
from matplotlib.ticker import MaxNLocator
import random
import itertools
import warnings
warnings.filterwarnings('ignore')
from chrmt_generator import RESOLUTION, EPS
import argparse
from tqdm import tqdm


def generate_data_vectors(cell_type_choice, window_size, path_to_save):

    CHROM = {'CXCR4':'chr2', 'TGFBR1':'chr9'}
    TSS = {'CXCR4':136118149, 'TGFBR1':99105113}
    STRAND = {'CXCR4':'-','TGFBR1':'+'}

    xInference = {}
    yInference = {}

    gene_list = ["CXCR4", "TGFBR1"]
    for gene in gene_list:
        
        prediction_generator = TranscriptomePredictor(window_size,
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

            # np.save(path_to_save + "." + gene + ".CT_" + str(cell_type_choice) + ".npy", X)
            # np.save(path_to_save + "." +  gene + ".CT_" + str(cell_type_choice) + ".TPM.npy", Y)

    return xInference, yInference, gene_list, CHROM, TSS, STRAND


def perturb_x(x, bin_wrt_tss, inserted_peak_width, inserted_lnp1_minuslog10_p_value, stranded_MNase_offset):

    # Here the shape of x would be 1 x window_size x NUMBER_OF_ASSAYS + 1 (for the MNase)
    half_window_size = (x.shape[1] - 1) // 2

    x_perturbed = np.copy(x)
    for p in range(bin_wrt_tss - inserted_peak_width // 2, 
                   bin_wrt_tss + inserted_peak_width // 2 + 1):
        
        position_in_x = p + half_window_size + stranded_MNase_offset

        if( (position_in_x >= 0) and (position_in_x < 2 * half_window_size) ):

            # Modify the H3K27ac peak NOTE: this is ln( -log10(transformed p-value) + 1)
            x_perturbed[:, position_in_x, 2] += (x[:, position_in_x, -1] * inserted_lnp1_minuslog10_p_value)

    return x_perturbed


# Perform in silico epi-mutagenesis
def ise(window_size, trained_model, x, y, bin_wrt_tss, inserted_peak_width, inserted_lnp1_minuslog10_p_value, stranded_MNase_offset):

    x_perturbed = perturb_x(x, bin_wrt_tss, inserted_peak_width, inserted_lnp1_minuslog10_p_value, stranded_MNase_offset)
   
    yPred = trained_model.predict(x[:, :, :-1])[0][0]
    yPred_perturbed = trained_model.predict(x_perturbed[:, :, :-1])[0][0]
    
    model_prediction_fold_change = (np.power(10, yPred_perturbed) - 1) / (np.power(10, yPred + EPS) - 1) 
    
    return model_prediction_fold_change


# Perform in silico epi-mutagenesis when the input is a pair of epigenetic tracks
def ise_pairwise_input(window_size, trained_model, x, y, bin_wrt_tss, inserted_peak_width, inserted_lnp1_minuslog10_p_value, stranded_MNase_offset):

    x_perturbed = perturb_x(x, bin_wrt_tss, inserted_peak_width, inserted_lnp1_minuslog10_p_value, stranded_MNase_offset)   
    # create the pairwise input
    half_window_size = (x.shape[1] - 1) // 2
    operative_half_window_size = (window_size - 1) // 2

    x_pairwise = np.concatenate([x[0, half_window_size - operative_half_window_size:half_window_size + operative_half_window_size + 1, 2], x[0, half_window_size - operative_half_window_size:half_window_size + operative_half_window_size + 1, 4]], axis=0)

    x_perturbed_pairwise = np.concatenate([x_perturbed[0, half_window_size - operative_half_window_size:half_window_size + operative_half_window_size + 1, 2], x_perturbed[0, half_window_size - operative_half_window_size:half_window_size + operative_half_window_size + 1, 4]], axis=0)

    yPred = trained_model.predict(np.expand_dims(x_pairwise, axis=0))
    yPred_perturbed = trained_model.predict(np.expand_dims(x_perturbed_pairwise, axis=0))
    
    # print(x.shape, x_pairwise.shape, 
    #       x_perturbed.shape, x_perturbed_pairwise.shape,
    #       yPred.shape, yPred_perturbed.shape)
    
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
        
        gene, peak_width, inserted_lnp1_minuslog10_p_value, stranded_MNase_offset, gRNA_ID, bin_wrt_tss, CRISPRa_qPCR_fold_change, model_prediction_fold_change = e        
        
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
        axis_dict[gene].set_xlim(-1, 6)
        axis_dict[gene].set_ylim(-1, 6)
        axis_dict[gene].tick_params(axis='both', which='major', labelsize=40)
        axis_dict[gene].tick_params(axis='both', which='minor', labelsize=40)
        axis_dict[gene].set_xlabel("CRISPRa qPCR fold change", size=60)
        axis_dict[gene].set_ylabel("Model prediction's fold change", size=45)
        axis_dict[gene].set_title(""+gene+
                                  " peak_width = "+str(peak_width)+
                                  " -log10(p-value) = "+str(p_value_mapping(inserted_lnp1_minuslog10_p_value))+
                                  "\nstranded MNase offset = "+str(stranded_MNase_offset)+
                                  "\nCorrelation between experimental and "+
                                  "predicted fold change\nPearson = "+
                                  str(round(pc, 2))+
                                  " ("+str(round(pp, 3))+
                                  ") Spearman = "+
                                  str(round(sc, 2))+
                                  " ("+str(round(sp, 3))+
                                  ")", size=40)

    return None


def perform_ise(fig, axs, 
                window_size,
                path_to_dataset, ise_function,
                peak_width_choices,
                inserted_lnp1_minuslog10_p_value_choices,
                MNase_offset,
                xInference, yInference,
                gene_list, CHROM, TSS, STRAND, 
                trained_model):

    assay_names = ['H3K36me3', 'H3K27me3', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K9me3', 'MNase']

    # Now load CRISPRa data from the Hilton Lab
    df = pd.read_csv(path_to_dataset, sep="\t")
    
    fig.set_size_inches(60, 60)

    plot_number = 0
    for peak_width in peak_width_choices:
        for inserted_lnp1_minuslog10_p_value in inserted_lnp1_minuslog10_p_value_choices:
            
            axis_dict = {}
            axis_dict['CXCR4'] = axs[plot_number, 0]
            axis_dict['TGFBR1'] = axs[plot_number, 1]

            axs[plot_number, 0].set_xlim(0.8, 6)
            axs[plot_number, 0].set_ylim(-0.2, 3)
            axs[plot_number, 1].set_xlim(0.8, 6)
            axs[plot_number, 1].set_ylim(-0.2, 3)

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

                gRNA_strand = df.iloc[index, 7]
                if(gRNA_strand == "plus"):
                    stranded_MNase_offset = MNase_offset * -1
                elif(gRNA_strand == "minus"):
                    stranded_MNase_offset = MNase_offset * +1
                else:
                    print("gRNA strand seems incorrect", file=sys.stderr)
                    stranded_MNase_offset = 0

                model_prediction_fold_change = ise_function(window_size,
                                                   trained_model,
                                                   xInference[gene],
                                                   yInference[gene],
                                                   bin_wrt_tss,
                                                   peak_width,
                                                   inserted_lnp1_minuslog10_p_value,
                                                   stranded_MNase_offset)
        
                ise_results.append([gene, peak_width, inserted_lnp1_minuslog10_p_value, stranded_MNase_offset, gRNA_ID, bin_wrt_tss, CRISPRa_qPCR_fold_change, model_prediction_fold_change])

            visualize_fold_change(axis_dict, ise_results)

    return fig


if __name__ == '__main__': 
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name')
    parser.add_argument('--trained_model')
    parser.add_argument('--window_size', type=int)
    args = parser.parse_args()

    trained_model = load_model(args.trained_model, compile=False)

    # Generate data vectors for CXCR4 and TGFBR1
    cell_type_choice = -1
    path_to_save = "../../Data/" + args.run_name
    xInference, yInference, gene_list, CHROM, TSS, STRAND = generate_data_vectors(cell_type_choice, args.window_size, path_to_save)

    peak_width_choices = [6, 8]
    inserted_lnp1_minuslog10_p_value_choices = [1.5]  # corresponds to 0.0003

    with PdfPages("../../Results/" + args.run_name + ".inference.pdf") as pdf:

        for MNase_offset in range(-1, 1 + 1):

            fig, axs = plt.subplots(len(peak_width_choices) * len(inserted_lnp1_minuslog10_p_value_choices), 2)

            path_to_dataset = "../../Data/p300_epigenome_editing_dataset.tsv"
            perform_ise(fig, axs,
                args.window_size,
                path_to_dataset, ise,
                peak_width_choices,
                inserted_lnp1_minuslog10_p_value_choices,
                MNase_offset,
                xInference, yInference,
                gene_list, CHROM, TSS, STRAND,                
                trained_model)

            pdf.savefig(fig)
            plt.close()

