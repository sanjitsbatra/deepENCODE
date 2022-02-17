import numpy as np
import pandas as pd
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
from tqdm import tqdm
warnings.filterwarnings('ignore')

from chrmt_generator import RESOLUTION

Data_Path = "/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/Data/"

window_size = 401
operative_half_window_size = 100

assay_names = ['DNase', 'H3K36me3', 'H3K27me3', 'H3K27ac',
               'H3K4me1', 'H3K4me3', 'H3K9me3', 
               'Methylation_pos', 'Methylation_neg']
assay_colors = ['black', 'red', 'green', 'blue', 'cyan', 
                'pink', 'brown', 'purple', 'darkpink']


# Define in silico epi-mutagenesis
def ise(original_gene_features,
        trained_model,
        inserted_lnp1_minuslog10p_value = 3,
        inserted_peak_width = 2)

    # obtain the middle portion of this
    X = original_gene_features[:, 
                               (window_size // 2) -
                               operative_half_window_size:
                               (window_size // 2) +
                               operative_half_window_size + 1,
                               :]

    # Perform inference by introducing p-value changes with a peak width
    positions = range(operative_half_window_size - operative_half_window_size,
                      operative_half_window_size + operative_half_window_size 
                      + 1)
    yPred = []
    for pos in positions:
        X_modified = np.copy(X)
        for p in range(pos - inserted_peak_width // 2, 
                       pos + inserted_peak_width // 2 + 1):
            if( (p >= 0) and (p < max(positions)) ):
                # NOTE: this is ln( -log10(transformed p-value) + 1)
                if(X_modified[:, p, 2] > 10): 
                    # If H3K27me3 peak exists, then p300 doesn't work
                    print("H3K27me3 exists!")
                    pass
                else:
                    # Modify the H3K27ac peak
                    X_modified[:, p, 3] = max(X_modified[:, p, 3], 
                                              inserted_lnp1_minuslog10p_value)


        yPred_value = trained_model.predict(X_modified)[0]
        yPred.append(yPred_value)

    # Instead of scaling, divide by yPred
    yPred_native = trained_model.predict(X)[0]
    yPred_fold_change = (np.power(10, yPred) -1) /
            (np.power(10, yPred_native + 0.000001) -1)

    return yPred_fold_change


# Transform p-value back from log-space
def p_value_mapping(inserted_lnp1_minuslog10p_value):
    minuslog10p_value = np.expm1(inserted_lnp1_minuslog10p_value)
    p_value = np.power(10, -1 * minuslog10p_value)
    return round(minuslog10p_value, 4)


# Define axes math for creating matplotlib subplots
def convert_to_2D(idx, nrows, ncols):
    return idx//ncols, idx%ncols


# Visualize Alan's data with the model predictions
def validate_model(trained_model,
                   inserted_lnp1_minuslog10p_value,
                   inserted_peak_width):

    # Load p300 epigenome editing data
    df_p300 = pd.read_csv(Data_Path +
                          "p300_epigenome_editing_dataset.tsv",
                          sep="\t")

    TSS = {}
    STRANDS = {}
    CHROMS = {}
    GENES = {}
    for index in range(len(df_p300)):
        tss = df_p300.iloc[index, 13]
        gene_strand = df_p300.iloc[index, 3]
        chrom = df_p300.iloc[index, 2]
        gene = df_p300.iloc[index, 0]

        TSS[gene] = int(tss)
        if(gene_strand == "plus"):
            STRANDS[gene] = "+"
        elif(gene_strand == "minus"):
            STRANDS[gene] = "-"
        else:
            print("something wrong with strand!")
        CHROMS[gene] = chrom
        GENES[gene] = 1

    GENES_LIST = set(list(GENES.keys()))    

    df_GENES_values = {}
    df_GENES_means = {}
    for gene in GENES_LIST:
        df_GENES_values[gene] = df_p300[df_p300["p300 target gene"] == gene]
        (df_GENES_values[gene]["Position_wrt_TSS"] = 
        pd.to_numeric(df_GENES_values[gene]
                                     ["gRNA position  wrt TSS (hg38)"],
                                     errors='coerce')/RESOLUTION)

        df_GENES_means[gene] = df_GENES_values[gene].
                               groupby('Position_wrt_TSS').mean()
        df_GENES_means[gene].index.name = 'Position_wrt_TSS'
        df_GENES_means[gene].reset_index(inplace=True)

    # Perform in-silico epi-mutagenesis
    xticklabels = range(-operative_half_window_size, 
                        operative_half_window_size + 1)

    GENES_LIST = ["CXCR4", "TGFBR1"]

    fig, axes = plt.subplots(nrows=len(GENES_LIST),
                             ncols=2,
                             figsize=(40, 30),
                             sharey=False)

    fig.tight_layout(pad=1, w_pad=20, h_pad=25)

    for idx, gene in enumerate(sorted(GENES_LIST)):

        idx_x, idx_y = convert_to_2D(idx, 
                                     nrows=len(GENES_LIST),
                                     ncols=1)
        ax_1 = axes[idx_x, 0]
        ax_2 = axes[idx_x, 1]

        original_gene_features = np.load(Data_Path + gene + ".npy")
        gene_features = np.squeeze(original_gene_features, axis=0)
                        [(window_size // 2) - operative_half_window_size:
                        (window_size // 2) +o perative_half_window_size+1, :]

        df_values = df_GENES_values[gene]
        df_means = df_GENES_means[gene]

        gene_ise = ise(original_gene_features, 
                       trained_model,
                       inserted_lnp1_minuslog10p_value,
                       inserted_peak_width)

        # Create a scatter plot of the means vs predictions
        gene_ise_at_means = []
        alan_means = []
        for p_idx in range(len(df_means)):
            position_m = df_means.iloc[p_idx, 0]
            alan_mean = df_means.iloc[p_idx, 1]
            if(position_m + operative_half_window_size < 0):
                continue
            elif(position_m > operative_half_window_size):
                continue
            else:
                gene_ise_at_means.append(gene_ise[int(position_m) +
                                         operative_half_window_size])
                alan_means.append(alan_mean)

        pc, pp = pearsonr(list(alan_means), gene_ise_at_means)
        sc, sp = spearmanr(list(alan_means), gene_ise_at_means)

        ax_1.plot(list(alan_means), gene_ise_at_means,
                  'o', markersize=30, color="#FF1493")
        ax_1.set_xlim(-1, 1.1 * max(alan_means))
        ax_1.set_ylim(-1, 1.1 * max(gene_ise_at_means))
        ax_1.tick_params(axis='both', which='major', labelsize=40)
        ax_1.tick_params(axis='both', which='minor', labelsize=40)
        ax_1.set_xlabel("Mean experimental fold change", size=60)
        ax_1.set_ylabel("Model prediction's fold change", size=45)
        ax_1.set_title("Correlation between experimental and "+
                       "model predictions fold change\nPearson = "+
                       str(round(pc, 2))+
                       " ("+str(round(pp, 3))+
                       ") Spearman = "+
                       str(round(sc, 2))+
                       " ("+str(round(sp, 3))+
                       ")", size=40)

        epigenetic_features = gene_features[:, 3] # H3K27ac
        color_for_assay = assay_color[3]
        label_for_assay = assays[3]

        # Scale the model predictions     
        scaling_ratio = np.median(df_means['Measured fold change']) / 
                                           np.median(gene_ise - 0.0)
        scaled_model_predictions = (scaling_ratio * (gene_ise - 0.0)) + 0.0

        # Scale the epigenetic features
        epigenetic_scaling_ratio = max(df_means['Measured fold change']) /
                                                max(epigenetic_features - 0.0)
        scaled_epigenetic_features = (epigenetic_scaling_ratio * 
                                      (epigenetic_features - 0.0)) + 0.0

        ax_2.plot(xticklabels, scaled_model_predictions, 
                  'o-', color="#4daf4a", linewidth=5, markersize=2, 
                  label="(Scaled) Model Predictions " + label_for_assay)
        ax_2.plot(xticklabels, scaled_epigenetic_features, 
                  'o-', color="#8470FF", linewidth=5, markersize=1, 
                  label="(Scaled) Epigenetic Features " + label_for_assay)

        ax_2.bar(df_means['Position_wrt_TSS'], 
                 0.0 + (df_means['Measured fold change']), 
                 color="#f781bf", bottom=0, width=2, 
                 label="Experimental mean from qPCR")
        ax_2.plot(df_values['Position_wrt_TSS'], 
                  0.0 + (df_values['Measured fold change']), 
                  'o', color="#e41a1c", 
                  label="Experimental data from qPCR", markersize=10)

        ax_2.set_xlim(-operative_half_window_size-10,
                      operative_half_window_size+10)
        ax_2.set_ylim(-1, 1.0 + max(df_means['Measured fold change'])*1.5)
        x_v = ax_2.get_xticks()
        ax_2.set_xticklabels(['{:3.0f}'.format(x * RESOLUTION) for x in x_v])
        ax_2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_2.tick_params(axis='both', which='major', labelsize=35)
        ax_2.tick_params(axis='both', which='minor', labelsize=35)
        ax_2.set_xlabel("Peak Position (in bp) w.r.t TSS", size=50)
        ax_2.set_ylabel("Gene expression fold change", size=50)
        ax_2.set_title(gene + " with H3K27ac + " +
                       assay_names[assay_index-1] + "\nincreasing " +
                       str(peak_width * RESOLUTION) + 
                       "bp peaks by -log10(p_value)=" +
                       str(p_value_mapping(inserted_lnp1_minuslog10p_value)),
                       size=40) 

        ax_2.legend(loc='upper center', prop={'size': 30}, ncol=2)

    plt.show()
    plt.close()
    