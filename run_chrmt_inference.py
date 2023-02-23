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


# Process all eight genes by first converting to ranks within each gene, then Z-scoring ranks and concatenating
def combined_visualize(df, mode, transcript_list):

    true_values = []
    true_ranks = []
    
    pred_values = []
    pred_ranks = []

    for gene in pd.unique(df.gene):

        true_values.extend((gene, v) for v in df.loc[df.gene == gene, 'true'])
        these_true_ranks = df.loc[df.gene == gene, 'true'].rank()
        these_true_ranks -= np.mean(these_true_ranks)
        these_true_ranks /= np.std(these_true_ranks)
        true_ranks.extend([(gene, x) for x in these_true_ranks])

        pred_values.extend((gene, v) for v in df.loc[df.gene == gene, 'pred'])
        these_pred_ranks = df.loc[df.gene == gene, 'pred'].rank()
        these_pred_ranks -= np.mean(these_pred_ranks)
        these_pred_ranks /= np.std(these_pred_ranks)
        pred_ranks.extend([(gene, x) for x in these_pred_ranks])

        num_genes = (df.gene == gene).sum()

    genes_list = [x[0] for x in pred_ranks]            

    true_values = [x[1] for x in true_values]
    true_ranks = [x[1] for x in true_ranks] # np.array(qpcr_ranks).flatten()

    pred_values = [x[1] for x in pred_values]        
    pred_ranks = [x[1] for x in pred_ranks] # np.array(pred_ranks).flatten()

    '''    
    # add some noise to ranks
    true_ranks = [x + np.random.normal(0, 0.05, 1) for x in true_ranks]
    pred_ranks = [x + np.random.normal(0, 0.05, 1) for x in pred_ranks]
    '''

    cross_gene_spearman = spearmanr(true_ranks, pred_ranks)


    '''
    colors = plt.cm.cool(np.linspace(0.0, 1.0, 8))
    gene_2_index = {}
    for i, x in enumerate(transcript_list):
        if(mode == "endogenous"):
            gene_2_index[df[0]] = i
        elif(mode == "CRISPRa"):
            gene_2_index[df[1]] = i
        else:
            print("WRONG MODE")
            
    Color_Blind_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c']# , '#dede00']

    index_2_color = dict((i, Color_Blind_color_cycle[i]) for i in range(len(Color_Blind_color_cycle)))
    colors_list = [index_2_color[gene_2_index[gene]] for gene in genes_list]    

    transcript_dict = dict(transcript_list)

    merged = [(x[0][0], x[1][0], x[2]) for x in list(zip(true_ranks, pred_ranks, genes_list))]
    values = set(map(lambda x:x[2], merged))
    newlist = [[(y[0], y[1], y[2]) for y in merged if y[2]==x] for x in values]

    fig = plt.figure()
    
    for i, l in enumerate(newlist):
        tr = [x[0] for x in l]
        pr = [x[1] for x in l]
        g = [x[2] for x in l]

        assert(list(set(g))[0] == g[0])
        
        if(mode == "endogenous"):
            label_value = transcript_dict[g[0]]
        elif(mode == "CRISPRa"):
            label_value = g[0]
        else:
            print("WRONG MODE")

        plt.scatter(tr, pr, color=index_2_color[gene_2_index[g[0]]], label=label_value, s=200)

    plt.xlim(-2.1, 2.0)
    plt.ylim(-2.1, 3.0)

    if(mode == "endogenous"):
        plt.xlabel("Normalized rank of RNA-seq expression across cell types within each gene", fontsize=28)
        plt.ylabel("Normalized rank of predicted expression", fontsize=28)
    elif(mode == "CRISPRa"):
        plt.xlabel("Normalized Rank of qPCR fold change across gRNAs within each gene", fontsize=28)
        plt.ylabel("Normalized Rank of predicted fold change", fontsize=28)
    else:
        print("WRONG MODE")
        

    plt.tick_params(axis='x', labelrotation=0, labelsize=30)
    plt.tick_params(axis='y', labelrotation=0, labelsize=30)

    
    if(mode == "endogenous"):
        plt.title("Spearman correlation = " + str(round(cross_gene_spearman[0], 2)) + " for endogenous expression", fontsize=38)
    elif(mode == "CRISPRa"):
        plt.title("Spearman correlation = " + str(round(cross_gene_spearman[0], 2)) + " for CRISPRa fold change", fontsize=38)        
    else:
        print("WRONG MODE")
        
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, loc="upper center", ncol=4, fontsize=32, markerscale=2)
    '''

    # return fig
    return cross_gene_spearman[0], cross_gene_spearman[1]


if __name__ == '__main__':

    RESOLUTION = 25
    context_size_list = [401]

    ALL_CELL_TYPES = ["T{0:0=2d}".format(x) for x in [1,2,3,4,5,6,7,8,9,10,11,12,13]]
    assays = ['H3K36me3', 'H3K27me3', 'H3K27ac', 'H3K4me1', 'H3K4me3', 'H3K9me3', 'MNase']

    model_type_rename_dict = {"linear":"ridge", "maxpool":"cnn"}

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type')
    parser.add_argument('--inserted_peak_width')
    parser.add_argument('--ise_radius')
    parser.add_argument('--stranded_MNase_offset')
    parser.add_argument('--inserted_scalar')
    parser.add_argument('--maximum_inserted_minuslog10_p_value')
    parser.add_argument('--gaussian_bandwidth')
    parser.add_argument('--nucleosome_lambda')
    parser.add_argument('--output_prefix')
    parser.add_argument('--H3K27me3_flag')
    parser.add_argument('--dCas9_binding_flag')
    parser.add_argument('--MNase_scaling_flag')
    parser.add_argument('--maximum_flag')
    args = parser.parse_args()

    output_file_name = "../../Logs/" + args.output_prefix + "_" + args.model_type + "_" + args.inserted_peak_width + "_" + args.ise_radius + "_" + args.stranded_MNase_offset + "_" + args.inserted_scalar + "_" + args.maximum_inserted_minuslog10_p_value + "_" + args.gaussian_bandwidth + "_" + args.nucleosome_lambda + "_" + str(args.H3K27me3_flag) + "_" + str(args.dCas9_binding_flag) + "_" + str(args.MNase_scaling_flag) + "_" + str(args.maximum_flag) + ".csv"

    print("Output_file_name is", output_file_name)

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
                       ("ENST00000322002.4", "SOX11")
                      ]


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
    
    cross_gene_results_list = []

    # Then, for a particular transcript, we compute the ISE scores mean-ed across many models and compare with Alan's data
    for transcript_index in range(len(transcript_list)):
        
        transcript = transcript_list[transcript_index][0]
        gene = transcript_list[transcript_index][1]

        inserted_scalar = float(args.inserted_scalar) 
        maximum_inserted_minuslog10_p_value = float(args.maximum_inserted_minuslog10_p_value)

        inserted_peak_width = int(args.inserted_peak_width)
        gaussian_bandwidth = float(args.gaussian_bandwidth)

        nucleosome_lambda = float(args.nucleosome_lambda)

        stranded_MNase_offset = int(args.stranded_MNase_offset) 

        ise_radius = int(args.ise_radius) 
        bin_wrt_tss_choices = list(range(-ise_radius, ise_radius)) # region where we perform ISE

        epigenetic_features = np.load("../../Data/saved_npy_arrays/." + transcript + ".CT_"+str(-1)+".npy")
        TPM = (np.power(10, np.load("../../Data/saved_npy_arrays/." + transcript + ".CT_"+str(-1)+".TPM.npy")) - 1)[0][0]    

        min_MNase = MNase_dict[gene][0]
        max_MNase = MNase_dict[gene][1]

        for model_type in [args.model_type]:

            for context_size in context_size_list: # 81, 161, 241, 321, 401]:
                half_window = context_size // 2

                num_layers = num_layers_dict[context_size]

                fold_change_dict = {}
                native_TPM_list = []

                for replicate in tqdm(range(1, 101, 1)):

                    K.clear_session() # 2022 and what an insane bug in Keras! Time to switch to PyTorch
                    trained_model = load_model("../../Models/manuscript.R13.stochastic_chromosomes.replicate_"+str(replicate)+"_0_transcriptome_"+str(context_size)+"_"+str(num_layers)+"_32_"+str(model_type)+"_mse_1_-1.hdf5", compile=False)

                    key = (inserted_scalar,
                           maximum_inserted_minuslog10_p_value,
                           gaussian_bandwidth,
                           nucleosome_lambda,
                           inserted_peak_width,
                           ise_radius,
                           stranded_MNase_offset)

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
                    
                    native_TPM_list.append(predicted_native_TPM)
                
                mean_native_TPM = np.mean(native_TPM_list)
                stdev_native_TPM = np.std(native_TPM_list)
                
                mean_fold_change_list = np.mean(fold_change_dict[key], axis=0)
                stdev_fold_change_list = np.std(fold_change_dict[key], axis=0)

                ise_mean_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)] = (mean_fold_change_list, mean_native_TPM)
                ise_stdev_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)] = (stdev_fold_change_list, stdev_native_TPM)

                # Now we compare the model predictions w.r.t Alan's data
                scatter_dict = {}
                scatter_dict[gene] = {}
                scatter_dict[gene][(model_type_rename_dict[model_type], context_size)] = []

                for (transcript, gRNA, pos) in p300_dict[gene].keys():

                    # If the gRNA_bin_wrt_TSS which is named pos, is not within the ise_radius, then ignore
                    if(abs(pos) >= ise_radius):
                        print(ise_radius, pos, file=sys.stderr)
                        continue

                    fold_change_list = p300_dict[gene][(transcript, gRNA, pos)]
                    mean_fold_change = np.mean(fold_change_list)
                    stdev_fold_change = np.std(fold_change_list)

#                     scatter_dict[gene][(model_type_rename_dict[model_type], context_size)].append((mean_fold_change, ise_mean_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)][ise_radius + pos]))
                    
                    scatter_dict[gene][(model_type_rename_dict[model_type], context_size)].append((mean_fold_change,                                 ise_mean_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)][0][ise_radius + pos],
                                                                                                   stdev_fold_change,
                    ise_mean_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)][1],
                    ise_stdev_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)][0][ise_radius + pos],
                    ise_stdev_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)][1],
                                                                                                   gene, gRNA, pos))

                    
                # mean_predictions = ise_mean_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)]
                # stdev_predictions = ise_stdev_fold_change_dict[(transcript, model_type_rename_dict[model_type], context_size, key)]

                scatter_x = [x[0] for x in scatter_dict[gene][(model_type_rename_dict[model_type], context_size)]]
                scatter_y = [x[1] for x in scatter_dict[gene][(model_type_rename_dict[model_type], context_size)]]

                sc, sp = spearmanr(scatter_x, scatter_y)
                results_list.append([transcript, gene,
                                     model_type_rename_dict[model_type],
                                     context_size, round(key[0], 4),
                                     round(key[1], 4), round(key[2], 4),
                                     round(key[3], 4), round(key[4], 4),
                                     round(key[5], 4), round(key[6], 4),
                                     round(sc, 3), round(sp, 3)])
                
                cross_gene_results_list.extend(scatter_dict[gene][(model_type_rename_dict[model_type], context_size)])
                
    df_alan_predictions = pd.DataFrame(cross_gene_results_list, columns=["mean_qPCR", "mean_predicted_fold_change", "stdev_qPCR", "mean_predicted_endogenous_TPM", "stdev_predicted_fold_change", "stdev_predicted_endogenous_TPM", "gene", "gRNA", "bin_wrt_TSS"])

    df_CRISPRa = df_alan_predictions[["mean_qPCR", "mean_predicted_fold_change", "gene"]].set_axis(['true', 'pred', 'gene'], axis=1, inplace=False)

    '''
    # Now we compute the cross cell type correlations
    endogenous_TPM_dict = {}
    predicted_TPM_dict = {}

    for i, t_pair in enumerate(transcript_list):

        transcript = t_pair[0]
        gene = t_pair[1]

        endogenous_TPM_dict[transcript] = []
        predicted_TPM_dict[transcript] = []
        
        for model_type in ["maxpool"]:
            for context_size in context_size_list: # list(num_layers_dict.keys())[16:]:
                half_window = context_size // 2
                num_layers = num_layers_dict[context_size]                
                
                for cell_type in np.arange(-len(ALL_CELL_TYPES), 0):

                    epigenetic_features = np.load("../../Data/saved_npy_arrays/." + transcript + ".CT_"+str(cell_type)+".npy")
                    TPM = (np.power(10, np.load("../../Data/saved_npy_arrays/." + transcript + ".CT_"+str(cell_type)+".TPM.npy")) - 1)[0][0]    
                
                    predicted_endogenous_TPM_list = []

                    for replicate in tqdm(range(1, 101, 1)):

                        K.clear_session() # 2022 and what an insane bug in Keras! Time to switch to PyTorch
                        trained_model = load_model("../../Models/manuscript.R13.stochastic_chromosomes.replicate_"+str(replicate)+"_0_transcriptome_"+str(context_size)+"_"+str(num_layers)+"_32_"+str(model_type)+"_mse_1_-1.hdf5", compile=False)    

                        predicted_endogenous_TPM = np.power(10, trained_model.predict(epigenetic_features[:, 200-half_window:200+half_window+1, :-1])[0][0]) -1 # Same reason for 200, as above
                        predicted_endogenous_TPM_list.append(predicted_endogenous_TPM)

                    predicted_endogenous_TPM_list = np.log10(np.asarray(predicted_endogenous_TPM_list) + 1)
                    mean_predicted_endogenous_TPM = round(np.mean(predicted_endogenous_TPM_list), 4)
                    stdev_predicted_endogenous_TPM = round(np.std(predicted_endogenous_TPM_list), 4)
               
                    endogenous_TPM_dict[transcript].append(TPM)
                    predicted_TPM_dict[transcript].append(mean_predicted_endogenous_TPM)
    '''
    
    '''
    true_values = []
    pred_values = []
    genes = []
    for t in transcript_list:
        transcript = t[0]
        
        true_values.extend(endogenous_TPM_dict[transcript])
        pred_values.extend(predicted_TPM_dict[transcript])
        
        genes.extend([transcript] * len(endogenous_TPM_dict[transcript]) )
    '''

    '''   
    # add some noise to values
    true_values = [x + np.random.normal(0, 0.0001, 1) for x in true_values]
    pred_values = [x + np.random.normal(0, 0.0001, 1) for x in pred_values]
    '''   

    '''
    df_endogenous = pd.DataFrame(list(zip(true_values, pred_values, genes)), columns =['true', 'pred', 'gene'])
    '''

    # Now compute the spearman correlations aggregating all eight genes into a single scatter plot

    # sc_e, sp_e = combined_visualize(df_endogenous, 'endogenous', transcript_list)
    sc_e = 0
    sp_e = 0

    sc_c, sp_c = combined_visualize(df_CRISPRa, 'CRISPRa', transcript_list)

    with open(output_file_name, 'w') as f_output:

        # for result in results_list:
        #     f_output.write(f"{result}\n")
        # f_output.write("Next\n")
        
        f_output.write(str(inserted_scalar) + "\t" +
                       str(maximum_inserted_minuslog10_p_value) + "\t" +
                       str(inserted_peak_width) + "\t" +
                       str(gaussian_bandwidth) + "\t" + 
                       str(nucleosome_lambda) + "\t" +
                       str(stranded_MNase_offset) + "\t" + 
                       str(context_size) + "\nSpearmans:\n")

        f_output.write(str(sc_e) + "\t" + str(sp_e) + "\n")
        f_output.write(str(sc_c) + "\t" + str(sp_c) + "\n")
        
        # for ii in range(len(qpcr_ranks)):
        #     f_output.write(str(genes_list[ii]) + "\t" + str(pred_ranks[ii]) + "\t" + str(qpcr_ranks[ii]) + "\n")
