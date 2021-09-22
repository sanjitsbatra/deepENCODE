import sys
import numpy as np
from chrmt_generator import TranscriptomeGenerator, TranscriptomePredictor
# from chrmt_train import maximum_likelihood_loss
from tensorflow.keras.models import load_model
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Define in-silico epigenesis
def ise(Data_Path, assays, gene, assay, inserted_lnlog10_p_value, peak_width):
    X = np.load(Data_Path + gene + ".npy")

    # Perform inference by introducing p-value changes with a peak width
    yPred = []
    center = window_size // 2
    positions = range(center - center, center + center + 1)
    for pos in positions:
        X_modified = np.copy(X)
        for p in range(pos - peak_width // 2, pos + peak_width // 2 + 1):
            if((p >= 0) and (p < max(positions))):
                if((assays[assay] == "H3K27ac") and
                   (X_modified[0, p, 2] > 1.825)):
                    # Then H3K27me3 already exists
                    # so, p300 activation shouldn't work
                    pass
                else:
                    X_modified[0, p, assay] += inserted_lnlog10_p_value

        yPred_value = trained_model.predict(X_modified)
        yPred_value = yPred_value[0][0]
        yPred.append(yPred_value)

    yPred_value = trained_model.predict(X)[0][0]
    yPred = (np.power(10, yPred) - 1) / (np.power(10, yPred_value) - 1)

    return yPred


def p_value_mapping(lnlog10p_value):
    log10p_value = np.expm1(lnlog10p_value)
    return log10p_value


def plot_ise(RESOLUTION, gene_ise, assays, gene, assay,
             inserted_p_value, peak_width):
    plt.figure(1)
    plt.plot(xticklabels, 1.0 * gene_ise, 'o-', color=assay_color[assay],
             markersize=10, label="Model Predictions " + assays[assay])
    plt.plot(df['Position_wrt_TSS'], df['Fold_Change'], 'o', color='gray',
             label="Experimental Data from qPCR")
    plt.xlim(-22, 22)
    plt.ylim(-3, 10)
    ax = plt.gca()
    x_vals = ax.axes.get_xticks()
    ax.set_xticklabels(['{:3.0f}'.format(x * 100) for x in x_vals])
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.tick_params(axis='both', which='minor', labelsize=30)
    plt.xlabel("Peak Position (in bp) w.r.t TSS", size=50)
    plt.ylabel("Gene expression fold change", size=40)
    plt.legend(loc=1, prop={'size': 18}, ncol=3)
    plt.title("Gene " + gene + ": inserting " + str(peak_width * RESOLUTION) +
              "bp peaks of -log10(p_value)=" +
              str(p_value_mapping(inserted_p_value)),
              size=35)

    return plt


if __name__ == '__main__':

    RESOLUTION = 25

    Data_Path = "/scratch/sanjit/ENCODE_Imputation_Challenge/2_April_2020/"
    Data_Path = Data_Path + "Data/"

    assays = ['DNase', 'H3K36me3', 'H3K27me3', 'H3K27ac',
              'H3K4me1', 'H3K4me3', 'H3K9me3']

    if(len(sys.argv) != 4):
        print("USAGE: python chrmt_report.py < model > < window_size >" +
              "< output_prefix >", file=sys.stderr)
        sys.exit(-1)

    trained_model_path = sys.argv[1]
    window_size = int(sys.argv[2])
    output_prefix = sys.argv[3]

    trained_model = load_model(trained_model_path)

    # Step-1: Compute True vs Predicted log10p1(TPM) correlation

    test_generator = TranscriptomeGenerator(window_size,
                                            256,
                                            shuffle=False,
                                            mode='testing'+'TSS_only',
                                            masking_probability=0.0)

    np.set_printoptions(precision=3, suppress=True)
    yTrue = []
    yPred = []
    for i in tqdm(range(100)):

        X, Y = test_generator.__getitem__(i)
        yPred_value = trained_model.predict(X)
        # print(X.shape, Y.shape, yPred_value.shape)
        yTrue.extend(np.squeeze(Y))
        yPred.extend(np.squeeze(yPred_value))

        pc, _ = pearsonr(yTrue, yPred)
        sc, _ = spearmanr(yTrue, yPred)
        # print("Pearson =", round(pc, 3), "Spearman =",
        # round(sc, 3))

    plt.figure(1)
    plt.rcParams["figure.figsize"] = (20, 12)
    plt.plot(yTrue, yPred, 'o', markersize=4, color='green')
    plt.xlabel("True Normalized TPM", size=40)
    plt.ylabel("Predicted Normalized TPM", size=40)
    plt.xlim(-0.5, 4)
    plt.ylim(-0.5, 2)
    plt.title("Pearson = "+str(round(pc, 3)) +
              "Spearman = "+str(round(sc, 3)), size=50)

    # Step-2a: Pre-compute the epigenetic features for a few exemplar genes
    STARTS = [136118149, 136116243, 99105113, 84905656, 33513875, 186694060]
    STRANDS = ["-", "-", "+", "+", "+", "+"]
    CHROMS = ["chr2", "chr2", "chr9", "chr2", "chr2", "chr2"]
    GENES = ["CXCR4", "CXCR4_alternative", "TGFBR1", "High", "Low", "Medium"]

    for gene_index in range(len(GENES)):
        start_value = int(STARTS[gene_index]/RESOLUTION)
        strand_value = STRANDS[gene_index]
        prediction_generator = TranscriptomePredictor(window_size,
                                                      1,
                                                      shuffle=False,
                                                      mode='testing',
                                                      masking_probability=0.0,
                                                      # The last cell type in
                                                      # CELL_TYPES is T13
                                                      cell_type=-1,
                                                      chrom=CHROMS[gene_index],
                                                      start=start_value,
                                                      strand=strand_value)

        for i in range(1):
            X, Y = prediction_generator.__getitem__(i)
            np.save(Data_Path + GENES[gene_index] + ".npy", X)

    # Step-2b: Load Alan's H3K27ac p300 dataset
    df_CXCR4 = pd.read_csv(Data_Path + "CXCR4.p300.tsv", sep="\t")
    df_CXCR4_values = df_CXCR4[df_CXCR4['Position_wrt_TSS'] != "Control"]
    v = pd.to_numeric(df_CXCR4_values["Position_wrt_TSS"], errors='coerce')
    v = v / RESOLUTION
    df_CXCR4_values["Position_wrt_TSS"] = v

    df_TGFBR1 = pd.read_csv(Data_Path + "TGFBR1.p300.tsv", sep="\t")
    df_TGFBR1_values = df_TGFBR1[df_TGFBR1['Position_wrt_TSS'] != "Control"]
    v = pd.to_numeric(df_TGFBR1_values["Position_wrt_TSS"], errors='coerce')
    v = v / RESOLUTION
    df_TGFBR1_values["Position_wrt_TSS"] = v

    # Step-2c:
    genes = ["CXCR4", "TGFBR1"]
    assay_color = ['black', 'red', 'green', 'blue', 'cyan', 'pink', 'brown']
    xticklabels = range(-(window_size // 2), (window_size // 2) + 1)

    peak_width_choices = range(1, 11)
    inserted_p_value_choices = range(1, 3, 0.25)

    for gene in genes:
        gene_features = np.squeeze(np.load(gene+".npy"), axis=0)
        for assay in range(1, 8):
            if(gene == "CXCR4"):
                df = df_CXCR4_values
            elif(gene == "TGFBR1"):
                df = df_TGFBR1
            else:
                df = df_CXCR4_values.iloc[0:0, :].copy()

            for inserted_p_value in inserted_p_value_choices:
                for peak_width in peak_width_choices:
                    gene_ise = ise(Data_Path, assays,
                                   gene, assay,
                                   inserted_p_value, peak_width)

                    plot_ise(gene_ise, assays,
                             gene, assay,
                             inserted_p_value, peak_width)
