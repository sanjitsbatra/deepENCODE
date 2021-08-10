# This script accepts as input TPM values for a cell line
# It then converts it into two 100bp resolution npy arrays; one for each strand
# It also performs a log10 of the TPM values for normality
import numpy as np
import sys


reference_npy_path = "/scratch/sanjit/ENCODE_Imputation_Challenge/" \
                     "2_April_2020/Data/100bp_12_7_Data_20_July_2020/T12A07"

output_npy_path = "/scratch/sanjit/ENCODE_Imputation_Challenge/" \
                  "2_April_2020/Data/Gene_Expression/genome_wide_TPM_npy"

chroms = ["chr"+str(i) for i in range(1, 23, 1)]

RESOLUTION = 100  # 100bp is the resolution of the npy tracks


if __name__ == '__main__':

    f_tpm = open(sys.argv[1], 'r')
    col_number = int(sys.argv[2])  # what column contains TPM

    # Load reference numpy arrays to obtain chromosome size
    chrom_npy = {}
    chrom_length = {}
    output_npy_pos = {}
    output_npy_neg = {}
    for chrom in chroms:
        chrom_npy[chrom] = np.load(reference_npy_path+"."+chrom+".npy")
        chrom_length[chrom] = chrom_npy[chrom].shape[0]  # at 100bp resolution

        # Initialize output numpy array
        output_npy_pos[chrom] = np.zeros(chrom_length[chrom])
        output_npy_neg[chrom] = np.zeros(chrom_length[chrom])

    # Read in TPM file
    line_number = 0
    for line in f_tpm:
        vec = line.rstrip("\n").split("\t")
        line_number += 1
        if(line_number == 1):
            cell_type_name = vec[col_number]
        else:
            chrom = vec[0]
            if(chrom not in chroms):
                continue
            strand = vec[3]
            if(strand == "+"):
                TSS = int(int(vec[1])/RESOLUTION)
                output_npy_pos[chrom][TSS] = float(vec[col_number])
            elif(strand == "-"):
                TSS = int(int(vec[2])/RESOLUTION)
                output_npy_neg[chrom][TSS] = float(vec[col_number])
            else:
                print("Something wrong with strand", strand, file=sys.stderr)
                sys.exit(-2)

    # Write the output numpy arrays and perform log10(x+1)
    for chrom in chroms:
        np.save(output_npy_path+"/"+cell_type_name+"."+chrom+".+.npy",
                np.log10(output_npy_pos[chrom] + 1))
        np.save(output_npy_path+"/"+cell_type_name+"."+chrom+".-.npy",
                np.log10(output_npy_neg[chrom] + 1))
