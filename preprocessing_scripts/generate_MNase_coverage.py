# This script parses a bamtobed output file (assuming only properly paired read
# pairs) and creates a bedGragh file that adds 1 to coverage of every genomic
# position spanned by the insert in a read pair, thereby preventing double
# counting obtained by simply running bamtocoverage on the PE150 MNase-seq data


import sys
import numpy as np


if __name__ == '__main__':

    chromosomes = ['chr' + str(c) for c in list(range(1, 23)) + ["X"]]

    f_output = open(sys.argv[2] + ".MNase_coverage.bedGraph", 'w')
    f = open(sys.argv[1], 'r')

    line_number = 0
    for line in f:
        line_number += 1

        vec = line.rstrip('\n').split('\t')
        chrom = vec[0]
        start = int(vec[1])
        end = int(vec[2])
        read_name = vec[3]
        mapq = int(vec[4])
        strand = vec[5]

        if(line_number % 2 == 1):
            first_read_prefix = read_name[:-2]
            first_read_suffix = read_name[-2:]

            first_chrom = chrom
            first_mapq = mapq

            s1 = start
            e1 = end - 1  # bed format is 0-based half-open
        else:
            second_read_prefix = read_name[:-2]
            second_read_suffix = read_name[-2:]
            assert(((first_read_suffix == "/1") and
                    (second_read_suffix == "/2")) or
                   ((first_read_suffix == "/2") and
                    (second_read_suffix == "/1"))
                   )
            assert(first_read_prefix == second_read_prefix)

            second_chrom = chrom
            second_mapq = mapq

            if((first_chrom not in chromosomes) or
               (second_chrom not in chromosomes)):
                continue

            assert(first_chrom == second_chrom)

            if((first_mapq < 30) or (second_mapq < 30)):
                continue

            s2 = start
            e2 = end - 1  # bed format is 0-based half-open

            positions = [s1, e1, s2, e2]
            min_position = np.min(positions)
            max_position = np.max(positions)

            # bedGraph is 0-based half-open and our midpoints are all 0-based
            print(first_chrom + "\t" +
                  str(min_position) + "\t" +
                  str(max_position + 1) + "\t1",
                  file=f_output)
