# This script parses a bamtobed output file (assuming only properly paired read
# pairs) and computes the midpoint of each read pair
# It then creates a bedGraph file containing the number of read pairs with
# their midpoint at a particular position


import sys
import numpy as np
from collections import Counter


if __name__ == '__main__':

    chromosomes = ['chr' + str(c) for c in list(range(1, 23)) + ["X"]]
    midpoints = {}
    for chrom in chromosomes:
        midpoints[chrom] = []

    f_insert_sizes = open(sys.argv[2] + ".insert_sizes.tsv", 'w')
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

            if((first_chrom not in midpoints) or
               (second_chrom not in midpoints)):
                continue

            if((first_mapq < 30) or (second_mapq < 30)):
                continue

            s2 = start
            e2 = end - 1  # bed format is 0-based half-open

            positions = [s1, e1, s2, e2]
            min_position = np.min(positions)
            max_position = np.max(positions)
            midpoint = (np.min(positions) + np.max(positions)) // 2
            insert_size = max_position - min_position + 1
            print(second_read_prefix + "\t" + str(insert_size),
                  file=f_insert_sizes)
            midpoints[chrom].append(midpoint)

    # Now we write out the bed file containing midpoint coverage
    f_output = open(sys.argv[2] + ".MNase.bedGraph", 'w')

    for chrom in chromosomes:
        midpoint_list = midpoints[chrom]
        count_dict = Counter(midpoint_list)
        midpoint_list = np.unique(sorted(midpoint_list))
        for e in midpoint_list:
            # bedGraph is 0-based half-open and our midpoints are all 0-based
            print(chrom + "\t" + str(e) + "\t" + str(e + 1) + "\t" +
                  str(count_dict[e]), file=f_output)
