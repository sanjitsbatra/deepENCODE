import pyBigWig
import sys
from scipy.stats.stats import pearsonr   
import numpy as np

chr_num = sys.argv[5]

c1 = sys.argv[1]
a1 = sys.argv[2]

c2 = sys.argv[3]
a2 = sys.argv[4]

s1 = "C"+c1+"M"+a1+".bigwig"
s2 = "C"+c2+"M"+a2+".bigwig"

#print s1+" "+s2

#sys.exit(-1)

bw1 = pyBigWig.open("C"+c1+"M"+a1+".bigwig")
bw2 = pyBigWig.open("C"+c2+"M"+a2+".bigwig")

v1 = bw1.values(chr_num, 0, bw1.chroms(chr_num), numpy=True)
v2 = bw2.values(chr_num, 0, bw2.chroms(chr_num), numpy=True)

a = v1
b = v2

nas = np.logical_or(np.isnan(a), np.isnan(b))
pearson_corr,p_val = pearsonr(a[~nas], b[~nas])

print c1+" "+a1+" "+c2+" "+a2+" "+chr_num+" "+str(pearson_corr)
