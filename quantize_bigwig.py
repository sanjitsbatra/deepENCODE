import pyBigWig
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.cluster import MiniBatchKMeans
from itertools import groupby

ALLOWED_CHROMS = ['chr1','chr10','chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19','chr2','chr20', 'chr21', 'chr22','chr3','chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9','chrX']

bw = pyBigWig.open(sys.argv[1]+".bigwig")
bw_out = pyBigWig.open(sys.argv[1]+".quantized_K10"+".bigwig", "w")

bw_out.addHeader([('chr1', 248956422), ("chr10", 133797422), ("chr11", 135086622), ("chr12", 133275309), ("chr13", 114364328), ("chr14", 107043718), ("chr15", 101991189), ("chr16", 90338345), ("chr17", 83257441), ("chr18", 80373285), ("chr19", 58617616), ("chr2", 242193529), ("chr20", 64444167), ("chr21", 46709983), ("chr22", 50818468), ("chr3", 198295559), ("chr4", 190214555), ("chr5", 181538259), ("chr6", 170805979), ("chr7", 159345973), ("chr8", 145138636), ("chr9", 138394717), ("chrX", 156040895)])

# print "Number of intervals on chr3 are: " + str( len(bw.intervals("chr3")) )
for chr in ALLOWED_CHROMS:

	values = bw.values(chr, 0, bw.chroms(chr), numpy=True)
	#At the end of the chromosome there are NaNs in the bigwig
	values = np.nan_to_num(values)

	num_intervals = len(bw.intervals(chr))

	# Now we quantize the values in v into K bins and write out the bigwig file
	K = 10
	v = np.array(values)
	v = v.reshape(-1,1)

	# lloyd_kmeans = KMeans(n_clusters = K, algorithm = "auto").fit(v)
	lloyd_kmeans = MiniBatchKMeans(n_clusters = K, batch_size = 10000, max_iter = 2).fit(v)

	centers = lloyd_kmeans.cluster_centers_
	cluster_assignment = lloyd_kmeans.predict(v)
	quantized_v = [centers[x] for x in cluster_assignment]
	incurred_mse = mean_squared_error(quantized_v, v)

	rle = [(k, sum(1 for _ in g)) for k,g in groupby(quantized_v)]
	cur = 0
	l = []
	chr_list = []
	for i in range(len(rle)):
		chr_list.append(chr)
		l.append([cur,cur+rle[i][1],rle[i][0]]) # bigWig has half-open coordinates
		cur = cur+rle[i][1]		

	# Output bigwig
	bw_out.addEntries(chr_list, [item[0] for item in l], ends=[item[1] for item in l], values=[float(item[2]) for item in l])

	print "Number of quantizated intervals before = " + str(num_intervals) + " and after clustering with K = " + str(K) + " is: " + str(len(l)) + " with incurred mse = " + str(incurred_mse)


bw_out.close()




