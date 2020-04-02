import sys
import numpy as np
from scipy.stats.stats import pearsonr

v1 = np.squeeze( np.load(sys.argv[1]) )
v2 = np.squeeze( np.load(sys.argv[2]) )

print str(pearsonr(v1, v2)[0])
