import numpy as np
import sys

v = np.load(sys.argv[1])

a = sys.argv[1].split('.')
b = '.'.join(a[:-1])
c = b+".txt"

np.savetxt(c, v)

