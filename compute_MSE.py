import numpy as np
import sys

validation = np.load(sys.argv[1])

imputed = np.load(sys.argv[2])

mse = (np.square(validation - imputed)).mean(axis=0)

print(str(mse))
