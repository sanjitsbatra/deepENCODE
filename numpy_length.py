#!/bin/bash

import numpy as np
import sys

v = np.load(sys.argv[1])
print len(v)
