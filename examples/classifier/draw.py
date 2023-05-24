#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

with open(args.filename) as fp:
    for line in fp.readlines():
        img = np.empty((16, 16), dtype=int)

        line.strip()
        bits = line.split(' ')
        for i in range(0, 16):
            for j in range(0, 16):
                img[i][j] = int(float(bits[i*16 + j]))

        plt.imshow(img, cmap='binary')
        plt.show()
