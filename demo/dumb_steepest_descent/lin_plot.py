#!/usr/bin/python3

import argparse
import json
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    with open(args.filename, 'r') as fp:
        data = json.load(fp)
        x = [x for x in range(1, len(data)+1)]

        plt.plot(x, data)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()
