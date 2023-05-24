#!/usr/bin/python3

import argparse
import json
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    for filename in args.filenames:
        with open(filename, 'r') as fp:
            data = json.load(fp)
            x = [x for x in range(1, len(data)+1)]
    
            plt.plot(x, data)
    
    plt.legend([filename.rsplit('.', 1)[0]
                for filename in args.filenames])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
