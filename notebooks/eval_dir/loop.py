import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)
args = parser.parse_args()
for _ in range(args.n):
    os.system('python ../notebooks/results_figures_refine.py')