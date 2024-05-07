#!/bin/bash

# module load anaconda/2023b
#SBATCH --job-name=gcn
#SBATCH -o saves/slurm-%j.out
#SBATCH --gres=gpu:volta:1


python -u hyperopt_gridsearch.py --save-path-dir "../data/opts_short" --data-dir "../data" --gnn-type "GCN"

wait

echo "All processes finished."