#!/bin/bash

# module load anaconda/2023b
#SBATCH --job-name=gat_smiles
#SBATCH -o saves/slurm-%j.out
#SBATCH --gres=gpu:volta:1


python -u hyperopt_gridsearch.py --save-path-dir "../data/opts_short_smiles" --data-dir "../data" --gnn-type "GAT"

wait

echo "All processes finished."