#!/bin/bash

# module load anaconda/2023b
#SBATCH --job-name=gcn_train
#SBATCH -o saves/slurm-%j.out
#SBATCH --gres=gpu:volta:1


python -u train_model.py --save-path-dir "../models" --data-dir "../data" --gnn-type "GCN"

wait

echo "All processes finished."