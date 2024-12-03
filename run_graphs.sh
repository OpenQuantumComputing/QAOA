#!/bin/bash
#SBATCH --job-name=run_graphs
#              d-hh:mm:ss
#SBATCH --time=30-00:00:00
#SBATCH --output=/home/franzf/MaxKCut/%j.out
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1

python run_graphs.py "$1" "$2" "$3" "$4" "$5" "$6" "$7"

