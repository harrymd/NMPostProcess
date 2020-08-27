#!/bin/bash
#SBATCH -J NMPP
#SBATCH -o NMPP_%j.txt
#SBATCH -e NMPP_%j.err
#SBATCH -p name-of-queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=01:00:00
#SBATCH -A name-of-allocation
#SBATCH --mail-user=you@example.com
#SBATCH --mail-type=all
python3 process.py 