#!/bin/bash 
#SBATCH --job-name=matrix
#SBATCH --account=gpuq
#SBATCH --partition=gpuq
##SBATCH --mail-user=wchen157@ucsc.edu 
##SBATCH --mail-type=ALL 
#SBATCH --output=matrix.out-%j 
#SBATCH --error=matrix.err-%j 
#SBATCH --mem=2G 
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1


./mat
