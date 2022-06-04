#!/bin/bash 
#SBATCH --job-name=matrix
#SBATCH --account=gpuq
#SBATCH --partition=gpuq
##SBATCH --mail-user=wchen157@ucsc.edu 
##SBATCH --mail-type=ALL 
#SBATCH --output=run.out-%j 
#SBATCH --error=run.err-%j 
#SBATCH --mem=2G 
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1


./$1
