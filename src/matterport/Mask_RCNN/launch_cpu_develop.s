#!/bin/bash

#SBATCH --job-name=lab1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=18:20:00
#SBATCH --output=out.%j

module purge
module load jupyter-kernels/py2.7
module load scikit-image/intel/0.13.1

#python ./Attempt1_Only_Test.py

python ./LabelImageMapper.py 



