#!/bin/bash

#SBATCH --job-name=lab1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=10GB
#SBATCH --time=20:20:00
#SBATCH --output=out.%j


unzip /scratch/at3577/cvpr/sample_submission.csv.zip -d /scratch/at3577/cvpr
unzip /scratch/at3577/cvpr/test_video_list_and_name_mapping.zip -d /scratch/at3577/cvpr
unzip /scratch/at3577/cvpr/test.zip -d /scratch/at3577/cvpr
unzip /scratch/at3577/cvpr/train_color.zip -d /scratch/at3577/cvpr
unzip /scratch/at3577/cvpr/train_label.zip -d /scratch/at3577/cvpr
unzip /scratch/at3577/cvpr/train_video_list.zip -d /scratch/at3577/cvpr


















