#!/bin/bash
#SBATCH --partition=sigman-kp
#SBATCH --account=sigman-kp
#SBATCH --qos=sigman-kp
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --mem=16G
#SBATCH -o $KID_DFT-%j.log
#SBATCH -e $KID_DFT_%j.error

conda activate kraken

python3 -u /uufs/chpc.utah.edu/common/home/u6053008/kraken_dft/new_kraken_dft/james_new.py --kid $KID \
        --nprocs $NPROCS \
        --dir "$DIR" \
