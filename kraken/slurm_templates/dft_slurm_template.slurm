#!/bin/bash
#SBATCH --partition=sigman-np
#SBATCH --account=sigman-np
#SBATCH --qos=sigman-low-np
#SBATCH --time=$TIME:00:00
#SBATCH --ntasks=$NPROCS
#SBATCH --mem=$MEMG
#SBATCH -o kraken_dft_$KID-%j.log
#SBATCH -e kraken_dft_$KID-%j.error

module purge
module load gaussian16/SSE4.C01

run_kraken_dft --kid $KID \
               --dir "$CALCDIR" \
               --nprocs $NPROCS \
