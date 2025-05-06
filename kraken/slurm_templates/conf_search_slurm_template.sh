#!/bin/bash
#SBATCH --partition=sigman-shared-np
#SBATCH --account=sigman-np
#SBATCH --qos=sigman-low-np
#SBATCH --time=$TIME:00:00
#SBATCH --ntasks=$NPROCS
#SBATCH --mem=$MEMG
#SBATCH -o kraken_conf_$KID-%j.log
#SBATCH -e kraken_conf_$KID-%j.error

module purge

setenv OMP_NUM_THREADS $NPROCS
setenv OMP_STACKSIZE 4G
ulimit -s unlimited

module load crest/2.12

run_kraken_conf_search.py \
    -i "$SMILES" \
    --kraken-id $KID \
    --nprocs $NPROCS \
    --conversion-flag $CONVERSION_FLAG \
    --calculation-dir "$CALCDIR" \
    --reduce_crest_output \
    --debug