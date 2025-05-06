#!/bin/csh
#SBATCH --partition=sigman-shared-np
#SBATCH --account=sigman-np
#SBATCH --time=48:00:00
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH -o kraken_$KRAKENID-%j.log
#SBATCH -e kraken_$KRAKENID-%j.error

module purge

setenv OMP_NUM_THREADS 16
setenv OMP_STACKSIZE 4G
ulimit -s unlimited

module load crest/2.12

python3 -u /uufs/chpc.utah.edu/common/home/u6053008/kraken/run_kraken.py \
    -i "$SMILES" \
    --kraken-id $KRAKENID \
    --nprocs 16 \
    --conversion-flag 4 \
    --calculation-dir $CALCDIR \
    --reduce_crest_output \
    --verbose \
    --debug