#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: James Howard, PhD
# Affiliation: Department of Chemistry, The University of Utah
# Date: 2025-04-10

'''
Simple script for submitting Kraken conformer
searches to HPC systems
'''

import re
import sys
import copy
import logging
import argparse
import subprocess

from pathlib import Path

import pandas as pd

# Template SLURM submission script with placeholders for submission
DEFAULT_SLURM_TEMPLATE = Path(__file__).parent.parent / 'scripts/kraken_conf_search.sh'

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
    datefmt='%m/%d/%Y:%H:%M:%S',  # Correct way to format the date
    handlers=[logging.StreamHandler(sys.stdout)]
)

DESCRIPTION = r"""


               ╔══════════════════════════════════════╗
               ║   | |/ / _ \  /_\ | |/ / __| \| |    ║
               ║   | ' <|   / / _ \| ' <| _|| .` |    ║
               ║   |_|\_\_|_\/_/ \_\_|\_\___|_|\_|    ║
               ╚══════════════════════════════════════╝


        Kolossal viRtual dAtabase for moleKular dEscriptors
                     of orgaNophosphorus ligands.

                              CLI SCRIPT

              """

def get_args() -> argparse.Namespace:
    '''Gets the arguments for running Kraken'''

    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, 2, 80),
        allow_abbrev=False,
        usage=argparse.SUPPRESS,
        add_help=False)

    parser.add_argument('-h',
                        '--help',
                        action='help',
                        default=argparse.SUPPRESS,
                        help='Show this help message and exit.\n\n')

    parser.add_argument('-i', '--input',
                        dest='input',
                        required=True,
                        type=Path,
                        help='Input .csv file',
                        metavar='CSV')

    parser.add_argument('-c', '--calcdir',
                        dest='calcdir',
                        required=False,
                        type=Path,
                        default=Path.cwd(),
                        help='Directory in which to submit the jobs',
                        metavar='CSV')

    args = parser.parse_args()

    # Do some path checking
    args.input = Path(args.input)
    if not args.input.exists():
        raise FileNotFoundError(f'{args.input.absolute()} does not exist.')

    args.calcdir = Path(args.calcdir)
    if not args.calcdir.exists():
        raise FileNotFoundError(f'{args.calcdir.absolute()} does not exist.')

    if args.input.suffix != '.csv':
        raise ValueError(f'Only .csv files can be used with this script not {args.input.suffix}')

    return args

def main():

    # Get the arguments
    args = get_args()

    # Read in the df
    df = pd.read_csv(args.input, dtype={'KRAKEN_ID': str})

    with open(DEFAULT_SLURM_TEMPLATE, 'r', encoding='utf-8') as _:
        slurm_template = _.read()

    # Iterate through all the smiles
    for i, row in df.iterrows():

        # Get the important things we'll use in the script
        smiles = row['SMILES']
        kraken_id = str(row['KRAKEN_ID'])
        calcdir = args.calcdir

        # Copy the slurm template
        text = copy.deepcopy(slurm_template)

        text = re.sub(r'\$KRAKENID', kraken_id, text)
        text = re.sub(r'\$SMILES', smiles, text)
        text = re.sub(r'\$CALCDIR', str(calcdir.absolute()), text)

        slurm_job_file = Path('.') / f'{kraken_id}.slurm'

        with open(slurm_job_file, 'w', encoding='utf-8') as o:
            o.write(text)

        subprocess.run(args=['sbatch', str(slurm_job_file.name)], cwd=slurm_job_file.parent, check=False)

if __name__ == "__main__":
    raise NotImplementedError('This script is unused')
    main()