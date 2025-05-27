#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: James Howard, PhD
# Affiliation: Department of Chemistry, The University of Utah
# Date: 2025-04-10

'''
Example submission script for submitting a batch
of Kraken DFT processing calculations
'''

import re
import sys
import logging
import argparse
import subprocess

from pathlib import Path
from importlib.resources import files

from .run_kraken_conf_search import _parse_csv, _correct_kraken_id

SLURM_TEMPLATE = files("kraken") / "slurm_templates" / "dft_slurm_template.slurm"

logger = logging.getLogger(__name__)

DESCRIPTION = r'''
╔══════════════════════════════════════╗
║   | |/ / _ \  /_\ | |/ / __| \| |    ║
║   | ' <|   / / _ \| ' <| _|| .` |    ║
║   |_|\_\_|_\/_/ \_\_|\_\___|_|\_|    ║
╚══════════════════════════════════════╝
Kolossal viRtual dAtabase for moleKular dEscriptors
of orgaNophosphorus ligands.


CLI SCRIPT

This is an example CLI script for submitting a batch
of Kraken DFT processing calculations to the Sigman group
Notchpeak owner nodes.
'''

DESCRIPTION = '\n'.join(line.center(80) for line in DESCRIPTION.strip('\n').split('\n'))

def get_args() -> argparse.Namespace:
    '''Gets the arguments for running Kraken'''

    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, 2, 40),
        allow_abbrev=False,
        add_help=False)

    parser.add_argument('-h', '--help',
                        action='help',
                        default=argparse.SUPPRESS,
                        help='Show this help message and exit.\n\n')

    parser.add_argument('-c', '--csv',
                        dest='csv',
                        required=True,
                        help='CSV file that contains the required input.\n\n',
                        metavar='STR')

    parser.add_argument('-n', '--nprocs',
                        dest='nprocs',
                        default=4,
                        help='Number of processors to request\n\n',
                        metavar='INT')

    parser.add_argument('-m', '--mem',
                        dest='mem',
                        default=8,
                        help='Amount of memory in GB to request\n\n',
                        metavar='INT')

    parser.add_argument('-t', '--time',
                        dest='time',
                        default=8,
                        help='Time requested in hours\n\n',
                        metavar='INT')

    parser.add_argument('--calculation-dir',
                        dest='calc_dir',
                        default='./data/',
                        help='Path to the directory that will contain the results of Kraken\n\n',
                        metavar='DIR')

    parser.add_argument('--slurm-template',
                        dest='slurm_template',
                        help='Formatted SLURM template with placeholders\n\n',
                        metavar='STR')

    parser.add_argument('--force',
                        action='store_true',
                        help='Forces the recalculation instead of reading potentially incomplete results from file\n\n')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Prints debug information\n\n')

    args = parser.parse_args()

    args.csv = Path(args.csv)
    if not args.csv.exists():
        raise FileNotFoundError(f'Could not locate {args.csv.absolute()}.')

    args.calc_dir = Path(args.calc_dir)
    if not args.calc_dir.exists():
        raise FileNotFoundError(f'{args.calc_dir.absolute()} does not exist.')

    if args.slurm_template is not None:
        args.slurm_template = Path(args.slurm_template)
        if not args.slurm_template.exists():
            raise FileNotFoundError(f'Could not locate {args.slurm_template.absolute()} does not exist.')

    return args

def write_dft_job_file(kraken_id: str,
                       directory: Path,
                       destination: Path,
                       time: int,
                       template: Path,
                       nprocs: int,
                       mem: int,
                       force: bool = False) -> Path:
    '''
    Writes the job file
    '''
    with open(template, 'r', encoding='utf-8') as infile:
        text = infile.read()

    text = re.sub(r'\$KID', str(kraken_id), text)
    text = re.sub(r'\$TIME', str(time), text)
    text = re.sub(r'\$NPROCS', str(nprocs), text)
    text = re.sub(r'\$MEM', str(mem), text)
    text = re.sub(r'\$CALCDIR', str(directory.absolute()), text)

    if force:
        text = text.strip() + '\n                  --force\n'

    with open(destination, 'w', encoding='utf-8') as o:
        o.write(text)

    return destination

def main() -> None:
    '''Main entry point'''

    # Get the args
    args = get_args()

    # Get the input file
    input_file = Path(args.csv)

    # Get the slurm template if specified
    if args.slurm_template is not None:
        slurm_template = Path(args.slurm_template)
    else:
        slurm_template = Path(SLURM_TEMPLATE)

    ids, inputs, conversion_flags = _parse_csv(input_file)

    ids = [_correct_kraken_id(x) for x in ids]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Iterate over the input
    for id, smiles, conversion_flag in zip(ids, inputs, conversion_flags):
        dest = Path(f'./{id}_dft.slurm')

        # Write the jobfile
        jobfile = write_dft_job_file(kraken_id=id,
                                     directory=args.calc_dir,
                                     destination=dest,
                                     time=args.time,
                                     template=slurm_template,
                                     nprocs=args.nprocs,
                                     mem=args.mem,
                                     force=args.force)

        # Submit it to SLURM
        subprocess.run(args=['sbatch', jobfile.name], cwd=jobfile.parent, check=False)

if __name__ == "__main__":
    main()