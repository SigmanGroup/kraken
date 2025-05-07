#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: James Howard, PhD
# Affiliation: Department of Chemistry, The University of Utah
# Date: 2025-04-10

'''
Example submission script for submitting a batch
of Kraken conformer searches
'''

import re
import sys
import logging
import argparse
import subprocess

from pathlib import Path
from .run_kraken_conf_search import _parse_csv, _correct_kraken_id

SLURM_TEMPLATE = Path(__file__).parent.parent / 'slurm_templates/conf_search_slurm_template.sh'

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
of Kraken conformer search calculations to the Sigman
group Notchpeak owner nodes.
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

    parser.add_argument('--debug', action='store_true', help='Prints debug information\n\n')

    args = parser.parse_args()

    args.csv = Path(args.csv)
    if not args.csv.exists():
        raise FileNotFoundError(f'Could not locate {args.csv.absolute()} does not exist.')

    return args

def write_job_file(kraken_id: str,
                   smiles: str,
                   conversion_flag: int,
                   directory: Path,
                   time: int,
                   destination: Path,
                   template: Path,
                   nprocs: int,
                   mem: int) -> Path:
    '''
    Writes the job file
    '''
    with open(template, 'r', encoding='utf-8') as infile:
        text = infile.read()

    text = re.sub(r'\$KID', str(kraken_id), text)
    text = re.sub(r'\$SMILES', str(smiles), text)
    text = re.sub(r'\$CALCDIR', str(directory.absolute()), text)
    text = re.sub(r'\$CONVERSION_FLAG', str(conversion_flag), text)
    text = re.sub(r'\$NPROCS', str(nprocs), text)
    text = re.sub(r'\$MEM', str(mem), text)
    text = re.sub(r'\$TIME', str(time), text)
    text = re.sub(r'\$DIR', str(directory.absolute()), text)

    with open(destination, 'w', encoding='utf-8') as o:
        o.write(text)

    return destination

def main() -> None:
    '''Main entry point'''

    # Get the args
    args = get_args()

    # Get the input file
    input_file = Path(args.csv)

    ids, inputs, conversion_flags = _parse_csv(input_file)

    ids = [_correct_kraken_id(x) for x in ids]

    calc_dir = Path(args.calc_dir)
    calc_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Iterate over the input
    for id, smiles, conversion_flag in zip(ids, inputs, conversion_flags):
        dest = Path(f'./{id}_conf.slurm')

        # Write the jobfile
        jobfile = write_job_file(kraken_id=id,
                                 smiles=smiles,
                                 conversion_flag=conversion_flag,
                                 directory=calc_dir,
                                 time=args.time,
                                 destination=dest,
                                 template=SLURM_TEMPLATE,
                                 nprocs=args.nprocs,
                                 mem=args.mem)

        # Submit it to SLURM
        subprocess.run(args=['sbatch', jobfile.name], cwd=jobfile.parent, check=False)

if __name__ == "__main__":
    main()




