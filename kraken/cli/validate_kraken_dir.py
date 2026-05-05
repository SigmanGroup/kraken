#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: James Howard, PhD
# Affiliation: Department of Chemistry, The University of Utah
# Date: 2026-03-27

'''
For validating the structure of kraken directories
'''

import sys
import yaml
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd

from collections import Counter

from kraken.utils import _correct_kraken_id

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

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

This is an example CLI script for validating a Kraken
directory is complete. It performs checks to confirm
the conformer search, DFT calculations, and DFT
processing are complete for each Kraken directory.
'''

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

    parser.add_argument('-d', '--directory',
                        dest='directory',
                        required=True,
                        type=Path,
                        help='Parent directory that contains multiple <kraken_id> dirs.\n\n',
                        metavar='DIR')

    parser.add_argument('--skip-kraken-ids',
                        dest='skip_kraken_ids',
                        required=False,
                        nargs='+',
                        type=str,
                        help='Skip Kraken IDs that contain these substrings.(OPTIONAL)\n\n',
                        metavar='DIR')

    parser.add_argument('--count-only',
                        action='store_true',
                        help='Requests only checking the correct number of .log/.com files exist.\n\n')

    parser.add_argument('--debug', action='store_true', help='Prints debug information\n\n')

    args = parser.parse_args()

    if not args.directory.exists():
        raise FileNotFoundError(f'Could not locate {args.directory.absolute()}.')

    return args


def _check_conformer_yaml_files(data_dir: Path,
                                kraken_id: str) -> None:
    '''
    '''
    # Check the individual conformer files are present
    nickel_confs = data_dir / f'{kraken_id}_Ni_confs.yml'
    nickel_combined = data_dir / f'{kraken_id}_Ni_combined.yml'
    no_nickel_confs = data_dir / f'{kraken_id}_noNi_confs.yml'
    no_nickel_combined = data_dir / f'{kraken_id}_noNi_combined.yml'

    if not nickel_confs.exists():
        logger.warning('nickel_confs file does not exist: %s', nickel_confs.name)
    if not nickel_combined.exists():
        logger.warning('nickel_combined file does not exist: %s', nickel_combined.name)
    if not no_nickel_confs.exists():
        logger.warning('no_nickel_confs file does not exist: %s', no_nickel_confs.name)
    if not no_nickel_combined.exists():
        logger.warning('no_nickel_combined file does not exist: %s', no_nickel_combined.name)


def _check_dft_conf_selection_dir(data_dir: Path,
                                  kraken_id: str) -> list[Path] | None:
    '''
    '''
    # Check if the dft folder exists, if it does, validate
    dft_dir = data_dir / 'dft'
    if not dft_dir.exists():
        logger.warning('DFT directory does not exist: %s', dft_dir.name)
    else:
        # Check if the selected conformer dir exists
        sel_conf = dft_dir / 'selected_conformers'
        if not sel_conf.exists():
            logger.warning('selected_conformers directory does not exist: %s', sel_conf.name)
        else:
            # Get the number of conformers produced from the CREST jobs
            return [x for x in sel_conf.glob('*.xyz') if x.is_file()]


def _find_missing_results(dft_dirs: list[Path],
                          logs: list[Path],
                          sel_confs: list[Path]) -> None:
    '''
    '''

    for conf in sel_confs:

        if conf.stem not in [x.stem for x in dft_dirs]:
            if conf.stem not in [x.stem for x in logs]:
                logger.warning('Missing %s', conf.stem)


def main():
    '''
    Main function
    '''
    # Set up logging
    logger = logging.getLogger(__name__)

    # Get arguments
    args = get_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    parent_dir = Path(args.directory)

    # Get a list of directories
    #TODO Write a function that validates if we have a legit kraken_id
    directories = [x for x in parent_dir.glob('*') if x.is_dir() and _correct_kraken_id(x.stem)]

    if args.skip_kraken_ids is not None:
        to_skip = [str(x) for x in args.skip_kraken_ids]
        directories = [x for x in directories if str(x.stem) not in to_skip]

    # Iterate over the directories
    for data_dir in directories:

        kraken_id = str(data_dir.stem)

        logger.info('Checking %s', data_dir.absolute())

        # Get the number of selected conformers (will use later for checking if the conformer search procedure finished)
        sel_confs = _check_dft_conf_selection_dir(data_dir=data_dir, kraken_id=kraken_id)
        if sel_confs is None:
            n_confs = None
            sel_confs = []
        else:
            n_confs = len(sel_confs)

        if args.count_only:
            dft_dir = data_dir / 'dft'

            if not dft_dir.exists():
                logger.error('DFT directory does not exist.')
                continue

            dft_dirs = [x for x in dft_dir.glob('*') if x.is_dir() and kraken_id in x.name]
            logs = [x for x in dft_dir.glob('*.log') if x.is_file() and kraken_id in x.name]

            logger.debug('Logs: %d\tdft_dirs: %d\tsel_confs: %d', len(logs), len(dft_dirs), len(sel_confs))

            if (len(logs) + len(dft_dirs)) != n_confs:
                nfound = len(logs) + len(dft_dirs)
                logger.warning('The number of conformers (%d) from the DFT procedure do not match the number selected (%d).', nfound, n_confs)
                _find_missing_results(dft_dirs=dft_dirs,
                                      logs=logs,
                                      sel_confs=sel_confs)
            continue


        # Check main files that should be present
        confdata = data_dir / f'{kraken_id}_confdata.yml'
        if not confdata.exists():
            logger.warning('confdata file does not exist: %s', confdata.name)
        data = data_dir / f'{kraken_id}_data.yml'
        if not data.exists():
            logger.warning('data file does not exist: %s', data.name)

        _check_conformer_yaml_files(data_dir=data_dir, kraken_id=kraken_id)

        if n_confs is None:
            logger.warning('Selected confs was None. The conformer search procedure failed.')
        else:
            # Validate that the dft calculations are correct
            dft_dir = data_dir / 'dft'
            dft_dirs = [x for x in dft_dir.glob('*') if x.is_dir() and kraken_id in x.name]
            logs = [x for x in dft_dir.glob('*.log') if x.is_file() and kraken_id in x.name]
            logger.debug('dirs: %d\tlogs: %d\tsel_confs: %d', len(dft_dirs), len(logs), len(sel_confs))

            if len(dft_dirs) == 0:
                logger.warning('No DFT directories were found in %s. The DFT processing has not started.', dft_dir.absolute())
            if len(dft_dirs) != 0 and len(logs) != 0:
                logger.warning('DFT directories and result .log files were found in %s. The DFT processing has incomplete.', dft_dir.absolute())
            if (len(logs) + len(dft_dirs)) != n_confs:
                nfound = len(logs) + len(dft_dirs)
                logger.warning('The number of conformers (%d) from the DFT procedure do not match the number selected (%d).', nfound, n_confs)
                _find_missing_results(dft_dirs=dft_dirs,
                                      logs=logs,
                                      sel_confs=sel_confs)



if __name__ == "__main__":
    main()