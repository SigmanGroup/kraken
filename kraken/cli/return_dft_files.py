#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: James Howard, PhD
# Affiliation: Department of Chemistry, The University of Utah
# Date: 2025-04-10

'''
Takes files from a single directory and places them
in the appropriate <KRAKEN_ID>/dft/ directory.
'''

import re
import sys
import shutil
import logging
import argparse

from pathlib import Path

KRAKEN_CALC_FILE_EXTENSIONS = ['.com', '.gjf', '.log', '.chk', '.fchk', '.wfn', '.error', '.output', '.slurm', '.slr']

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


        Kolossal viRtual dAtabase for moleKular dEscriptorsclear
                     of orgaNophosphorus ligands.

                              CLI SCRIPT

        This script moves DFT files (.com, .chk, .log, .wfn)
       from a generic directory used to run calculations and
         places the resultant files into Kraken calculation
                 that are specified by <KRAKEN_ID>/dft/



              """

def get_args() -> argparse.Namespace:
    '''Gets the arguments for running Kraken'''

    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, 2, 80),
        allow_abbrev=False,
        usage=argparse.SUPPRESS,
        add_help=False)

    parser.add_argument('-h', '--help',
                        action='help',
                        default=argparse.SUPPRESS,
                        help='Show this help message and exit.\n\n')

    parser.add_argument('-i', '--input',
                        dest='input',
                        required=True,
                        type=Path,
                        help='Input directory that contains the .com, .chk, .log, .wfn files.\n',
                        metavar='DIR')

    parser.add_argument('-d', '--destination',
                        dest='destination',
                        required=True,
                        type=Path,
                        help='Parent path that contains directories oranized like <KRAKEN_ID>/dft/\n',
                        metavar='DIR')

    parser.add_argument('--skip-kraken-ids',
                        dest='skip_kraken_ids',
                        required=False,
                        nargs='+',
                        type=str,
                        help='(UNUSED) Skip Kraken IDs that contain these substrings. (OPTIONAL)\n\n',
                        metavar='DIR')

    parser.add_argument('--dry',
                        action='store_true',
                        help='Disables movement of files for debugging purposes.\n\n')

    args = parser.parse_args()

    # Do some path checking
    args.input = Path(args.input)
    if not args.input.exists():
        raise FileNotFoundError(f'The directory {args.input.absolute()} does not exist.')
    if not args.input.is_dir():
        raise NotADirectoryError(f'{args.input.absolute()} is not a directory.')

    args.destination = Path(args.destination)
    if not args.destination.exists():
        raise FileNotFoundError(f'The directory {args.destination.absolute()} does not exist.')
    if not args.destination.is_dir():
        raise NotADirectoryError(f'{args.destination.absolute()} is not a directory.')

    return args

def main():
    '''
    Main function
    '''
    logger = logging.getLogger(__name__)

    args = get_args()

    # Make the inputs Path objects
    calc_dir = Path(args.input)
    destination = Path(args.destination)

    # Get all the files
    files = [x for x in calc_dir.glob('*') if x.is_file() and x.suffix in KRAKEN_CALC_FILE_EXTENSIONS]

    logger.debug('There are %d files to move in total', len(files))

    # Get the coms for debugging
    com_files = [x for x in calc_dir.glob('*.com') if x.is_file()]
    logger.debug('There are %d .com files to move', len(com_files))

    # Get all the stems
    stems = [x.stem for x in files]

    # Remove common strings in kraken
    stems = [re.sub(r'_sp_ra', '', x) for x in stems]
    stems = [re.sub(r'_sp_rc', '', x) for x in stems]
    stems = [re.sub(r'_sp_solv', '', x) for x in stems]

    # Remove the conformer suffix
    kraken_ids = [re.sub(r'_noNi_\d\d\d\d\d', '', x) for x in stems]
    kraken_ids = [re.sub(r'_Ni_\d\d\d\d\d', '', x) for x in kraken_ids]

    # Remove kraken IDs that have a period in them
    # This is sometimes an artifact of .slurm, .error
    # and .output files from slurm where the numbers
    # after the . are the job ID from slurm
    kraken_ids = [x for x in kraken_ids if '.' not in x]

    kraken_ids = sorted(list(set(kraken_ids)))

    logger.debug('Kraken IDs to move: %s', str(kraken_ids))

    # Iterate through the kraken IDs
    for kraken_id in kraken_ids:

        # Get the stems of all the .com and .log files
        # We're doing the file check because who knows
        # what strange stuff Kraken will do
        files_to_move = [x for x in files if kraken_id in x.name]

        # Find a destination to move the files
        dest = destination / kraken_id / 'dft'

        logger.debug('Found %d files to move for %s to destination %s', len(files_to_move), kraken_id, dest.relative_to(destination))

        if not dest.exists():
            logger.error('%s does not exist', dest.absolute())
            continue

        for file in files_to_move:
            if args.dry:
                logger.info('DRY RUN Moving %s to %s', file.name, dest.relative_to(destination))
            else:
                shutil.move(file, dest / file.name)

if __name__ == "__main__":
    main()
