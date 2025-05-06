#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: James Howard, PhD
# Affiliation: Department of Chemistry, The University of Utah
# Date: 2025-04-10

'''
Extracts DFT files from Kraken directories and puts them in
one directory for easy submission to CHPC systems.
'''

import sys
import shutil
import logging
import argparse

from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
    datefmt='%m/%d/%Y:%H:%M:%S',  # Correct way to format the date
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

DESCRIPTION = '''
╔══════════════════════════════════════╗
║   | |/ / _ \  /_\ | |/ / __| \| |    ║
║   | ' <|   / / _ \| ' <| _|| .` |    ║
║   |_|\_\_|_\/_/ \_\_|\_\___|_|\_|    ║
╚══════════════════════════════════════╝


Kolossal viRtual dAtabase for moleKular dEscriptors
of orgaNophosphorus ligands.

CLI SCRIPT

This script moves DFT files (.com, .chk, .log, .wfn) from
Kraken calculation directories to a single directory for
easy submission to HPC systems. Specify the directory that
contains subdirectories named with Kraken IDs. These Kraken
ID directories should have the following structure.


<INPUT_DIR>/<KRAKEN_ID>/dft/
'''

DESCRIPTION = '\n'.join(line.center(80) for line in DESCRIPTION.strip('\n').split('\n'))

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
                        help='Input directory that contains the subdirectories <KRAKEN_ID>/dft/\n\n',
                        metavar='DIR')

    parser.add_argument('-d', '--destination',
                        dest='destination',
                        required=True,
                        type=Path,
                        help='Destination to place the files.\n\n',
                        metavar='DIR')

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

    args = get_args()

    calc_dirs = sorted([Path(x) / 'dft' for x in args.input.glob('*') if x.is_dir()])

    destination = Path(args.destination)

    # Iterate through all the dirs the user wants to move files from
    for directory in calc_dirs:

        logger.info('Moving files for %s', directory.absolute())

        # Get the stems of all the .com and .log files
        # We're doing the file check because who knows
        # what strange stuff Kraken will do
        com_files = sorted(list(set([x for x in directory.glob('*.com') if x.is_file()])))

        # Check that the .log files exist for the .com files
        for com in com_files:
            if not com.with_suffix('.log').exists():
                logger.warning('.log file for %s does not exist', com.name)

        log_files = sorted(list(set([x for x in directory.glob('*.log') if x.is_file()])))
        for log in log_files:
            if not log.with_suffix('.com').exists():
                logger.warning('.com file for %s does not exist', log.name)

        # Get all the stems
        stems = [x.stem for x in com_files]
        stems.extend([x.stem for x in log_files])
        stems = sorted(list(set(stems)))

        # Iterate through them
        for stem in stems:

            # Make all of the files we expect from a completed Kraken calculation
            com = directory / f'{stem}.com'
            log = directory / f'{stem}.log'
            chk = directory / f'{stem}.chk'
            wfn = directory / f'{stem}.wfn'
            sp_ra_chk = directory / f'{stem}_sp_ra.chk'
            sp_rc_chk = directory / f'{stem}_sp_rc.chk'
            sp_solv_chk = directory / f'{stem}_sp_solv.chk'

            # Check that they exist and move them
            for file in [com, log, chk, wfn, sp_ra_chk, sp_rc_chk, sp_solv_chk]:
                if not file.exists():
                    #logger.warning('%s does not exist in %s', file.name, directory.resolve())
                    continue

                shutil.move(file, destination / file.name)

if __name__ == "__main__":
    main()
