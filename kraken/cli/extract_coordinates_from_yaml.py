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

import logging
import argparse

from pathlib import Path

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

INSERT INSERT INSERT INSERT INSERT INSERT INSERT INSERT
INSERT INSERT INSERT INSERT INSERT INSERT INSERT INSERT
INSERT INSERT INSERT INSERT INSERT INSERT INSERT INSERT
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

    parser.add_argument('-y', '--yaml',
                        dest='yaml',
                        required=True,
                        help='Number of processors to request\n\n',
                        metavar='INT')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Prints debug information\n\n')

    args = parser.parse_args()

    args.yaml = Path(args.yaml)
    if not args.yaml.exists():
        raise FileNotFoundError(f'Could not locate {args.yaml.absolute()}.')

    return args

def main():
    pass

if __name__ == "__main__":
    # Do this Kraken ID
    kid = '00000401'