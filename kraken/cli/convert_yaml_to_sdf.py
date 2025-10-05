#!/usr/bin/env python3
# coding: utf-8

'''
For converting confdata yamls from Kraken into SDF files
'''

import sys
import yaml
import logging
import argparse
import subprocess

from pathlib import Path
from yaml import CLoader as Loader

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds

from rdkit import Chem

from morfeus.io import write_xyz

DESCRIPTION = r'''
╔══════════════════════════════════════╗
║   | |/ / _ \  /_\ | |/ / __| \| |    ║
║   | ' <|   / / _ \| ' <| _|| .` |    ║
║   |_|\_\_|_\/_/ \_\_|\_\___|_|\_|    ║
╚══════════════════════════════════════╝
Kolossal viRtual dAtabase for moleKular dEscriptors
of orgaNophosphorus ligands.


CLI SCRIPT

This script converts the <kraken_id>_confdata.yml
to .xyz and .sdf files.
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

    parser.add_argument('-f', '--file',
                        dest='file',
                        required=True,
                        type=Path,
                        help='<kraken_id>_confdata.yml file to convert.\n\n',
                        metavar='STR')

    parser.add_argument('--debug', action='store_true', help='Prints debug information\n\n')

    args = parser.parse_args()

    if not args.file.exists():
        raise FileNotFoundError(f'Could not locate {args.file.absolute()}.')

    return args

def write_single_sdf_file(mol,
                          destination: Path) -> None:
    '''

    '''

    if destination.suffix != '.sdf':
        raise ValueError('Destination must be an SDF file (.sdf suffix)')

    with Chem.SDWriter(str(destination.absolute())) as w:
        w.write(mol)

def main():
    '''
    Main function
    '''
    # Get arguments
    args = get_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    confdata_file = Path(args.file)

    # Define path for saving SDF files
    xyz_folder = confdata_file.parent / 'converted_xyz_files'
    sdf_folder = confdata_file.parent / 'converted_sdf_files'
    xyz_folder.mkdir(exist_ok=True)
    sdf_folder.mkdir(exist_ok=True)

    files = sorted([x for x in Path('./initial_data/DFT_yamls/').glob('*_confdata.yml')])

    # Read in file and get elements, coords, and conmat
    with open(confdata_file, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=Loader)

    # Iterate through the conformers, conformer_name is the stem of the file
    for conformer_name, conformer_data_dictionary in data.items():

        elements = conformer_data_dictionary['elements']
        coordinates = conformer_data_dictionary['coords']
        conmat = np.array(conformer_data_dictionary['conmat'])

        # Write the xyz
        xyz_file = xyz_folder / f'{conformer_name}.xyz'
        write_xyz(xyz_file,
                  elements=elements,
                  coordinates=coordinates)

        # Convert with obabel through subprocess to get connectivity
        # This is required since the binary conmat does not reconstruct
        # aromatic and double bonds well
        sdf_file = sdf_folder / f'{conformer_name}.sdf'
        cmd = ['obabel', '-ixyz', str(xyz_file.absolute()), '-osdf', f'-O{sdf_file.absolute()}']
        subprocess.run(args=cmd, cwd=sdf_folder, check=False)

if __name__ == "__main__":
    main()