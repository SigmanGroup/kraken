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

def combine_sdf_files(sdf_files: list[Path], out_sdf: Path) -> None:
    '''
    Combine SDF files by reading them all and writing one output file.

    Parameters
    ----------
    sdf_files: list[Path]
        List of SDF file paths to concatenate in order.

    out_sdf: Path
        Output SDF file path.

    Returns
    -------
    None
    '''
    # Ensure output directory exists
    out_sdf.parent.mkdir(parents=True, exist_ok=True)

    # Open the file to which we will write
    with open(out_sdf, 'w', encoding='utf-8', newline='\n') as out_f:

        # Iterate
        for sdf_file in sdf_files:

            # Get the text
            text = sdf_file.read_text(encoding='utf-8')
            out_f.write(text)

            # Prevent the next file from gluing onto the last line
            if text and not text.endswith('\n'):
                out_f.write('\n')

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

    # Get the kraken ID (<kraken_id>_confdata.yml)
    KID = confdata_file.stem.split('_')[0]

    # Define path for saving SDF files
    xyz_folder = confdata_file.parent / 'converted_xyz_files'
    sdf_folder = confdata_file.parent / 'converted_sdf_files'
    xyz_folder.mkdir(exist_ok=True)
    sdf_folder.mkdir(exist_ok=True)

    # Read in file and get elements, coords, and conmat
    with open(confdata_file, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=Loader)

    written_sdf_files = []

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

        # Change the first line (title line) to the conformer name
        with open(sdf_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            logger.error('No lines were found in %s', str(sdf_file.absolute()))
            continue

        lines[0] = f'{conformer_name}\n'

        with open(sdf_file, 'w', encoding='utf-8') as o:
            o.writelines(lines)

        written_sdf_files.append(sdf_file)

    # Combine all of the xyz and sdf files to a complete
    combine_sdf_files(sdf_files=written_sdf_files, out_sdf=Path(confdata_file.parent / f'{KID}_dft_conformers.sdf'))

if __name__ == "__main__":
    main()