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
import numpy as np
import pandas as pd

from collections import Counter

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

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

    parser.add_argument('-d', '--directory',
                        dest='directory',
                        required=True,
                        type=Path,
                        help='Parent directory that contains multiple <kraken_id> dirs.\n\n',
                        metavar='STR')

    parser.add_argument('--debug', action='store_true', help='Prints debug information\n\n')

    args = parser.parse_args()

    if not args.directory.exists():
        raise FileNotFoundError(f'Could not locate {args.directory.absolute()}.')

    return args

def convert_yml_to_series(data_yml: Path,
                          kraken_id: str,
                          ):
    '''
    Reads in the summary property yml
    '''
    # Read in the summary data file
    with open(data_yml, 'r', encoding='utf-8') as f:
        data = yaml.full_load(f)

    # Make a dictionary to convert the data
    main_dict = {'name': str(kraken_id),
                 'smiles': ''}

    # Boltzmann weighted properties
    boltz = list(data['boltzmann_averaged_data'])
    boltz_values = []
    for value in boltz:
        boltz_desc = data['boltzmann_averaged_data'][value]
        boltz_values.append(boltz_desc)
    boltz_dict = {boltz[i] + str('_boltz'): boltz_values[i] for i in range(len(boltz))}
    main_dict.update(boltz_dict)

    # Delta descriptors
    delta = list(data['delta_data'])
    delta_values = []
    for value in delta:
        delta_desc = data['delta_data'][value]
        delta_values.append(delta_desc)
    delta_dict = {delta[i] + str('_delta'): delta_values[i] for i in range(len(delta))}
    main_dict.update(delta_dict)

    # Max descriptors
    max_data = list(data['max_data'])
    max_data_values = []
    for value in max_data:
        max_desc = data['max_data'][value]
        max_data_values.append(max_desc)
    max_dict = {max_data[i] + str('_max'): max_data_values[i] for i in range(len(max_data))}
    main_dict.update(max_dict)

    # Min descriptors
    min_data = list(data['min_data'])
    min_data_values = []
    for value in min_data:
        min_desc = data['min_data'][value]
        min_data_values.append(min_desc)
    min_dict = {min_data[i] + str('_min'): min_data_values[i] for i in range(len(min_data))}
    main_dict.update(min_dict)

    # Vbur_min_conf descriptors
    vburminconf = list(data['vburminconf_data'])
    vburminconf_values = []
    for value in vburminconf:
        vburminconf_desc = data['vburminconf_data'][value]
        vburminconf_values.append(vburminconf_desc)
    vburminconf_dict = {vburminconf[i] + str('_vburminconf'): vburminconf_values[i] for i in range(len(vburminconf))}
    main_dict.update(vburminconf_dict)

    return pd.Series(data=main_dict, dtype=str)

def mol_from_elements_coords_connectivity(elements: list[str],
                                          coords: list[tuple[float, float, float]] | np.ndarray,
                                          connectivity: np.ndarray) -> Chem.Mol:
    '''
    Construct a sanitized RDKit Mol object from atomic elements, 3D coordinates,
    and a 0/1 connectivity matrix.

    Parameters
    ----------
    elements: list of str
        Atomic symbols

    coords: list of tuple[float, float, float] or np.ndarray
        Atomic coordinates (angstrom)

    connectivity:np.ndarray
        Symmetric 0/1 matrix indicating bonded atom pairs.

    Returns
    -------
    Mol
        RDKit Mol object with inferred bond orders.
    '''
    if isinstance(coords, list):
        coords = np.array(coords)
    if coords.shape[0] != len(elements) or connectivity.shape != (len(elements), len(elements)):
        raise ValueError('Inconsistent dimensions among elements, coords, and connectivity matrix.')

    mol = Chem.RWMol()
    for symbol in elements:
        mol.AddAtom(Chem.Atom(symbol))

    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if connectivity[i, j] == 1:
                mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)

    conf = Chem.Conformer(len(elements))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    mol.AddConformer(conf, assignId=True)

    # Convert to Mol and sanitize (infers bond types, valences, aromaticity)
    mol = mol.GetMol()
    #Chem.SanitizeMol(mol)

    return mol

def main():
    '''
    Main function
    '''
    # Set up logging
    logger = logging.getLogger(__name__)

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

    parent_dir = Path(args.directory)

    # Get a list of directories
    #TODO Write a function that validates if we have a legit kraken_id
    directories = [x for x in parent_dir.glob('*') if x.is_dir() and len(x.stem) == 8]

    list_of_series = []

    for data_dir in directories:
        kraken_id = str(data_dir.stem)

        logger.info('Converting %s', data_dir.absolute())

        if not data_dir.exists():
            raise FileNotFoundError(f'Could not locate {data_dir.absolute()}')

        # Define the two yamls
        confdata_yml = data_dir / f'{kraken_id}_confdata.yml'
        data_yml = data_dir / f'{kraken_id}_data.yml'

        # Get an intial pd.Series which is missing the SMILES
        series = convert_yml_to_series(data_yml,
                                       kraken_id=kraken_id)

        # Read in the confdata to get the smiles
        with open(confdata_yml, 'r', encoding='utf-8') as f:
            data = yaml.full_load(f)

        # Make a list to hold all the smiles
        smiles_list = []

        # Iterate through the conformers to get the SMILES column
        for conformer_name, conformer_data_dictionary in data.items():

            try:
                elements = conformer_data_dictionary['elements']
                coordinates = conformer_data_dictionary['coords']
                conmat = np.array(conformer_data_dictionary['conmat'])

                mol = mol_from_elements_coords_connectivity(elements=elements,
                                                            coords=coordinates,
                                                            connectivity=conmat)

                # Write it to a .xyz file
                xyz_file = data_dir / f'{conformer_name}.xyz'
                Chem.MolToXYZFile(mol, str(xyz_file.absolute()))

                # Convert it to a .mol file
                mol_file = data_dir / f'{conformer_name}.mol'
                cmd = ['obabel', '-ixyz', str(xyz_file.absolute()), '-omol', f'-O{mol_file.absolute()}']
                proc = subprocess.run(args=cmd,
                                    cwd=data_dir,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    check=False)

                if proc.returncode != 0:
                    logger.error('obabel return code was %d for conformer %s', proc.returncode, conformer_name)

                # Read it in (now with the correct bonding info)
                mol = Chem.MolFromMolFile(str(mol_file.absolute()))
                mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol=mol, canonical=True))

                # Append to smiles list
                smiles_list.append(Chem.MolToSmiles(mol=mol, canonical=True))

                # Unlink
                xyz_file.unlink()
                mol_file.unlink()
            except Exception as e:
                logger.error('Could not get SMILES of %s because %s', conformer_name, e)

        if len(smiles_list) == 0:
            logger.error('Could not generate SMILES for %s', kraken_id)
            series['smiles'] = ''

        else:
            # Count SMILES occurences
            counts = Counter(smiles_list)

            # Get the entry with the greatest count
            most_common_smiles, count = counts.most_common(1)[0]

            # Sanity check
            if len(counts) != 1:
                logger.warning('Found multiple SMILES from the conformer data. Possible topology change. %s', str(counts.items()))
                logger.warning('Using %s because it was the most common', most_common_smiles)

            series['smiles'] = most_common_smiles

        list_of_series.append(series)

    df = pd.DataFrame(list_of_series)
    df.set_index('name', inplace=True, drop=True)
    print(df)

    df.to_csv(f'./{parent_dir.resolve().stem}_kraken_descriptors.csv')

if __name__ == "__main__":
    main()