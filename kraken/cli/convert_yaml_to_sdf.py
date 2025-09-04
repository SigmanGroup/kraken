#!/usr/bin/env python3
# coding: utf-8

'''
For converting confdata yamls from Kraken into SDF files
'''

import sys
import yaml
import logging

from pathlib import Path
from yaml import CLoader as Loader

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds

from rdkit import Chem
from rdkit.Geometry import Point3D


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
    Chem.SanitizeMol(mol)

    return mol

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
    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Define path for saving SDF files
    desination_folder = Path('./initial_data/conformer_sdfs')

    files = sorted([x for x in Path('./initial_data/DFT_yamls/').glob('*_confdata.yml')])

    for file in files:

        logger.info('Working on file %s', file.name)

        with open(file, 'r', encoding='utf-8') as f:
            data = yaml.load(f, Loader=Loader)

        # Iterate through the conformer
        for conformer_name, conformer_data_dictionary in data.items():

            elements = conformer_data_dictionary['elements']
            coordinates = conformer_data_dictionary['coords']
            conmat = np.array(conformer_data_dictionary['conmat'])

            mol = mol_from_elements_coords_connectivity(elements=elements,
                                                        coords=coordinates,
                                                        connectivity=conmat)

            write_single_sdf_file(mol=mol, destination=desination_folder / f'{conformer_name}.sdf')

        exit()

if __name__ == "__main__":
    main()