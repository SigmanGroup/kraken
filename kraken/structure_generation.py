#!/usr/bin/env python3
# coding: utf-8

'''
Structure generation
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import annotations

import re
import logging
import subprocess

from pathlib import Path

import numpy as np

from numpy.typing import NDArray

import scipy.spatial as scsp

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def get_coords_from_smiles_from_rdkit(smiles: str) -> tuple[NDArray, NDArray] | tuple[None, None]:
    '''
    Generates 3D coordinates and atomic elements from a SMILES string using RDKit.

    Attempts to:
    - Parse the SMILES string to an RDKit molecule.
    - Add explicit hydrogens.
    - Generate a 3D conformation via embedding.
    - Extract element symbols atomic coordinates from the resulting mol block.

    Includes a sanity check to reject geometries where all atoms are clustered
    unrealistically close to the molecular centroid.

    Parameters
    ----------
    smiles : str
        SMILES string representing the molecule.

    Returns
    -------
    coords : NDArray or None
        (N, 3) array of atomic coordinates, or None if generation failed.

    elements : NDArray or None
        (N,) array of atomic element symbols as strings, or None if generation failed.
    '''

    # Parse the SMILES string into an RDKit molecule object
    try:
        m = Chem.MolFromSmiles(smiles)
    except Exception as e:
        logger.error('Could not convert smiles with RDKit %s because %s', smiles, str(e))
        return None, None

    # Add explicit hydrogen atoms
    try:
        m = Chem.AddHs(m)
    except Exception as e:
        logger.error('Could not add hydrogens to smiles with RDKit %s because %s', smiles, str(e))
        return None, None

    # Generate 3D coordinates via distance geometry
    try:
        AllChem.EmbedMolecule(m)
    except Exception as e:
        logger.error('Could not embed smiles with RDKit %s because %s', smiles, str(e))
        return None, None

    try:
        # Convert to MOL block format to extract coordinates and element symbols
        block = Chem.MolToMolBlock(m)
        blocklines = block.split('\n')

        coords = []
        elements = []

        # Parse atomic coordinate lines (start from line 4; stop at bond block)
        for line in blocklines[4:]:
            if len(line.split()) == 4:  # Start of bond block
                break
            x, y, z, elem = line.split()[0:4]
            elements.append(elem)
            coords.append([float(x), float(y), float(z)])

        coords = np.array(coords)
        mean = np.mean(coords, axis=0)

        # Compute distances from geometric center to each atom
        distances = scsp.distance.cdist([mean], coords)[0]

        # Reject if all atoms are unrealistically close to center (bad geometry)
        if np.max(distances) < 0.1:
            logger.error('Max distance between atoms is %f', np.max(distances))
            return None, None

    except Exception as e:
        logger.error('Could not compute distances because %s', str(e))
        return None, None

    return coords, np.array(elements)

def get_coords_from_smiles_from_obabel(smiles: str) -> tuple[NDArray, NDArray] | tuple[None, None]:
    '''
    Generates 3D coordinates and atomic elements from a SMILES string using Open Babel.

    Uses the `obabel` command-line tool to:
    - Convert a SMILES string to 3D XYZ format with hydrogens added.
    - Extract atomic coordinates and element symbols from the XYZ output.
    - Perform a sanity check to discard degenerate conformers.

    Parameters
    ----------
    smiles : str
        SMILES string representing the molecule.

    Returns
    -------
    coords : NDArray or None
        (N, 3) array of atomic coordinates, or None if generation failed.
    elements : NDArray or None
        (N,) array of atomic element symbols as strings, or None if generation failed.
    '''

    # Run Open Babel to convert SMILES to 3D XYZ format with hydrogens
    cmd = ['obabel', f'-:{smiles}', '-oxyz', '--gen3d', '-h']
    proc = subprocess.run(args=cmd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          check=False)
    stdout = proc.stdout.decode('utf-8')
    stderr = proc.stderr.decode('utf-8')

    # Check if conversion succeeded based on output message
    if '1 molecule converted' not in stderr:
        raise ValueError(f'Could not convert SMILES {smiles} to MolBlock with obabel')

    # Parse the XYZ output block
    blocklines = stdout.split('\n')
    coords = []
    elements = []

    # Parse atomic lines (skip first two lines: atom count and comment)
    for line in blocklines[2:]:
        line = re.sub(r'\s+', ' ', line)
        if len(line.split()) != 4:
            break
        el, x, y, z = line.split(' ')
        elements.append(el)
        coords.append([float(x), float(y), float(z)])

    # Convert to NumPy array and compute distances from centroid
    coords = np.array(coords)
    mean = np.mean(coords, axis=0)
    distances = scsp.distance.cdist([mean], coords)[0]

    # Reject geometries where all atoms are tightly clustered
    if np.max(distances) < 0.1:
        logger.error('Max distance between atoms is %f', np.max(distances))
        return None, None

    return coords, np.array(elements)
