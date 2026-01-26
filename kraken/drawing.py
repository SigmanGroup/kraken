#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Drawing functions
'''

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from PIL import Image


def draw_molecule_to_file(mol: Chem.Mol,
                          file: Path) -> Path:
    '''
    Draw an RDKit molecule and save it to an image file.

    Parameters
    ----------
    mol: Chem.Mol
        RDKit molecule to draw.

    file_path: Path
        Output file path (e.g., Path("toluene.png")). Parent directories are created if needed.

    Returns
    -------
    saved_path: Path
        Path to the saved image file.
    '''

    # Ensure the output directory exists
    file.parent.mkdir(parents=True, exist_ok=True)

    # Compute 2D coordinates if they are missing
    if mol.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol)

    # Render to a PIL image and save
    img: Image.Image = Draw.MolToImage(mol)
    img.save(str(file.absolute()))

    return file
