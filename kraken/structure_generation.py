#!/usr/bin/env python3
# coding: utf-8

'''
Structure generation
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import annotations

import re
import time
import logging
import subprocess

from pathlib import Path

import numpy as np

from numpy.typing import NDArray

import scipy.spatial as scsp

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType, HybridizationType

from morfeus import read_xyz

from kraken.file_io import write_xyz
from kraken.utils import get_num_bonds_P, add_Hs_to_P
from kraken.utils import add_to_smiles, remove_complex
from kraken.utils import get_P_bond_indeces_of_ligand
from kraken.geometry import get_Ni_CO_3, replace
from kraken.xtb import xtb_opt

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

def get_coords_from_smiles(smiles: str,
                           conversion_method: str) -> tuple[NDArray, NDArray]:
    '''
    Generates a 3D geometry from SMILES using a
    conversion method.

    These are the flags used in the spreadsheet
    0 = RDKit
    1 = Chemaxon
    2 = manual
    3 = obabel
    4 = any (uses obabel then rdkit)

    Parameters
    ----------
    smiles: str
        Smiles to be converted to 3D geometry

    conversion_method: str
        A conversion method to attempt. Acceptable values
        are "rdkit", "obabel", "molconvert", or "any".

    Returns
    ----------
    float | list[float]
    '''

    # If it is any
    if conversion_method == 'any':
        try:
            coords, elements = get_coords_from_smiles_from_obabel(smiles=smiles)
            if (coords is None ) or (elements is None):
                raise ValueError(f'obabel conversion of smiles {smiles} returned None coordinates')
        except Exception as e:
            logger.error('Failed 3D generation with obabel because %s. Trying with RDKit', str(e))
            coords, elements = get_coords_from_smiles_from_rdkit(smiles=smiles)
    elif conversion_method == 'rdkit':
        coords, elements = get_coords_from_smiles_from_rdkit(smiles=smiles)
    elif conversion_method == 'obabel':
        coords, elements = get_coords_from_smiles_from_obabel(smiles=smiles)
    elif conversion_method == 'molconvert':
        raise NotImplementedError(f'molconvert has been deprecated')
    else:
        raise ValueError(f'Could not understand converion method {conversion_method}')

    return np.array(coords), np.array(elements)

def perform_pdcl5_complexation_to_get_metal_complexation_geometry(kraken_id: str,
                                                                  smiles: str,
                                                                  conversion_method: str,
                                                                  mol_dir: Path,
                                                                  spacer_smiles: str = '[Pd]([Cl])([Cl])([Cl])([Cl])[Cl]',
                                                                  ) -> tuple[NDArray, list]:
    '''
    In the original Kraken workflow for conformer generation, the SMILES
    string was analyzed to determine the number of bonds to phosphorus and then
    replace the phosphorus atom in the smiles to contain hydrogens (if there are
    fewer than 3 bonds to phosphorus).

    Then a spacer (originally [Pd]([Cl])([Cl])([Cl])([Cl])[Cl]) is added to
    the modified SMILES string wih add_to_smiles().

    The coordinates of this new complex with the spacer is then generated with the
    get_coords_from_smiles function. The coordinates of the ligand from this
    Pd-bound complex are extracted with the remove_complex function. This
    gets us an initial geometry that should be compatible with the Ni(CO3) template.

    i.e., this function will get a geometry that allows Ni(CO)3 to fit!

    However, it can fail occasionally. A sanity check is included to make sure that
    the number of atoms removed in the remove_complex function was truly 6 - the number
    of atoms in the spacer. This can fail if the complex coordinate generation produces
    a geometry where one of the Cl atoms is placed too closely to an atom in the ligand
    causing the code to determine that the Cl atom is part of the ligand.

    The Cl-<ATOM> bond (where <ATOM> is an atom from the ligand)is interpreted as part
    of the ligand and the Cl atom is not removed. This makes the sanity check fail.

    Example:
    COc1ccc(OC)c(P(c2cc(C(F)(F)F)cc(C(F)(F)F)c2)c2cc(C(F)(F)F)cc(C(F)(F)F)c2)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C
    COc1ccc(OC)c([P](c2cc(C(F)(F)F)cc(C(F)(F)F)c2)c2cc(C(F)(F)F)cc(C(F)(F)F)c2)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)
    COc1ccc(OC)c([P]([Pd]([Cl])([Cl])([Cl])([Cl])[Cl])(c2cc(C(F)(F)F)cc(C(F)(F)F)c2)c2cc(C(F)(F)F)cc(C(F)(F)F)c2)c1-c1c(C(C)C)cc(C(C)C)cc1C(C)C
    '''

    # Get the number of bonds to phosphorus
    num_bonds_P = get_num_bonds_P(smiles)

    logger.debug('num_bonds_P of smiles %s: %d', smiles, num_bonds_P)

    # Add the Hs to smiles phosphorus atom
    # This just adds square brackets if there are 3-bonds to phosphorus
    smiles_Hs = add_Hs_to_P(smiles, num_bonds_P)

    logger.debug('New formatted smiles is %s', smiles_Hs)

    smiles_incl_spacer = add_to_smiles(smiles_Hs, spacer_smiles)

    logger.debug('smiles_incl_space: %s', smiles_incl_spacer)

    coords_ligand_complex, elements_ligand_complex = get_coords_from_smiles(smiles=smiles_incl_spacer,
                                                                            conversion_method=conversion_method)

    # Get the number of atoms in the fake Pd(Cl)5 complex
    num_atoms_with_fake_complex = len(coords_ligand_complex)

    logger.debug('Number of atoms after adding fake complex: %d', num_atoms_with_fake_complex)

    # Remove the complex and get the coordinates of just the ligand (why)
    coords_ligand, elements_ligand = remove_complex(coords=coords_ligand_complex,
                                                    elements=elements_ligand_complex,
                                                    smiles=smiles,
                                                    metal_char='Pd')

    if coords_ligand is None or elements_ligand is None:
        raise ValueError(f'Removal of complex from smiles {smiles} failed to generate coordinates')

    # Get the number of atoms without the fake complex
    num_atoms_without_fake_complex = len(coords_ligand)

    logger.debug('Number of atoms of %s after removing the fake complex: %d', kraken_id, num_atoms_without_fake_complex)

    # Compute the difference
    difference = num_atoms_with_fake_complex - num_atoms_without_fake_complex

    # Sanity check
    if difference != 6:
        write_xyz(destination=Path(mol_dir / f'{kraken_id}_failed_complex_in_generate_xyz_atom_no_difference.xyz'),
                    coords=coords_ligand_complex,
                    elements=elements_ligand_complex,
                    comment='Failed complex generation for Pd(Cl)5 complex',
                    mask=[])
        logger.critical('Failure in making templation complex. This could be from incomplete geometry generation from RDKit/Obabel')
        #logger.critical('Attempting to generate Ni(CO)3 complex directly.')
        logger.critical('Try rerunning with a different conversion method, or the same one and hope to get lucky.')
        raise ValueError(f'number of removed atoms is {difference}, but should be 6 for Pd(Cl)5. Saved file to {Path(mol_dir / f"{kraken_id}_failed_complex_in_generate_xyz_atom_no_difference.xyz").absolute()}')

    return coords_ligand, elements_ligand

def get_nickel_co3_complex_with_replace_method(kraken_id: str,
                                               smiles: str,
                                               conversion_method: str,
                                               charge: int,
                                               mol_dir: Path,
                                               nprocs: int,
                                               metal_char: str = 'Ni',
                                               spacer_smiles: str = '[Pd]([Cl])([Cl])([Cl])([Cl])[Cl]') -> tuple[list[str], np.ndarray]:
    '''
    This is the original Kraken's procedure for generating a


    Returns
    -------
    tuple[list[str], np.ndarray]
        List of atomic symbols and Numpy array of cartesian coordinates.
    '''
    # This will get us a geometry that should be able to accomodate Ni(CO)3
    coords_ligand, elements_ligand = perform_pdcl5_complexation_to_get_metal_complexation_geometry(kraken_id=kraken_id,
                                                                                                   smiles=smiles,
                                                                                                   conversion_method=conversion_method,
                                                                                                   mol_dir=mol_dir,
                                                                                                   spacer_smiles=spacer_smiles)

    # Get the 0-based index of phosphorus and a list of
    # the 0-based indices of the atoms bound to phosphorus
    P_index, bond_indeces = get_P_bond_indeces_of_ligand(coords_ligand, elements_ligand)

    if len(bond_indeces) != 3:
        logger.warning('Number of P-bonds before adding complex was %d instead of 3 for SMILES %s', len(bond_indeces), smiles)

    # Make an empty direction vector
    direction = np.zeros((3))

    # For every atom bound to phosphorus
    for bond_index in bond_indeces:
        direction += (coords_ligand[bond_index] - coords_ligand[P_index])

    direction /= (-np.linalg.norm(direction))
    coords_ligand=np.array(coords_ligand.tolist() + [(coords_ligand[P_index] + 2.25 * direction).tolist()])
    elements_ligand.append(metal_char)
    match_pd_ind=len(elements_ligand) - 1
    match_p_idx = P_index

    coords_pd, elements_pd, pd_idx, p_idx = get_Ni_CO_3()
    success, coords, elements = replace(coords_pd, elements_pd, coords_ligand, elements_ligand, pd_idx, p_idx, match_pd_ind, match_p_idx, smiles, rotate_third_axis=True)

    if elements == None:
        logger.fatal('Elements is None for %s. Exiting gracefully.', smiles)
        exit()
    if len(elements) == 0:
        logger.fatal('Elements is empty for %s. Exiting gracefully.', smiles)
    if not success:
        exit('[FATAL] Pd complexation code to generate coordinates failed. Exiting gracefully.')

    #print(coords[0])
    xtb_scr_dir = mol_dir / 'xtb_scr_dir'
    xtb_scr_dir.mkdir(exist_ok=True)

    logger.info('Optimizing preliminary complex.')

    coords, elements = xtb_opt(coords=coords,
                               elements=elements,
                               smiles=smiles,
                               scratch_dir=xtb_scr_dir,
                               charge=charge,
                               nprocs=nprocs,
                               freeze=[])

    return elements, coords

def fix_ni_dative_directions(mol: Chem.Mol) -> Chem.Mol:
    '''
    Make all Ni-C bonds dative and point donor→acceptor (C -> Ni).

    Parameters
    ----------
    mol: rdkit.Chem.Mol
        Molecule possibly containing Ni-C bonds with wrong direction.

    Returns
    -------
    mol: rdkit.Chem.Mol
        A sanitized copy where Ni-C bonds are BondType.DATIVE with C as begin and Ni as end.
    '''
    # Make a read-write mol
    rw = Chem.RWMol(mol)

    # Collect Ni–C bonds first to avoid mutating while iterating
    bonds_to_flip = []
    for b in mol.GetBonds():

        # Get atoms and symbol
        a0, a1 = b.GetBeginAtom(), b.GetEndAtom()
        sy0, sy1 = a0.GetSymbol(), a1.GetSymbol()

        # If they're incorrectly flipped, note it
        if {'Ni','C'} == {sy0, sy1}:
            ni_idx = a0.GetIdx() if sy0 == 'Ni' else a1.GetIdx()
            c_idx  = a1.GetIdx() if sy0 == 'Ni' else a0.GetIdx()
            bonds_to_flip.append((c_idx, ni_idx, b.GetIdx(), a0.GetIdx(), a1.GetIdx()))

    # Flip the bond direction by removing original bond
    # Then add C -> Ni dative
    for c_idx, ni_idx, bidx, begin_idx, end_idx in bonds_to_flip:
        rw.RemoveBond(begin_idx, end_idx)
        rw.AddBond(int(c_idx), int(ni_idx), BondType.DATIVE)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out

def generate_nickel_carbonyl_complex(kraken_id: str,
                                     smiles: str,
                                     charge: int,
                                     structure_gen_dir: Path):

    # Make a molecule
    mol = Chem.MolFromSmiles(smiles)

    # Define the reaction
    NICKEL_CO3_COMPLEXATION_REACTION = AllChem.ReactionFromSmarts(
    "[#15X3H0:1]([*:2])([*:3])([*:4])."
    "[Ni:5]([C-:6]#[O+:7])([C-:8]#[O+:9])[C-:10]#[O+:11]>>"
    "[#15X3H0:1]([*:2])([*:3])([*:4])->[Ni:5]([C-:6]#[O+:7])([C-:8]#[O+:9])[C-:10]#[O+:11]"
)

    # Get initial product list
    products = NICKEL_CO3_COMPLEXATION_REACTION.RunReactants((mol, Chem.MolFromSmarts('[#28]([#6-]#[#8+])([#6-]#[#8+])[#6-]#[#8+]')))

    # Flatten tuple of tuple of products
    products = [x for xs in products for x in xs]

    logger.debug('There are %d initial products before filtering.', len(products))

    if len(products) < 1:
        raise ValueError(f'Found {len(products)} products when attempting to generate Ni(CO)3 complex for {smiles}')

    # Sanitize
    products = list(set([Chem.MolToSmiles(x) for x in products]))
    products = [Chem.MolFromSmiles(x) for x in products]

    if len(products) != 1:
        raise ValueError(f'Found {len(products)} products when attempting to generate Ni(CO)3 complex for {smiles}')

    # Fix dative bonds
    product = products[0]
    for bond in product.GetBonds():
        atoms = [bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]
        if ('Ni' in atoms) and ('C' in atoms):
            bond.SetBondType(BondType.DATIVE)
        if ('Ni' in atoms) and ('P' in atoms):
            bond.SetBondType(BondType.DATIVE)

    # Correct the Ni hybridization
    for _atom in product.GetAtoms():
        if _atom.GetSymbol() == 'Ni':
            _atom.SetHybridization(HybridizationType.SP3)

    product = Chem.AddHs(product)
    product = fix_ni_dative_directions(mol=product)
    AllChem.EmbedMolecule(product)

    # Write to a file
    init_structure = Path(structure_gen_dir / f'{kraken_id}_Ni_initial_structure.xyz')

    if init_structure.exists():
        logger.warning('%s already exists!', init_structure.absolute())
        init_structure = Path(structure_gen_dir / f'{kraken_id}_Ni_initial_structure_{int(time.time())}.xyz')
        logger.warning('Writing file instead to %s', init_structure.absolute())

    Chem.MolToXYZFile(product, str(init_structure.absolute()))

    logger.debug('Writing preliminary Ni(CO)3 complex to %s', init_structure.absolute())

    # Run the xTB optimization
    cmd = ['xtb', str(init_structure.name), '--opt', 'vtight', '--chrg', str(charge), '--gfn2']
    with open(structure_gen_dir / 'xtb.log', 'w', encoding='utf-8') as o:
        subprocess.run(args=cmd, cwd=structure_gen_dir, stdout=o, stderr=o)

    # Read in the file
    elements, coords = read_xyz(Path(structure_gen_dir / 'xtbopt.xyz').absolute())

    return elements, coords