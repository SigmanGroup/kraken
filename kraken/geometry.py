#!/usr/bin/env python3
# coding: utf-8

'''
Holds geometry for certain complexes and does geometry manipulations
'''

import os
import logging
import tempfile
import numpy as np
import scipy.spatial as scsp
import scipy.linalg as scli

from pathlib import Path

from typing import Literal

from numpy.typing import NDArray

from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField

from morfeus import BuriedVolume, read_xyz

from kraken.utils import get_num_bonds_P, add_Hs_to_P
from kraken.utils import add_to_smiles, remove_complex
from kraken.structure_generation import get_coords_from_smiles
from kraken.file_io import write_xyz

logger = logging.getLogger(__name__)

def get_Ni_CO_3() -> tuple[NDArray, list[str], Literal[0], Literal[1]]:
    '''
    Return Cartesian coordinates, elements, metal_index, and P_index
    for a tetrahedral Ni(CO)₃P model complex.

    Returns
    -------
    tuple
        coords: ndarray of shape (8, 3)
            XYZ coordinates in ångströms.  Row 0 is the Ni atom.

        elements: list[str]
            Atomic symbols in the same order as *coords*.

        metal_idx: Literal[0]
            Index of the nickel atom (always 0).

        p_idx: Literal[1]
            Index of the phosphorus atom (always 1).
    '''

    # In the old version, the capitalized atomic symbols are used
    elements = ['NI', 'P', 'C', 'O', 'C', 'O', 'C', 'O']

    # Coordinate array (it is unclear where this was acquired)
    coords = np.array([[-2.05044275300666, 0.06382544955011, 0.09868120676498],
                       [-2.80714796997979, -1.10266971180507, -1.69574169412280],
                       [-2.69200378269657, -0.76605024888162, 1.57419568293391],
                       [-3.04257804499007, -1.20995335174270, 2.55963300719774],
                       [-2.69223663646763, 1.74898458637508, -0.06255834794434],
                       [-3.04279673881760, 2.82969533618590, -0.06960307962299],
                       [-0.24189533762829, 0.01881947327896, 0.02959721559736],
                       [0.89275735454075, 0.05117679841698, 0.07869727019190],
        ]
    )

    return coords, elements, 0, 1

def get_Pd_NH3_Cl_Cl():
    crest_best=""" Pd         -1.89996002172552   -0.02498444632011    2.10982622577294
 N          -1.56965112209091   -2.05219215877655    2.00001618954387
 As          -2.21595829857879    2.00450177777031    2.22007410905701
 H          -2.40942129799767    2.36875215537164    1.28398287161819
 H          -3.02318569399418    2.18955283434028    2.82011004940424
 H          -1.37353382758245    2.44891664756754    2.59391276210718
 Cl          0.35060095551484    0.32532669157403    2.26937306191342
 Cl         -4.15039897250316   -0.37607926860031    1.97331323844022"""
    coords=[]
    elements=[]
    for line in crest_best.split("\n"):
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    coords=np.array(coords)
    pd_idx=0
    p_idx=1
    return(coords, elements, pd_idx, p_idx)


def get_Pd_PH3_Cl_Cl():
    crest_best=""" Pd        -0.0000038844        0.0000159819        0.0000111133
 P         -1.6862635579       -1.4845823545        0.0000219312
 As         1.6863052034        1.4845534610        0.0000263723
 H          1.5596931931        2.8713746717        0.0001941369
 H          2.5992646617        1.3913133533       -1.0337086367
 H          2.5995574579        1.3910615548        1.0334736685
 Cl        -1.8219820508        1.3831400099       -0.0000386628
 Cl         1.8219489915       -1.3831565314       -0.0000388596"""
    coords=[]
    elements=[]
    for line in crest_best.split("\n"):
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    coords=np.array(coords)
    pd_idx=0
    p_idx=1
    return(coords, elements, pd_idx, p_idx)


def get_Pd_Cl_Cl():
    crest_best=""" Pd        -0.0000038844        0.0000159819        0.0000111133
 P         -1.6862635579       -1.4845823545        0.0000219312
 Cl        -1.8219820508        1.3831400099       -0.0000386628
 Cl         1.8219489915       -1.3831565314       -0.0000388596"""
    coords=[]
    elements=[]
    for line in crest_best.split("\n"):
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    coords=np.array(coords)
    pd_idx=0
    p_idx=1
    return(coords, elements, pd_idx, p_idx)


def rotationMatrix(vector,angle):
    angle=angle/180.0*np.pi
    norm=(vector[0]**2.0+vector[1]**2.0+vector[2]**2.0)**0.5
    direction=vector/norm

    matrix=np.zeros((3,3))
    matrix[0][0]=direction[0]**2.0*(1.0-np.cos(angle))+np.cos(angle)
    matrix[1][1]=direction[1]**2.0*(1.0-np.cos(angle))+np.cos(angle)
    matrix[2][2]=direction[2]**2.0*(1.0-np.cos(angle))+np.cos(angle)

    matrix[0][1]=direction[0]*direction[1]*(1.0-np.cos(angle))-direction[2]*np.sin(angle)
    matrix[1][0]=direction[0]*direction[1]*(1.0-np.cos(angle))+direction[2]*np.sin(angle)

    matrix[0][2]=direction[0]*direction[2]*(1.0-np.cos(angle))+direction[1]*np.sin(angle)
    matrix[2][0]=direction[0]*direction[2]*(1.0-np.cos(angle))-direction[1]*np.sin(angle)

    matrix[1][2]=direction[1]*direction[2]*(1.0-np.cos(angle))-direction[0]*np.sin(angle)
    matrix[2][1]=direction[1]*direction[2]*(1.0-np.cos(angle))+direction[0]*np.sin(angle)

    return(matrix)

def replace(c1_i, e1_i, c2_i, e2_i,  Au_index, P_index, match_Au_index, match_P_index, smiles, rotate_third_axis=True):

    # copy all the initial things to not change the original arrays
    c1=np.copy(c1_i)
    e1=np.copy(e1_i)
    c2=np.copy(c2_i)
    e2=np.copy(e2_i)

    clash_dist=1.0

    # shift the ligand
    c1-=c1[P_index]
    # shift the ferrocene
    c2-=c2[match_P_index]

    # rotate He-P-axis of ligand
    dir1=c1[Au_index]-c1[P_index]
    dir1/=scli.norm(dir1)
    dir2=np.array([0.0,1.0,0.0])
    dir2/=scli.norm(dir2)
    if np.abs(1.0-np.sum(dir1*dir2))>1e-3:
        cross_dir1_dir2=np.cross(dir1,dir2)
        cross_dir1_dir2/=scli.norm(cross_dir1_dir2)
        angle=np.arccos(np.sum(dir1*dir2))/np.pi*180.0
        rotation=rotationMatrix(cross_dir1_dir2, angle)
        coords_rotated=[]
        for atom in c1:
            coords_rotated.append(np.dot(rotation, atom).tolist())
        c1=np.array(coords_rotated)


    # rotate P-He_replacement-axis of ligand
    dir1=c2[match_Au_index]-c2[match_P_index]
    dir1/=scli.norm(dir1)
    dir2=np.array([0.0,1.0,0.0])
    dir2/=scli.norm(dir2)
    if np.abs(1.0-np.sum(dir1*dir2))>1e-3:
        cross_dir1_dir2=np.cross(dir1,dir2)
        cross_dir1_dir2/=scli.norm(cross_dir1_dir2)
        angle=np.arccos(np.sum(dir1*dir2))/np.pi*180.0
        rotation=rotationMatrix(cross_dir1_dir2, angle)
        coords_rotated=[]
        for atom in c2:
            coords_rotated.append(np.dot(rotation, atom).tolist())
        c2=np.array(coords_rotated)
    #c2+=np.array([0.0,0.7,0.0])
    # rotatble bonds to P
    #print(smi1)
    #rot_bonds=get_rotatable_bonds(smi1)
    #print(rot_bonds)
    #print(Au_index, P_index)


    if rotate_third_axis:
        # rotate third axis
        axis2=np.array([0.0,1.0,0.0])
        axis2/=scli.norm(axis2)
        #min_dist_opt=1.0
        min_best=clash_dist
        angle2_best=None

        all_steps=[]
        all_elements=[]
        for angle2 in np.linspace(0.0,360.0,361):
            rotation2=rotationMatrix(axis2, angle2)
            # shift to zero
            coords_rotated2=[]
            for atom in c2:
                coords_rotated2.append(np.dot(rotation2, atom))
            coords_rotated2=np.array(coords_rotated2)

            all_steps.append(np.copy(coords_rotated2))
            all_elements.append(e2)

            # shift back
            mask1=np.ones((len(c1)))
            mask1[Au_index]=0
            mask1[P_index]=0
            mask2=np.ones((len(c2)))
            mask2[match_Au_index]=0
            mask2[match_P_index]=0
            indeces1=np.where(mask1==1)[0]
            indeces2=np.where(mask2==1)[0]
            min_dist=np.min(scsp.distance.cdist(c1[indeces1],coords_rotated2[indeces2]))

            if min_dist>min_best: #min_dist>min_dist_opt and
                min_best=min_dist
                angle2_best=angle2
                #print("found better RMSD: %f"%(RMSD_best))

        if angle2_best == None:
            #print("FAILED")
            print("ERROR: Did not find a good rotation angle without clashes! %s"%(smiles))
            return(False,None,None)


        rotation2=rotationMatrix(axis2, angle2_best)
        # shift to zero
        coords_rotated_final=[]
        for atom in c2:
            coords_rotated_final.append(np.dot(rotation2, atom))
        c2=np.array(coords_rotated_final)


    c_final=[]
    e_final=[]
    c2_final=[]
    e2_final=[]
    for idx in range(len(c1)):
        if idx!=P_index:
            c_final.append(c1[idx].tolist())
            e_final.append(e1[idx])
    for idx in range(len(c2)):
        if idx!=match_Au_index:
            c_final.append(c2[idx].tolist())
            e_final.append(e2[idx])
            c2_final.append(c2[idx].tolist())
            e2_final.append(e2[idx])

    c_final=np.array(c_final)

    #all_steps.append(np.copy(c2_final))
    #all_elements.append(["K" for e in e2_final])

    #all_steps.append(np.copy(c_final))
    #all_elements.append(e_final)


    #exportXYZs(all_steps,all_elements,"group_rotation.xyz")

    e_final=[str(x) for x in e_final]
    return(True, c_final, e_final)

def mirror_mol(mol: Mol):
    '''
    Generate the mirror image of a 3D RDKit molecule.

    This function assumes that the input molecule contains a
    single conformer. It reflects the Cartesian coordinates
    through the origin to mirror the molecule.

    Parameters
    ----------
    mol : Mol
        An RDKit Mol object

    Returns
    -------
    Mol
        A new RDKit Mol with mirrored coordinates
    '''

    read_write_mol = Chem.RWMol(mol)

    # Get the first conformer
    conformer = read_write_mol.GetConformers()[0]

    # Get the coordinates
    cartesians = np.array(conformer.GetPositions())

    # Mirror the coordinates
    cartesians_mirrored = -cartesians

    # Set the new atomic positions
    for i in range(read_write_mol.GetNumAtoms()):
        conformer.SetAtomPosition(i, Geometry.Point3D(cartesians_mirrored[i][0], cartesians_mirrored[i][1], cartesians_mirrored[i][2]))

    # Conver the molecule and return
    mol = read_write_mol.GetMol()

    Chem.AssignAtomChiralTagsFromStructure(mol)

    return mol

def get_lower_energy_conformer(mol: Chem.Mol,
                               nconfs: int = 50) -> Chem.Mol:
    '''
    Runs a quick RDKit conformer search, optimizes them, then evaluates
    the energy of them to produce a lower energy conformation of the molecule.

    Parameters
    ----------
    mol: Chem.Mol
        The input molecule.

    nconfs : int, optional, default=50
        Number of conformers to generate.

    Returns
    ----------
    Chem.Mol
        The molecule with the lowest energy conformation.
    '''
    # Set the conf search parameters
    params = AllChem.ETKDGv3()

    # Use all available threads
    params.numThreads = 0

    # Prune similar conformers
    params.pruneRmsThresh = 0.5

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=nconfs, params=params)

    min_energy = float('inf')
    min_conf_id = None

    for conf_id in conf_ids:
        ff = UFFGetMoleculeForceField(mol, confId=conf_id)
        ff.Minimize()
        energy = ff.CalcEnergy()

        if energy < min_energy:
            min_energy = energy
            min_conf_id = conf_id

    if min_conf_id is not None:

        # Extract the lowest-energy conformer before modifying the molecule
        low_energy_conf = mol.GetConformer(min_conf_id)

        # Create a copy to prevent modifying the original
        mol = Chem.Mol(mol)
        mol.RemoveAllConformers()
        mol.AddConformer(low_energy_conf, assignId=True)

    return mol

def get_binding_geometry_of_ligand(smiles: str,
                                   coordination_distance: float = 2.1,
                                   nconfs=10) -> tuple[NDArray, list]:
    '''
    THIS IS UNUSED, BUT COULD BE USED IN A FUTURE UPDATE

    Uses RDKit and MORFEUS to generate a set of coordinates and corresponding
    atomic symbols (coords, elements) for a monophosphine ligand that will
    better accomodate the Ni(CO)3 geometry. This is accomplished by generating
    a series of conformations using RDKit and testing the BuriedVolume at some
    distance away from the phosphorus atom.

    Using this metric, the lowest Vbur conformation is selected.

    Parameters
    ----------
    smiles: str
        SMILES string for the monophosphine

    coordination_distance: float (default=1.8)
        Distance from the phosphorus to the dummy metal atom

    Returns
    -------
    coords, elements
    '''
    pass

    # Parse the SMILES string into an RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)

    # Add explicit hydrogen atoms
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates via distance geometry
    AllChem.EmbedMolecule(mol)

    # Get a lower energy conformer
    mol = get_lower_energy_conformer(mol=mol, nconfs=5)

    # Set the conf search parameters
    params = AllChem.ETKDGv3()

    # Use all available threads
    params.numThreads = 0

    # Prune similar conformers
    params.pruneRmsThresh = 0.5

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=nconfs, params=params)

    # Get the indices of the atoms bound to phosphorus
    p_atom = [x for x in mol.GetAtoms() if x.GetSymbol() == 'P']
    if len(p_atom) != 1:
        raise ValueError(f'Could not locate P atom in {smiles}')
    p_atom = p_atom[0]

    # Get the bonds to phosphorus
    p_bonds = [x for x in mol.GetBonds() if p_atom.GetIdx() in (x.GetBeginAtom().GetIdx(), x.GetEndAtom().GetIdx())]

    # Get the atoms bound directly to phosphorus
    bound_atoms = [[bond.GetBeginAtom(), bond.GetEndAtom()] for bond in p_bonds]

    # Flatten the list of lists (by removing the P atom itself since it's included in the bond)
    bound_atoms = [x for xs in bound_atoms for x in xs if x.GetSymbol() != 'P']

    # Storing the progress here
    min_vbur = float('inf')
    min_conf_id = None

    # Iterate through the conformers
    for conf_id in conf_ids:

        ff = UFFGetMoleculeForceField(mol, confId=conf_id)
        ff.Minimize()
        energy = ff.CalcEnergy()

        # Get the array of positions
        positions = np.array([mol.GetConformer(id=conf_id).GetAtomPosition(x.GetIdx()) for x in bound_atoms])

        # Compute the geometric centroid of the substituents
        centroid = np.mean(positions, axis=0)

        # Get the vector that points from centroid to phosphorus
        direction = mol.GetConformer(id=conf_id).GetAtomPosition(p_atom.GetIdx()) - centroid
        direction_unit_vector = direction / np.linalg.norm(direction)

        # Make"metal" position by adding the direction vector to the phosphorus atom pos
        metal_pos = (coordination_distance * direction_unit_vector) + mol.GetConformer(id=conf_id).GetAtomPosition(p_atom.GetIdx())

        # Make a writable mol
        rw_mol = Chem.RWMol(mol, confId=conf_id)
        conf = rw_mol.GetConformer()

        # Add an atom
        new_idx = rw_mol.AddAtom(Chem.Atom(2))
        conf.SetAtomPosition(new_idx, Point3D(*metal_pos))

        _mol = rw_mol.GetMol()

        #with open('tmp.xyz', 'a') as o:
        #    o.write(Chem.MolToXYZBlock(_mol))

        with tempfile.NamedTemporaryFile('w+', suffix='.xyz', delete=True) as f:
            f.write(Chem.MolToXYZBlock(mol=_mol))
            f.flush()
            elements, coordinates = read_xyz(f.name)

        # Compute the vbur
        bv = BuriedVolume(elements=elements,
                          coordinates=coordinates,
                          radius=5,
                          metal_index=new_idx + 1,) # Have to add one because MORFEUS uses 1-indexing

        bv = bv.fraction_buried_volume

        if bv < min_vbur:
            logger.info('Found new lower Vbur conformer when constructing geometry %.3f', bv)
            min_vbur = bv
            min_conf_id = conf_id

    # Pick the winning conformations
    final_conformation = mol.GetConformer(min_conf_id)

    # Create a copy to prevent modifying the original
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(final_conformation, assignId=True)

    with tempfile.NamedTemporaryFile('w+', suffix='.xyz', delete=True) as f:
            f.write(Chem.MolToXYZBlock(mol=new_mol))
            f.flush()
            elements, coordinates = read_xyz(f.name)

    return coordinates, elements


def perform_pdcl5_complexation_to_get_metal_complexation_geometry(kraken_id: str,
                                                                  smiles: str,
                                                                  conversion_method: str,
                                                                  mol_dir: Path,
                                                                  spacer_smiles: str = '[Pd]([Cl])([Cl])([Cl])([Cl])[Cl]',
                                                                  ) -> tuple[NDArray, list]:
    '''
    In the original Kraken workflow for conformer generation, the SMILES
    string analyzed to determine the number of bonds to phosphorus and then
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
        logger.critical('Try rerunning with a different conversion method, or the same one and hoping to get lucky.')
        raise ValueError(f'number of removed atoms is {difference}, but should be 6 for Pd(Cl)5. Saved file to {Path(mol_dir / f"{kraken_id}_failed_complex_in_generate_xyz_atom_no_difference.xyz").absolute()}')

    return coords_ligand, elements_ligand

