#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Utility functions for the conformer search portion of Kraken
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import annotations

import re
import math
import logging

from pathlib import Path

import numpy as np

from numpy.typing import NDArray

import scipy.spatial as scsp
import scipy.linalg as scli

from rdkit import Chem

logger = logging.getLogger(__name__)

# Constants
kcal_to_eV = 0.0433641153
kB = 8.6173303e-5 #eV/K
T = 298.15
kBT = kB*T
AToBohr = 1.889725989
ANGSTROM_TO_BOHR = 1.889725989

masses = {
    'H' : 1.008,'HE' : 4.003, 'LI' : 6.941, 'BE' : 9.012,
    'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,
    'F' : 18.998, 'NE' : 20.180, 'NA' : 22.990, 'MG' : 24.305,
    'AL' : 26.982, 'SI' : 28.086, 'P' : 30.974, 'S' : 32.066,
    'CL' : 35.453, 'AR' : 39.948, 'K' : 39.098, 'CA' : 40.078,
    'SC' : 44.956, 'TI' : 47.867, 'V' : 50.942, 'CR' : 51.996,
    'MN' : 54.938, 'FE' : 55.845, 'CO' : 58.933, 'NI' : 58.693,
    'CU' : 63.546, 'ZN' : 65.38, 'GA' : 69.723, 'GE' : 72.631,
    'AS' : 74.922, 'SE' : 78.971, 'BR' : 79.904, 'KR' : 84.798,
    'RB' : 84.468, 'SR' : 87.62, 'Y' : 88.906, 'ZR' : 91.224,
    'NB' : 92.906, 'MO' : 95.95, 'TC' : 98.907, 'RU' : 101.07,
    'RH' : 102.906, 'PD' : 106.42, 'AG' : 107.868, 'CD' : 112.414,
    'IN' : 114.818, 'SN' : 118.711, 'SB' : 121.760, 'TE' : 126.7,
    'I' : 126.904, 'XE' : 131.294, 'CS' : 132.905, 'BA' : 137.328,
    'LA' : 138.905, 'CE' : 140.116, 'PR' : 140.908, 'ND' : 144.243,
    'PM' : 144.913, 'SM' : 150.36, 'EU' : 151.964, 'GD' : 157.25,
    'TB' : 158.925, 'DY': 162.500, 'HO' : 164.930, 'ER' : 167.259,
    'TM' : 168.934, 'YB' : 173.055, 'LU' : 174.967, 'HF' : 178.49,
    'TA' : 180.948, 'W' : 183.84, 'RE' : 186.207, 'OS' : 190.23,
    'IR' : 192.217, 'PT' : 195.085, 'AU' : 196.967, 'HG' : 200.592,
    'TL' : 204.383, 'PB' : 207.2, 'BI' : 208.980, 'PO' : 208.982,
    'AT' : 209.987, 'RN' : 222.081, 'FR' : 223.020, 'RA' : 226.025,
    'AC' : 227.028, 'TH' : 232.038, 'PA' : 231.036, 'U' : 238.029,
    'NP' : 237, 'PU' : 244, 'AM' : 243, 'CM' : 247, 'BK' : 247,
    'CT' : 251, 'ES' : 252, 'FM' : 257, 'MD' : 258, 'NO' : 259,
    'LR' : 262, 'RF' : 261, 'DB' : 262, 'SG' : 266, 'BH' : 264,
    'HS' : 269, 'MT' : 268, 'DS' : 271, 'RG' : 272, 'CN' : 285,
    'NH' : 284, 'FL' : 289, 'MC' : 288, 'LV' : 292, 'TS' : 294,
    'OG' : 294
    }

# This covalent radius data differs from the one in kraken.Kraken_conformer_selection_only.py
# In other file, H rcov = 0.32 whereas here it is 0.34
# In other file, Pd rcov = 1.08 whereas here it is 1.19
rcov = {
    "H": 0.34, "He": 0.46, "Li": 1.2, "Be": 0.94, "B": 0.77,
    "C": 0.75,"N": 0.71,"O": 0.63,"F": 0.64,"Ne": 0.67,"Na": 1.4,
    "Mg": 1.25,"Al": 1.13,"Si": 1.04,"P": 1.1,"S": 1.02,"Cl": 0.99,
    "Ar": 0.96,"K": 1.76,"Ca": 1.54,"Sc": 1.33,"Ti": 1.22,"V": 1.21,
    "Cr": 1.1,"Mn": 1.07,"Fe": 1.04,"Co": 1.0,"Ni": 0.99,"Cu": 1.01,
    "Zn": 1.09,"Ga": 1.12,"Ge": 1.09,"As": 1.15,"Se": 1.1,"Br": 1.14,
    "Kr": 1.17,"Rb": 1.89,"Sr": 1.67,"Y": 1.47,"Zr": 1.39,"Nb": 1.32,
    "Mo": 1.24,"Tc": 1.15,"Ru": 1.13,"Rh": 1.13,"Pd": 1.19,"Ag": 1.15,
    "Cd": 1.23,"In": 1.28,"Sn": 1.26,"Sb": 1.26,"Te": 1.23,"I": 1.32,
    "Xe": 1.31,"Cs": 2.09,"Ba": 1.76,"La": 1.62,"Ce": 1.47,"Pr": 1.58,
    "Nd": 1.57,"Pm": 1.56,"Sm": 1.55,"Eu": 1.51,"Gd": 1.52,"Tb": 1.51,
    "Dy": 1.5,"Ho": 1.49,"Er": 1.49,"Tm": 1.48,"Yb": 1.53,"Lu": 1.46,
    "Hf": 1.37,"Ta": 1.31,"W": 1.23,"Re": 1.18,"Os": 1.16,"Ir": 1.11,
    "Pt": 1.12,"Au": 1.13,"Hg": 1.32,"Tl": 1.3,"Pb": 1.3,"Bi": 1.36,
    "Po": 1.31,"At": 1.38,"Rn": 1.42,"Fr": 2.01,"Ra": 1.81,"Ac": 1.67,
    "Th": 1.58,"Pa": 1.52,"U": 1.53,"Np": 1.54,"Pu": 1.55
    }

def get_ligand_indices(coords: NDArray,
                       elements: NDArray,
                       P_index: int,
                       smiles: str,
                       metal_char: str):
    '''
    Returns mask then bool?
    '''
    bonds = get_bonds(coords,
                      elements,
                      force_bonds=True,
                      forced_bonds=[[P_index, list(elements).index(metal_char)]])

    found = False

    for bondidx, bond in enumerate(bonds):
        elements_bond=[elements[bond[0]],elements[bond[1]]]
        if metal_char in elements_bond and "P" in elements_bond and P_index in bond:
            found = True
            break
    if found:
        indeces1, indeces2 = separate_at_bond(coords, elements, bonds, bondidx, smiles)

        if metal_char == elements_bond[0]:
            mask=indeces2
        else:
            mask=indeces1

        logger.debug('In get_ligand_indices: mask: %s, %s', str(mask), str(type(mask)))

        return mask, True
    else:
        logger.error('No %s-P bond found in smiles %s', metal_char, smiles )

        return None, False

def get_bonds(coords, elements, force_bonds=False, forced_bonds=[]):
    '''
    Gets the bonds of a coords and elements set.
    #TODO Understand what this function does.
    '''

    # Covalent radii, from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197
    # values for metals decreased by 10% according to Robert Paton's Sterimol implementation

    # Partially based on code from Robert Paton's Sterimol script,
    # which based this part on Grimme's D3 code
    # Get the connectivity matrix
    natom = len(coords)
    k1 = 16.0
    k2 = 4.0/3.0
    conmat = np.zeros((natom,natom))
    bonds = []
    for i in range(0,natom):
        if elements[i] not in rcov.keys():
            continue
        for iat in range(0,natom):
            if elements[iat] not in rcov.keys():
                continue
            if iat != i:
                dx = coords[iat][0] - coords[i][0]
                dy = coords[iat][1] - coords[i][1]
                dz = coords[iat][2] - coords[i][2]
                r = np.linalg.norm([dx,dy,dz])
                rco = rcov[elements[i]]+rcov[elements[iat]]
                rco = rco*k2
                rr=rco/r
                damp=1.0/(1.0 + math.exp(-k1*(rr-1.0)))
                if damp > 0.85: #check if threshold is good enough for general purpose
                    conmat[i,iat],conmat[iat,i] = 1,1
                    pair=[min(i,iat),max(i,iat)]
                    if pair not in bonds:

                        # add some empirical rules here:
                        is_bond=True
                        elements_bond = [elements[pair[0]], elements[pair[1]]]
                        if "Pd" in elements_bond:
                            if not ("As" in elements_bond or "Cl" in elements_bond or "P" in elements_bond):
                                is_bond=False
                        elif "Ni" in elements_bond:
                            if not ("C" in elements_bond or "P" in elements_bond):
                                is_bond=False
                        elif "As" in elements_bond:
                            if not ("Pd" in elements_bond or "F" in elements_bond):
                                is_bond=False
                        if is_bond:
                            bonds.append(pair)

    # Remove bonds in certain cases
    bonds_to_remove = []

    # P has too many bonds including one P-Cl bond which is probably to the spacer
    P_bond_indeces = []
    P_bonds_elements = []

    # Iterate through the bonds to find bonds to phosphorus
    for bondidx, bond in enumerate(bonds):

        # Get the elemental symbols that make the bond
        elements_bond = [elements[bond[0]], elements[bond[1]]]

        # If it's bound to P, keep track of it
        if 'P' in elements_bond:
            P_bond_indeces.append(bondidx)
            P_bonds_elements.append(elements_bond)

        # Check for Cl-Cl bonds
        if sorted(['Cl', 'Cl']) == elements_bond:
                bonds_to_remove.append(bondidx)

    # If there are more than 4 bonds to phosphorus
    if len(P_bond_indeces) > 4:
        logger.warning('Found phosphorus with more than 4 bonds (Found %d). Attempting to remove a bond', len(P_bond_indeces))

        # If there is a P-Cl bond, remove it
        if ['P', 'Cl'] in P_bonds_elements:
            bonds_to_remove.append(P_bond_indeces[P_bonds_elements.index(['P', 'Cl'])])

        # If there is a Cl-O bond, remove it
        elif ['Cl', 'O'] in P_bonds_elements:
            bonds_to_remove.append(P_bond_indeces[P_bonds_elements.index(['Cl', 'O'])])

    # Define a list of new bonds that do not include previously removed bonds
    bonds = [bond for idx, bond in enumerate(bonds) if idx not in bonds_to_remove]

    # Very special case where the C atoms of Ni(CO)3 make additional bonds to lone pairs of N
    # Get the indeces of the Ni(CO)3 C bonds
    c_atom_indeces = []

    # Iterate through the currents bonds that aren't scheduled to be removed
    for bondidx, bond in enumerate(bonds):

        # Get the elements involved in the bond
        elements_bond = [elements[bond[0]], elements[bond[1]]]

        # If it is a C-Ni bond
        if 'Ni' == elements_bond[0] and 'C' == elements_bond[1]:

            # Iterate through the bonds again with respect to the C-Ni bond
            for bondidx2, bond2 in enumerate(bonds):

                # Get the elements involved in this second bond
                elements_bond2 = [elements[bond2[0]], elements[bond2[1]]]

                # If the C atom (which is bound to nickel) is in the second bond
                # and that C atom is bound to an oxygen, remove the C-Ni bond
                if bond[1] in bond2 and 'O' in elements_bond2:
                    c_atom_indeces.append(bond[1])
                    break

        # Repeat it for the other order of elements
        elif "Ni" == elements_bond[1] and "C" == elements_bond[0]:
            for bondidx2, bond2 in enumerate(bonds):
                elements_bond2=[elements[bond2[0]],elements[bond2[1]]]
                if bond[0] in bond2 and "O" in elements_bond2:
                    c_atom_indeces.append(bond[0])
                    break

    # Check what this does
    if len(c_atom_indeces) > 0:

        logger.warning('Removing these Ni-carbonyl carbon indices bound to Nitrogen and Nickel: %s', str(c_atom_indeces))

        bonds_to_remove = []

        for c_atom_idx in c_atom_indeces:

            for bondidx, bond in enumerate(bonds):

                elements_bond = [elements[bond[0]], elements[bond[1]]]

                if c_atom_idx in bond and "N" in elements_bond:
                    logger.debug('Found a carbonyl carbon atom %d bound to a nitrogen atom. Removing bond idx %s', c_atom_idx, str(bondidx))
                    bonds_to_remove.append(bondidx)

        # Reset the bonds
        bonds = [bond for idx, bond in enumerate(bonds) if idx not in bonds_to_remove]

    # Add forced bonds
    if forced_bonds:
        for b in forced_bonds:
            b_to_add = [min(b),max(b)]
            if b_to_add not in bonds:
                logger.warning('Force addition of %s-%s bond that was not automatically detected.', elements[b_to_add[0]], elements[b_to_add[1]])
                bonds.append(b_to_add)

    # Add bonds for atoms that are floating around
    indices_used = []
    for b in bonds:
        indices_used.append(b[0])
        indices_used.append(b[1])

    indices_used=list(set(indices_used))

    if len(indices_used) < len(coords):
        for i in range(len(coords)):
            if i not in indices_used:
                e = elements[i]
                c = coords[i]
                distances = scsp.distance.cdist([c],coords)[0]
                next_atom_indices = np.argsort(distances)[1:]
                for next_atom_idx in next_atom_indices:
                    b_to_add = [min([i, next_atom_idx]),max([i, next_atom_idx])]
                    elements_bond=[elements[b_to_add[0]],elements[b_to_add[1]]]
                    if elements_bond not in [["Cl","H"],["H","Cl"],["Cl","F"],["F","Cl"],["F","H"],["H","F"],["Pd","F"],["F","Pd"],["H","H"],["F","F"],["Cl","Cl"]]:
                        logger.warning('Had to add a %s-%s bond that was not detected automatically.', elements[b_to_add[0]], elements[b_to_add[1]])

                        bonds.append(b_to_add)
                        break
                    else:
                        pass
    return bonds

def separate_at_bond(coords, elements, bonds, bondidx, smiles):

    start1=bonds[bondidx][0]
    start2=bonds[bondidx][1]
    dihedral_atoms=[]
    connections1_all=[]
    connections1_to_check=[]
    for bondidx2,bond in enumerate(bonds):
        if bondidx2!=bondidx:
            if start1 == bond[0]:
                connection_new=bond[1]
            elif start1 == bond[1]:
                connection_new=bond[0]
            else:
                continue
            connections1_all.append(connection_new)
            connections1_to_check.append(connection_new)
    if len(connections1_to_check)==0:
        exit("ERROR: no metal-P dihedral found for %s"%(smiles))
    else:
        dihedral_atoms.append(connections1_to_check[0])

    dihedral_atoms.append(start1)
    dihedral_atoms.append(start2)

    while len(connections1_to_check)>0:
        for connection in connections1_to_check:
            for bondidx2,bond in enumerate(bonds):
                if bondidx2!=bondidx:
                    if connection == bond[0]:
                        connection_new=bond[1]
                    elif connection == bond[1]:
                        connection_new=bond[0]
                    else:
                        continue
                    if connection_new not in connections1_all and connection_new not in connections1_to_check:
                        connections1_to_check.append(connection_new)
                        connections1_all.append(connection_new)
            connections1_to_check.remove(connection)

    connections2_all=[]
    connections2_to_check=[]
    for bondidx2,bond in enumerate(bonds):
        if bondidx2!=bondidx:
            if start2 == bond[0]:
                connection_new=bond[1]
            elif start2 == bond[1]:
                connection_new=bond[0]
            else:
                continue
            connections2_all.append(connection_new)
            connections2_to_check.append(connection_new)
    if len(connections2_to_check)==0:
        exit("ERROR: no metal-P dihedral found for %s"%(smiles))
    else:
        dihedral_atoms.append(connections2_to_check[0])

    while len(connections2_to_check)>0:
        for connection in connections2_to_check:
            for bondidx2,bond in enumerate(bonds):
                if bondidx2!=bondidx:
                    if connection == bond[0]:
                        connection_new=bond[1]
                    elif connection == bond[1]:
                        connection_new=bond[0]
                    else:
                        continue
                    if connection_new not in connections2_all and connection_new not in connections2_to_check:
                        connections2_to_check.append(connection_new)
                        connections2_all.append(connection_new)
            connections2_to_check.remove(connection)
    connections1_all=sorted(connections1_all)
    connections2_all=sorted(connections2_all)
    return(connections1_all, connections2_all)

def sanitize_smiles(smiles: str) -> str:
    '''
    Sanitzes and canonicalizes a SMILES string
    '''
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=True), isomericSmiles=False, canonical=True)

def get_dummy_positions(lmocent_coord_file: Path) -> list[list[float]]:
    '''
    Extracts the coordinates (in Bohr) of helium dummy atoms from an xTB
    LMO centers file (lmocent.coord)

    Parameters
    ----------
    lmocent_coord_file: Path
        Path to the lmocent.coord file produced by an xTB calculation.

    Returns
    -------
    list[list[float]]
        Each inner list contains the [x, y, z] coordinates (in Bohr) of a helium dummy atom.

    Raises
    ------
    FileNotFoundError
        If lmocent_coord_file does not exist.
    '''
    dummy_positions = []

    if not lmocent_coord_file.exists():
        raise FileNotFoundError(f'{lmocent_coord_file.absolute()} does not exist.')

    # Read in the lines
    with open(lmocent_coord_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        number_of_raw_lines = len(lines)

    # Strip the lines of new line terminators and spaces
    lines = [x.strip() for x in lines]

    # Replace spaces with a single space
    lines = [re.sub(r'\s+', ' ', x) for x in lines]

    # Split on spaces
    lines = [x.split(' ') for x in lines]

    # Remove everything that has length of 4 (x + y + z + atomic symbol)
    lines = [x for x in lines if len(x) == 4]

    if len(lines) != number_of_raw_lines - 2:
        raise ValueError(f'Found {len(lines)} lines when expecting {number_of_raw_lines - 2}')

    # Convert the carts to Bohr
    lines = [[float(x[0]) / ANGSTROM_TO_BOHR, float(x[1]) / ANGSTROM_TO_BOHR, float(x[2]) / ANGSTROM_TO_BOHR, x[3]] for x in lines]

    # Iterate through the lines
    for l in lines:
        if str(l[3]).casefold() == 'he':
            dummy_positions.append([l[0], l[1], l[2]])

    return dummy_positions

def _str_is_smiles(smiles_or_path: str | Path) -> bool:
    '''
    Determines if a string is a valid smiles or a Path
    '''

    path_check = Path(smiles_or_path)

    if not path_check.exists():
        mol = Chem.MolFromSmiles(smiles_or_path)

        if mol is None:
            return False
        else:
            return True

    return False

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

def overlap(coords1, coords_ref, idx1, idx2, elements):

    coords1_np=np.array(coords1)
    coords_ref_np=np.array(coords_ref)
    #print("overlap: coords1: %i, coords2: %i"%(len(coords1),len(coords_ref)))
    #print(idx1,idx2)

    # shift
    coords_shifted=coords1_np-coords1_np[idx2]+coords_ref_np[idx2]

    # rotate P-dummy-axis
    dir1=coords_shifted[idx1]-coords_shifted[idx2]
    dir1/=scli.norm(dir1)
    dir2=coords_ref_np[idx1]-coords_ref_np[idx2]
    dir2/=scli.norm(dir2)
    cross_dir1_dir2=np.cross(dir1,dir2)
    cross_dir1_dir2/=scli.norm(cross_dir1_dir2)
    angle=np.arccos(np.sum(dir1*dir2))/np.pi*180.0
    rotation=rotationMatrix(cross_dir1_dir2, angle)
    # shift to zero
    coords_shifted-=coords_shifted[idx2]
    coords_rotated=[]
    for atom in coords_shifted:
        coords_rotated.append(np.dot(rotation, atom).tolist())
    coords_rotated=np.array(coords_rotated)
    # shift back
    coords_rotated+=coords_ref_np[idx2]


    # rotate third axis
    axis2=coords_rotated[idx1]-coords_rotated[idx2]
    axis2/=scli.norm(axis2)
    RMSD_best=1e10
    angle2_best=0.0
    for angle2 in np.linspace(0.0,360.0,361):
        rotation2=rotationMatrix(axis2, angle2)
        # shift to zero
        coords_rotated-=coords_rotated[idx2]
        coords_rotated2=[]
        for atom in coords_rotated:
            coords_rotated2.append(np.dot(rotation2, atom))
        coords_rotated2=np.array(coords_rotated2)
        # shift back
        coords_rotated2+=coords_ref_np[idx2]
        RMSD=np.mean((coords_rotated2-coords_ref_np)**2.0)**0.5
        if RMSD<RMSD_best:
            RMSD_best=RMSD
            angle2_best=angle2
            #print("found better RMSD: %f"%(RMSD_best))

    rotation2=rotationMatrix(axis2, angle2_best)
    # shift to zero
    coords_rotated-=coords_rotated[idx2]
    coords_rotated_final=[]
    for atom in coords_rotated:
        coords_rotated_final.append(np.dot(rotation2, atom))
    coords_rotated_final=np.array(coords_rotated_final)
    # shift back
    coords_rotated_final+=coords_ref_np[idx2]
    #exportXYZs([coords_rotated_final,coords_ref_np],[elements+["H"],elements+["H"]],"test.xyz")
    return(coords_rotated_final.tolist())

def reduce_data(data_here: dict) -> tuple[dict, dict, dict]:
    '''
    Takes a dictionary of data produced by run_kraken
    and generates Boltzmann weighted/min/max/stdev descriptors.

    Also deletes some dictionary entries that were deemed unuseful.
    '''
    data_here["boltzmann_averaged_data"] = {}
    data_here["min_data"] = {}
    data_here["max_data"] = {}
    data_here_esp_points={}

    # Get a list of conf names and number of conformers
    confnames=[]
    counter=0
    for key in data_here.keys():
        if "conf_" in key:
            confnames.append(f"conf_{counter}")
            counter+=1

    # Intialize boltzmann stuff
    weights=[]
    energies=[]
    degeneracies=[]

    # Iterate through each conf name
    for confname in confnames:
        weights.append(float(data_here[confname]["boltzmann_data"]["weight"]))
        degeneracies.append(int(data_here[confname]["boltzmann_data"]["degen"]))
        energies.append(float(data_here[confname]["boltzmann_data"]["energy"]) * kcal_to_eV)

    for confname in confnames:
        if not "elements" in data_here:
            data_here["elements"]=data_here[confname]["elements"]

    # own weight calculation for comparison
    # KEEP THIS CODE
    #print(weights)
    #print(np.sum(weights))
    #Z=np.sum(np.array(degeneracies)*np.exp(-np.array(energies)/kBT))
    #weights2=1.0/Z*np.array(degeneracies)*np.exp(-np.array(energies)/kBT)
    #print(weights2)

    # electronic_properties
    #######################
    keys_to_delete = []

    for key in data_here["conf_0"]["electronic_properties"].keys():

        # Process esp_points specially
        if key == "esp_points":

            # List for holding the esp data for each conformer
            data = []

            #
            min_q=-0.2
            max_q=0.2

            # Generate a list of bins and appropriate binwidth
            # This value looks to be hard coded by defining
            # min_q and max_q (-0.2 and 0.2, respectively)
            bins_q = np.linspace(min_q, max_q, 50)
            binwidth = bins_q[1] - bins_q[0]

            # Get the conformer-dependent esp_points data
            for confname in confnames:
                try:
                    xyzq = np.array(data_here[confname]["electronic_properties"][key])
                    histdata=np.histogram(xyzq.T[3],bins=bins_q, density=True)[0]
                    data.append(histdata)
                except IndexError:
                    print(f'\t[WARNING] No esp_point data was found for {confname}. Setting to None.')
                    data.append(None)

                # shift it to the esp points dictionary
                if confname not in data_here_esp_points:
                    data_here_esp_points[confname]={}
                data_here_esp_points[confname][key]=data_here[confname]["electronic_properties"][key]

            # Now get some statistics about it
            if any([_ is None for _ in data]):
                print(f'\t[WARNING] None was in the esp_points list. --esp calculations failed for a conformer.')
            try:
                data=np.array(data)
                data_averaged=np.average(data, weights=weights, axis=0)
                data_std = np.average((data-data_averaged)**2.0, weights=weights, axis=0)**0.5

                data_here["boltzmann_averaged_data"]["esp_hist_bins"] = bins_q.tolist()
                data_here["boltzmann_averaged_data"]["esp_hist"] = data_averaged.tolist()
                data_here["boltzmann_averaged_data"]["esp_hist_std"] = data_std.tolist()
            except ValueError:
                print(f'\t[WARNING] Could not get esp_hist data. Setting to None.')
                data_here["boltzmann_averaged_data"]["esp_hist_bins"] = None
                data_here["boltzmann_averaged_data"]["esp_hist"] = None
                data_here["boltzmann_averaged_data"]["esp_hist_std"] = None

            # Add the esp_points key to the keys to delete (for some reason)
            keys_to_delete.append(key)

        elif key == "esp_profile":

            # We calculate this with fixed bins
            # so we can remove the xtb data
            keys_to_delete.append(key)

        elif key == "dummy_positions":
            keys_to_delete.append(key)

        elif key == "dip":
            # Averaging the dipole does not  sense because it can
            # rotate completely between conformer
            # thus, we also average the norm of the dipole moment
            data = []
            data_norm = []

            # Get the dipole moment for each conformer
            for confname in confnames:
                _dip = data_here[confname]["electronic_properties"][key]
                if _dip == [0.0, 0.0, 0.0]:
                    logger.warning('\tconfname: %s had a dipole of [0, 0, 0]. Did this calculation finish?', confname)

                data.append(data_here[confname]["electronic_properties"][key])

                data_norm.append(np.linalg.norm(data_here[confname]["electronic_properties"][key]))

            # Check for 0 dipole!
            if [0.0, 0.0, 0.0] in data:
                logger.warning('Dipole data contains [0, 0, 0] which may influence the dipole statistics.')

            # Turn them into an array
            data = np.array(data)
            data_norm = np.array(data_norm)

            # Get the average dipole/norm
            data_averaged = np.average(data,weights=weights,axis=0)
            data_averaged_norm = np.average(data_norm, weights=weights, axis=0)

            # Get the min and max norm
            data_min_norm=np.min(data_norm,axis=0)
            data_max_norm=np.max(data_norm,axis=0)

            # Get the stddev
            data_std = np.average((data-data_averaged)**2.0, weights=weights, axis=0)**0.5
            data_norm_std = np.average((data_norm-data_averaged_norm)**2.0, weights=weights, axis=0)**0.5

            # Assign the values to the master data dictionary
            data_here['boltzmann_averaged_data'][key] = data_averaged.tolist()
            data_here['boltzmann_averaged_data'][key + '_std'] = data_std.tolist()
            data_here['boltzmann_averaged_data']['dip_norm'] = data_averaged_norm.tolist()
            data_here['boltzmann_averaged_data']['dip_norm_std'] = data_norm_std.tolist()
            data_here['min_data']['dip_norm'] = data_min_norm.tolist()
            data_here['max_data']['dip_norm'] = data_max_norm.tolist()

        # Else if it's a general electronic property
        else:
            logger.info('Reducing data for %s', key)

            # Lists for holding the values and weights
            data = []
            weights_here = []

            # Collect all the values
            for confidx, confname in enumerate(confnames):

                # Get the value of the property for that particular conformer
                x = data_here[confname]["electronic_properties"][key]

                # If it is not None, add it to the data list
                if x is not None:
                    data.append(x)
                    # Get the weight of this particular conformer and add it to the weights list
                    weights_here.append(weights[confidx])
                else:
                    logger.warning('Missing electronic property %s for confidx: %d confname: %s', str(key), confidx, str(confname))

            if len(data) > 0:
                #if debug:
                #    print(f'[DEBUG] data:\n{data}')
                #    if isinstance(data[0], list):
                #        print([len(x) for x in data])
                data=np.array(data)
                weights_here=np.array(weights_here)
                #print(data)
                #print(data.shape)
                #print(weights_here)
                #print(weights_here.shape)
                data_averaged=np.average(data, weights=weights_here, axis=0)
                data_min=np.min(data, axis=0)
                data_max=np.max(data, axis=0)
                data_std = np.average((data-data_averaged)**2.0, weights=weights_here, axis=0)**0.5
                #print(weights_here)
                #print(data_averaged.tolist(), data_min.tolist(), data_max.tolist())
                data_here["boltzmann_averaged_data"][key]=data_averaged.tolist()
                data_here["boltzmann_averaged_data"][key+"_std"]=data_std.tolist()
                data_here["min_data"][key]=data_min.tolist()
                data_here["max_data"][key]=data_max.tolist()
            # If there is no data, then the calculation or
            # parameter extraction failed for each conformer
            else:
                logger.error('min/max/boltz/stdev properties for %s can not be computed because len(data) = %d', str(key), len(data))

                # Add None values for those properties for which there is no data
                data_here["boltzmann_averaged_data"][key] = None
                data_here["boltzmann_averaged_data"][key + "_std"] = None
                data_here["min_data"][key] = None
                data_here["max_data"][key] = None

    # Delete a bunch of keys
    for key in keys_to_delete:
        for confname in confnames:
            del data_here[confname]["electronic_properties"][key]

    # Begin working on the sterimol parameters
    keys_to_delete = []
    for key in data_here["conf_0"]["sterimol_parameters"].keys():

        # Special handling of certain keys
        if key == "elements_extended":
            elements_extended_list = []
            for confname in confnames:
                elements_extended_list.append(data_here[confname]['sterimol_parameters'][key])
            pass
        elif key == 'selected_dummy_idx':
            keys_to_delete.append(key)
        elif key == 'dummy_idx':
            pass
        elif key == 'p_idx':
            pass
        elif key == 'coords_extended':
            coords_extended_list=[]
            for confname in confnames:
                coords_extended_list.append(data_here[confname]['sterimol_parameters'][key])
            pass

        # If its a general sterimol parameter
        else:
            data = []
            weights_here = []
            for confidx, confname in enumerate(confnames):
                x = data_here[confname]["sterimol_parameters"][key]
                if x is not None:
                    data.append(x)
                    weights_here.append(weights[confidx])
                else:
                    print(f'\t[WARNING] Found {key} of None for conformer {confname}')

            # If we have data to use, get the statistics
            if len(data) > 0:
                data=np.array(data)
                weights_here=np.array(weights_here)
                data_averaged=np.average(data, weights=weights_here, axis=0)
                data_min=np.min(data, axis=0)
                data_max=np.max(data, axis=0)
                data_std = np.average((data-data_averaged)**2.0, weights=weights_here, axis=0)**0.5
                data_here["boltzmann_averaged_data"][key]=data_averaged.tolist()
                data_here["boltzmann_averaged_data"][key + "_std"]=data_std.tolist()
                data_here["min_data"][key]=data_min.tolist()
                data_here["max_data"][key]=data_max.tolist()
            else:
                print(f'\t[WARNING] min/max/boltz/stdev data for {key} can not be computed.')
                data_here["boltzmann_averaged_data"][key]=None
                data_here["boltzmann_averaged_data"][key+"_std"]=None
                data_here["min_data"][key]=None
                data_here["max_data"][key]=None

    # Delete the keys again for some reason
    for key in keys_to_delete:
        for confname in confnames:
            del data_here[confname]["sterimol_parameters"][key]

    # this code averages over the conformers
    # to do so, we first need to overlap the conformers as good as possible
    # for this, we shift and rotate the molecules in a way that their P and dummy atom
    # position are the same then we rotate around the P-dummy axis to minimize the RMSD
    coords_extended_list_rotated=[coords_extended_list[0]]

    for idx, coords_to_turn in enumerate(coords_extended_list[1:]):
        idx1=data_here["conf_0"]["sterimol_parameters"]["dummy_idx"]
        idx2=data_here["conf_0"]["sterimol_parameters"]["p_idx"]
        coords_turned = overlap(coords_to_turn, coords_extended_list_rotated[0], idx1, idx2, elements_extended_list[0])
        confname = f'conf_{idx + 1}'
        data_here[confname]["sterimol_parameters"]["coords_extended"] = coords_turned
        coords_extended_list_rotated.append(coords_turned)

    coords_extended_list_rotated=np.array(coords_extended_list_rotated)

    data_averaged=np.average(coords_extended_list_rotated, weights=weights, axis=0)

    data_std = np.average((coords_extended_list_rotated-data_averaged)**2.0, weights=weights, axis=0)**0.5
    data_here["boltzmann_averaged_data"]["coords_extended"] = data_averaged.tolist()
    data_here["boltzmann_averaged_data"]["coords_extended_std"] = data_std.tolist()

    # Shift and delete some more data
    for confname in confnames:
        data_here[confname]["coords_extended"] = data_here[confname]["sterimol_parameters"]["coords_extended"]

        if "dummy_idx" not in data_here:
            data_here["dummy_idx"] = data_here[confname]["sterimol_parameters"]["dummy_idx"]

        if "p_idx" not in data_here:
            data_here["p_idx"] = data_here[confname]["sterimol_parameters"]["p_idx"]

        del data_here[confname]["coords"]
        del data_here[confname]["sterimol_parameters"]["coords_extended"]
        del data_here[confname]["sterimol_parameters"]["dummy_idx"]
        del data_here[confname]["sterimol_parameters"]["p_idx"]

        #del data_here[confname]["sterimol_parameters"] <--- This was commented out 11 Apr 2024

        data_here[confname]["dip"]=data_here[confname]["electronic_properties"]["dip"]
        del data_here[confname]["electronic_properties"]["dip"]

        #del data_here[confname]["electronic_properties"] <--- This was commented out 11 Apr 2024

        del data_here[confname]["elements"]

    data_here["number_of_conformers"] = len(confnames)
    data_here["boltzmann_weights"] = weights

    # move the conformer data to a separate dictionary
    data_confs = {}
    for confname in confnames:
        data_confs[confname]=data_here[confname]
        del data_here[confname]

    return data_here, data_confs, data_here_esp_points

def get_weights(energies_here, degeneracies_here, selection=[]):
    T_kcal = 0.001987191686486*300.0

    if len(selection)==0:
        selection = np.array(list(range(len(degeneracies_here))))

    # This is the old version that raises an exception because of a negation of a string
    #weights_own = np.array(degeneracies_here)[selection] * np.exp(-np.array(energies_here, dtype=float)[selection]/T_kcal)
    weights_own = np.array(degeneracies_here)[selection].astype(int) * np.exp(-np.array(energies_here)[selection].astype(float)/T_kcal)
    weights_own /= np.sum(weights_own)
    return weights_own

def combine_yaml(molname: str,
                 data_here,
                 data_here_confs) -> dict:
    '''

    '''
    datagroups=["lval","B1","B5","sasa","sasa_P","sasa_volume","cone_angle",
                "global_electrophilicity_index","dip_norm","alpha","EA_delta_SCC",
                "HOMO_LUMO_gap","IP_delta_SCC","nucleophilicity", "cone_angle", "p_int",
                "p_int_atom", "p_int_area", "pyr_val", "pyr_alpha", "qvbur_min", "qvbur_max", "qvtot_min",
                "qvtot_max", "max_delta_qvbur", "max_delta_qvtot", "ovbur_min", "ovbur_max", "ovtot_min",
                "ovtot_max", "vbur", "vtot", "near_vbur", "far_vbur", "near_vtot", "far_vtot"]

    datagroups_vec = ["muls", "wils", "fukui", "alphas"]

    ligand_data = {}

    # read the boltzmann averages results files to get information about each ligand
    #outfilename="%s/%s.yml"%(resultsdir, molname)
    #print("   ---   read molecule %s"%(outfilename))
    #outfile=open(outfilename, "r")
    #data_here=yaml.load(outfile, Loader=yaml.FullLoader)
    #outfile.close()

    ligand_data[molname]={}
    ligand_data[molname]["number_of_atoms"] = len(data_here["elements"])
    ligand_data[molname]["num_rotatable_bonds"] = data_here["num_rotatable_bonds"]
    ligand_data[molname]["number_of_conformers"] = data_here["number_of_conformers"]
    ligand_data[molname]["smiles"] = data_here["smiles"]
    ligand_data[molname]["boltzmann_weights"] = data_here["boltzmann_weights"]
    for key in datagroups:
        ligand_data[molname][key+"_boltzmann"] = data_here["boltzmann_averaged_data"][key]
        ligand_data[molname][key+"_max"] = data_here["max_data"][key]
        ligand_data[molname][key+"_min"] = data_here["min_data"][key]
    p_idx=data_here["p_idx"]
    ligand_data[molname]["p_idx"] = p_idx

    logger.debug('In combine_yaml: p_idx: %s', str(p_idx))

    for key in datagroups_vec:
        logger.debug('In combine_yaml datagroups_vec: KEY: %s', str(key))
        ligand_data[molname][key] = data_here["boltzmann_averaged_data"][key][p_idx]

    # read the conformer results files to get more information about each single conformer
    #outfilename_confs="%s/%s_confs.yml"%(resultsdir, molname)
    #outfile=open(outfilename_confs,"r")
    #data_here_confs=yaml.load(outfile, Loader=yaml.FullLoader)
    #outfile.close()

    n_conformers = ligand_data[molname]["number_of_conformers"]
    energies_here=[]
    degeneracies_here=[]
    weights_here=[]
    for c_idx in range(0,n_conformers):
        energies_here.append(data_here_confs["conf_%i"%(c_idx)]["boltzmann_data"]["energy"])
        degeneracies_here.append(data_here_confs["conf_%i"%(c_idx)]["boltzmann_data"]["degen"])
        weights_here.append(data_here_confs["conf_%i"%(c_idx)]["boltzmann_data"]["weight"])
    ligand_data[molname]["degeneracies"] = degeneracies_here
    ligand_data[molname]["energies"] = energies_here

    weights_own = get_weights(energies_here, degeneracies_here)

    # draw N random conformers (including lowest)
    N_max=10
    N = min(N_max, n_conformers)
    conformers_to_use = np.array([0] + sorted(np.random.choice(list(range(1,n_conformers)), size=N-1, replace=False).tolist()))
    weights_N = get_weights(energies_here, degeneracies_here, selection=conformers_to_use)

    coords_all = []
    elements_all = []
    for c_idx in range(0,n_conformers):
        #print(data_here_confs["conf_%i"%(c_idx)].keys())
        x = data_here_confs["conf_%i"%(c_idx)]["coords_extended"]
        e = data_here_confs["conf_%i"%(c_idx)]["sterimol_parameters"]["elements_extended"]
        coords_all.append(x)
        elements_all.append(e)
        #exportXYZ(x,e,"structures/single_files/%s_conformer_%i.xyz"%(molname, c_idx))
    coords_all = np.array(coords_all)
    #exportXYZs(coords_all,elements_all,"structures/%s_all_conformers.xyz"%(molname))

    ligand_data[molname]["confdata"]={}
    ligand_data[molname]["confdata"]["coords"] = coords_all.tolist()
    ligand_data[molname]["confdata"]["elements"] = elements_all

    electronic_properties = ['EA_delta_SCC', 'HOMO_LUMO_gap', 'IP_delta_SCC', 'alpha', 'alphas', 'global_electrophilicity_index', 'muls', 'nucleophilicity', 'wils']
    sterimol_parameters = ['B1', 'B5', 'lval', 'sasa', 'sasa_P', 'sasa_volume', "cone_angle", "p_int", "p_int_atom", "p_int_area", "pyr_val", "pyr_alpha", "qvbur_min", "qvbur_max", "qvtot_min", "qvtot_max", "max_delta_qvbur", "max_delta_qvtot", "ovbur_min", "ovbur_max", "ovtot_min", "ovtot_max", "vbur", "vtot", "near_vbur", "far_vbur", "near_vtot", "far_vtot"]

    # Go through each electronic property
    for p in electronic_properties:

        # If the property is in a data group
        if p in datagroups:

            # Get a boltzmann weighted reference value?
            feature_ref = ligand_data[molname][p + "_boltzmann"]
        else:
            feature_ref = ligand_data[molname][p]

        # Make sure the feature reference is not None. If it is, do nothing?
        if feature_ref is not None:
            data_here=[]
            mask_here=[]

            # Iterate through each conformer
            for c_idx in range(0, n_conformers):

                # If the property is in the datagroups_vec (i.e., is an atom-level property)
                if p in datagroups_vec:

                    # Get the data for all atoms
                    x = data_here_confs[f'conf_{c_idx}']["electronic_properties"][p]

                    if x is None:
                        logger.warning('\tProperty %s was None for all atoms of conf_%d', p, c_idx)
                        data_here.append(None)
                        continue

                    # Get the value of the property at phosphorus
                    x = data_here_confs[f'conf_{c_idx}']["electronic_properties"][p][p_idx]

                    if x is None:
                        logger.warning('\tProperty %s was None for all atoms of conf_%d', p, c_idx)
                        data_here.append(None)
                        continue

                    # If we have a valid atom-level property that is not None
                    else:
                        # Append the conformer idx to mask_here, and the property to data_here
                        mask_here.append(c_idx)
                        data_here.append(float(x))

                # Otherwise, if it is not an atom-level property
                else:
                    x = data_here_confs[f'conf_{c_idx}']["electronic_properties"][p]

                    if x is None:
                        logger.warning('\tProperty %s was None for conf_%d', p, c_idx)
                        data_here.append(x)
                    else:
                        mask_here.append(c_idx)
                        data_here.append(float(x))

            # Make array mask
            mask_here = np.array(mask_here)

            if len(mask_here) != len(weights_here):
                logger.warning('\tProperty %s could not be computed because the mask is not the same length as the weights.', p)
                ligand_data[molname][p] = None
            else:
                # Converted np.array(weights_here) to dtype of float
                feature_all = np.sum(np.array(data_here) * np.array(weights_here, dtype=float))
                feature_N = np.sum(np.array(data_here)[conformers_to_use]*weights_N)
                #print("%s:\naverage over all (%i): %.3f / %.3f\naverage over %i: %.3f"%(p, n_conformers, feature_all, feature_ref, N, feature_N))
            ligand_data[molname]["confdata"][p]=data_here

    for p in sterimol_parameters:
        if p in datagroups:
            feature_ref = ligand_data[molname][p + "_boltzmann"]
        else:
            feature_ref = ligand_data[molname][p]
        if feature_ref is not None:
            data_here = []
            mask_here = []
            for c_idx in range(0,n_conformers):
                f'conf_{c_idx}'
                x = data_here_confs[f'conf_{c_idx}']["sterimol_parameters"][p]
                if x is None:
                    #print("WARNING: found None in %s of conformer %i"%(p, c_idx))
                    #x = 0.0
                    data_here.append(x)
                else:
                    mask_here.append(c_idx)
                    data_here.append(float(x))
            mask_here = np.array(mask_here)
            if len(mask_here) != len(weights_here):
                logger.warning('Property %s could not be computed because the mask is not the same length as the weights.', p)

                ligand_data[molname][p] = None
            else:
                # Convert the line below to np.array(weights_here, dtype=float) from np.array(weights_here)
                feature_all = np.sum(np.array(data_here) * np.array(weights_here, dtype=float))
                feature_N = np.sum(np.array(data_here)[conformers_to_use]*weights_N)
                #print("%s:\naverage over all (%i): %.3f / %.3f\naverage over %i: %.3f"%(p, n_conformers, feature_all, feature_ref, N, feature_N))
            ligand_data[molname]["confdata"][p] = data_here
        #else:
        #    print("WARNING: %s is None"%(p))

    return ligand_data[molname]

def get_rotatable_bonds(smiles):
    '''
    This function is called.
    '''
    m = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts('[*&!F&!Cl]-&!@[*&!F&!Cl]')
    single_bonds=m.GetSubstructMatches(patt)
    rotatable_bonds=[]
    for x in single_bonds:
        rotatable_bonds.append([x[0],x[1]])
    return rotatable_bonds

def get_num_bonds_P(smiles: str) -> int:
    '''
    Calculate the total bond order (valence) of the **single** phosphorus atom
    present in a SMILES string.

    The bond order is computed as the sum of the numeric bond values
    (single = 1, double = 2, triple = 3, aromatic = 1.5) for every bond
    attached to the first phosphorus atom encountered.

    If RDKit fails to parse the SMILES, the function returns the default
    valence of 3 with a warning message.

    Parameters
    ----------
    smiles: str
        SMILES of a molecule that contains exactly one phosphorus atom.

    Returns
    -------
    int
        Integer bond order (valence) of the selected phosphorus atom.

    Raises
    ------
    ValueError
        If the SMILES contains no phosphorus atom, contains more than one
        phosphorus atom, includes an unsupported bond type, or yields a
        non-integer total bond order.
    '''

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        logger.critical('Could not create mol from %s. Assuming P has 3 bonds.', smiles)
        return 3

    atoms = mol.GetAtoms()
    atoms = [x for x in mol.GetAtoms() if x.GetSymbol() == 'P']

    if len(atoms) == 0:
        raise ValueError(f'SMILES {smiles} does not have a phosphorus atom.')

    elif len(atoms) != 1:
        logger.warning('Found more than one phosphorus atom in %s. Selecting first phosphorus atom.', smiles)

    # Get the phosphous RDKit atom object
    P_atom = atoms[0]

    # Assign the number of bonds to phosphorus
    num_bonds = 0.0

    # Dictionary of bond values that will define
    # the valence of an atom
    bond_values = {
        'SINGLE': 1.0,
        'DOUBLE': 2.0,
        'TRIPLE': 3.0,
        'AROMATIC': 1.5,
    }

    # Iterate through the bonds
    for bond in P_atom.GetBonds():

        # Get the bond type
        bondtype = str(bond.GetBondType())

        if bondtype not in bond_values:
            raise ValueError(f'Unknown bondtype {str(bondtype)}')

        logger.debug('Found a P bond: %s', str(bondtype))

        num_bonds += bond_values[bondtype]

    # Check if there is a non-integer number of bonds
    if abs(num_bonds - round(num_bonds)) > 0.1:
        raise ValueError(f'SMILES {smiles} had {num_bonds} bonds when expecting an integer.')

    return int(num_bonds)

def get_P_bond_indeces_of_ligand(coords, elements):
    bonds = get_bonds(coords, elements)
    #for bond in bonds:
    #    els=[elements[bond[0]],elements[bond[1]]]
    #    if "P" in els and "Pd" in els:
    #        if "P"==les[0]:
    #            P_index=bond[0]
    #        else:
    #            P_index=bond[1]
    #        break
    for P_index, element in enumerate(elements):
        if element=="P":
            break
    bond_indeces=[]
    for bond in bonds:
        idx1=bond[0]
        idx2=bond[1]
        if P_index==idx1:
            #print(idx1,idx2)
            bond_indeces.append(idx2)
        if P_index==idx2:
            #print(idx1,idx2)
            bond_indeces.append(idx1)
    return(P_index, bond_indeces)

def add_Hs_to_P(smiles, num_bonds_P: int):

    if "[P@" in smiles:
        return smiles

    if "P" in smiles:
        P_index = smiles.index("P")
        if P_index > 0:
            if smiles[P_index-1]=="[":
                exit(f'[FATAL] P is already in a square bracket. Cannot add explicit hydrogen atoms {smiles}.')

        if num_bonds_P==3:
            add="[P]"
        elif num_bonds_P==2:
            add="[PH]"
        elif num_bonds_P==1:
            add="[PH2]"
        elif num_bonds_P==0:
            add="[PH3]"
        else:
            add="[P]"
            print(f'[WARNING] Weird number of bonds (num_bonds_P) for P in {smiles}')


    elif "p" in smiles:
        P_index=smiles.index("p")
        if P_index>0:
            if smiles[P_index-1]=="[":
                exit(f'[FATAL] P is already in a square bracket. Cannot add explicit hydrogen atoms {smiles}')
        if num_bonds_P==3:
            add="p"
        elif num_bonds_P==2:
            add="[pH]"
        elif num_bonds_P==1:
            add="[pH2]"
        elif num_bonds_P==0:
            add="[pH3]"
        else:
            add="[p]"
            logger.warning('Weird number of bonds (num_bonds_P) for P in %s. Found %s', smiles, str(num_bonds_P))

    else:
        exit(f'[FATAL] no P or p found in {smiles}')

    p1=smiles[:P_index]
    p2=smiles[P_index+1:]

    smiles_new=p1+add+p2
    return(smiles_new)

def add_to_smiles(smiles, add):
    if "P" in smiles:
        P_index=smiles.index("P")
        if P_index>0:
            if smiles[P_index-1]=="[":

                logger.info('Found "P" in square brackets for SMILES %s', smiles)
                if smiles[P_index+1]=="]":
                    P_index=P_index+1
                elif smiles[P_index+2]=="]":
                    P_index=P_index+2
                elif smiles[P_index+3]=="]":
                    P_index=P_index+3

    elif "p" in smiles:
        P_index=smiles.index("p")
        if P_index>0:
            if smiles[P_index-1]=="[":
                logger.info('Found "P" in square brackets for SMILES %s', smiles)
                if smiles[P_index+1]=="]":
                    P_index=P_index+1
                elif smiles[P_index+2]=="]":
                    P_index=P_index+2
                elif smiles[P_index+3]=="]":
                    P_index=P_index+3
    else:
        logger.error('Could not find "P" or "p" in SMILES %s', smiles)

    p1 = smiles[:P_index + 1]
    p2 = smiles[P_index + 1:]

    return p1 + f"({add})" + p2

def remove_complex(coords: NDArray,
                   elements: NDArray,
                   smiles: str,
                   metal_char: str) -> tuple[NDArray, list] | tuple[None, None]:
    '''
    Returns the coordinate array (angstrom) and the list of elements (list[str])
    '''

    # Get the index of the phosphorus atom
    P_index = list(elements).index("P")

    # Get ligand indices
    mask, done = get_ligand_indices(coords, elements, P_index, smiles, metal_char=metal_char)

    # Return None, None if get ligand indices fails
    if not done:
        return None, None

    # Make a list for holding
    coords_ligand=[]
    elements_ligand=[]

    for atomidx in mask:
        atom = coords[atomidx]
        elements_ligand.append(elements[atomidx])
        coords_ligand.append([atom[0],atom[1],atom[2]])
    coords_ligand=np.array(coords_ligand)

    return coords_ligand, elements_ligand

def get_mass(elements: list[str]) -> float:
    '''
    Gets the mass of a molecule from array of its elemental symbols

    Parameters
    ----------
    elements: ArrayLike
        List of element symbols.

    Returns
    ----------
    float
    '''
    mass = 0.0
    for el in elements:
        mass += masses[el.upper()]
    return mass

if __name__ == "__main__":
    #logfile = Path('/home/sigmanuser/James-Kraken/calculations_Ni/2158_Ni/crest.log')
    #read_crest_log(logfile)
    pass

