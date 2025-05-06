
import numpy as np

import itertools

from pathlib import Path

from rdkit import Chem,Geometry
from rdkit.Chem import rdmolfiles, AllChem, rdMolAlign, rdmolops

def dict_key_rmsd(candidate_pair):
    return float(rmsd_matrix(candidate_pair)[0, 1])

def delete_haloalkane_halides(mol):
    """Remove halides in perhaloalkyl moieties. Match CX2 where both X are the same halide and there is no H at the same carbon, and delete both X from the molecule object."""
    halides = ["F","Cl","Br","I"]
    matches = ()
    for hal in halides:  # this is to avoid matching mixed halides
        matches += mol.GetSubstructMatches(Chem.MolFromSmarts(f"{hal}[CH0]({hal})")) # matches CX3 and --CX2-- . In --CF2Cl, this would match CF2 only, keeping Cl
    if len(matches) == 0:
        return(mol)
    match_atoms = set([i for sub in matches for i in sub]) # this still includes the carbon atoms
    del_halides = [i for i in match_atoms if mol.GetAtomWithIdx(i).GetSymbol() in halides]
    del_halides.sort(reverse=True) # delete from the end to avoid index problems
    e_mol = Chem.EditableMol(mol)
    for atom in del_halides:
        e_mol.RemoveAtom(atom)
    new_mol = e_mol.GetMol()
    return(new_mol)

def mirror_mol(mol0):
    """Create mirror image of the 3D structure in an RDKit molecule object (assumes that one structure/conformer is present in the object)."""
    # Iris Guo
    mol1 = Chem.RWMol(mol0)
    conf1 = mol1.GetConformers()[0]    # assumption: 1 conformer per mol
    cart0 = np.array(conf1.GetPositions())
    cart1 = -cart0
    for i in range(mol1.GetNumAtoms()):
        conf1.SetAtomPosition(i,Geometry.Point3D(cart1[i][0],cart1[i][1],cart1[i][2]))
    mol = mol1.GetMol()
    rdmolops.AssignAtomChiralTagsFromStructure(mol)
    return(mol)

def get_rmsd_matrix(conformers: list[Path]):
        '''
        Takes a list of .sdf files
        '''

        if not all([x.suffix == '.sdf' for x in conformers]):
            raise ValueError(f'Not all file extensions in get_rmsd_matrix conformers were .sdf')

        # Get the mol objects
        molobjects = [next(Chem.SDMolSupplier(str(_.absolute()), removeHs=False, strictParsing=False)) for _ in conformers]

        # Remove all Hs to speed up RMSD calculation
        molobjects = [Chem.RemoveHs(mol) for mol in molobjects]

        # Remove halides in perhaloalkyl moieties. Improves RMSD matching and timing
        molobjects = [delete_haloalkane_halides(mol) for mol in molobjects]

        # Create mirror images of each conformer
        molobjects_inv = [mirror_mol(mol) for mol in molobjects]

        # Make the rmsd matrix
        rmsd_mat = np.zeros((len(conformers), len(conformers)))

        # Do a stupid loop
        for i, j in itertools.product(range(len(conformers)), range(len(conformers))):
            if i<j:
                continue
            if i==j:
                rmsd_mat[i, j] = 1
            else:
                rmsd_mat[i, j] = min((rdMolAlign.GetBestRMS(molobjects[i], molobjects[j]), rdMolAlign.GetBestRMS(molobjects[i], molobjects_inv[j])))
                rmsd_mat[j, i] = rmsd_mat[i, j]

        return rmsd_mat

def find_dupes():
    pass