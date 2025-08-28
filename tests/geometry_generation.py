from pathlib import Path

import numpy as np
from kraken.geometry import get_binding_geometry_of_ligand

import logging
import sys
import subprocess

from kraken.file_io import write_xyz
from kraken.geometry import perform_pdcl5_complexation_to_get_metal_complexation_geometry
from kraken.utils import get_P_bond_indeces_of_ligand

from morfeus import read_xyz

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType, HybridizationType

logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

logger = logging.getLogger(__name__)

def earlier_test():
    coords, elements = get_binding_geometry_of_ligand(smiles='c1ccc(cc1)c1nn(c(c1n1nccc1[P](C12CC3CC(C2)CC(C1)C3)C12CC3CC(C2)CC(C1)C3)c1ccccc1)c1ccccc1',
                                                      coordination_distance=1.8,
                                                      nconfs=25)

    write_xyz(Path('./final_geom.xyz'),
              coords=coords,
              elements=elements,
              comment='testing!')

def incomplete_test_of_old_code():
    smiles = 'C[N+](C)(C)CCP(C1CCCCC1)C2CCCCC2'

    coords_ligand, elements_ligand = perform_pdcl5_complexation_to_get_metal_complexation_geometry(kraken_id='01010101',
                                                                                             smiles=smiles,
                                                                                             conversion_method='any',
                                                                                             mol_dir=Path('.'),
                                                                                             spacer_smiles='[Pd]([Cl])([Cl])([Cl])([Cl])[Cl]')

    P_index, bond_indeces = get_P_bond_indeces_of_ligand(coords_ligand, elements_ligand)

    print(P_index, bond_indeces)

    #
    direction = np.zeros((3))

    for bond_index in bond_indeces:
        direction += (coords_ligand[bond_index] - coords_ligand[P_index])

    print(direction)
    print(f'MakeSphere {direction[0]}, {direction[1]}, {direction[2]}')
    direction /= (-np.linalg.norm(direction))
    print(f'MakeSphere {direction[0]}, {direction[1]}, {direction[2]}')

    #print(coords_lig)
    #print(elements_lig)

    write_xyz(Path('./palladium_spacer_complex.xyz'),
              coords=coords_ligand,
              elements=elements_ligand,
              comment='Testing palladium complexation')


def add_nickel_to_smiles():
    pass


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

def generate_nickel_carbonyl_complex(smiles: str,
                                     kraken_id: str,
                                     structure_generation_directory: Path):

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
    Chem.MolToXYZFile(product, str(init_structure.absolute()))

    # Run the xTB optimization
    cmd = ['xtb', str(init_structure.name), '--opt', 'vtight', '--chrg', '0']
    with open(structure_gen_dir / 'xtb.log', 'w', encoding='utf-8') as o:
        subprocess.run(args=cmd, cwd=structure_gen_dir, stdout=o, stderr=o)

    # Read in the file
    elements, coords = read_xyz(Path(structure_gen_dir / 'xtbopt.xyz').absolute())

    return elements, coords

if __name__ == "__main__":

    # How we'll set it up in the code
    smiles = 'C[N+](C)(C)CCP(C1CCCCC1)C1CCCCC1'
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    kraken_id = '01010101'
    nprocs = 4

    # Make sure the mol_dir exists
    mol_dir = Path(f'./{kraken_id}')
    mol_dir.mkdir(exist_ok=True)

    # Make a "structure_gen" directory
    structure_gen_dir = Path(mol_dir / 'structure_gen')
    structure_gen_dir.mkdir(exist_ok=True)

    elements, coords = generate_nickel_carbonyl_complex(smiles=smiles,
                                                        kraken_id=kraken_id,
                                                        structure_generation_directory=structure_gen_dir)



