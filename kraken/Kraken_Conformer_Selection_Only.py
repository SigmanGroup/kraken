#!/usr/bin/env python3
# coding: utf-8

'''
Functions for selecting conformers
based on xTB-derived properties
'''

import os
import time
import math
import uuid
import logging
import subprocess
import itertools

from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy.typing as npt

from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from openbabel import openbabel
obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats('xyz', 'sdf')

logger = logging.getLogger(__name__)

from .gaussian_input import write_coms
from .geometry import mirror_mol
from .ConfPruneIdx import StrictRMSDPrune

# This covalent radius data differs from the one in kraken.utils
# In utils, H rcov = 0.34 whereas here it is 0.32
# In utils, Pd rcov = 1.19 whereas here it is 1.08
CONFORMER_SELECTION_RCOV = {'H': 0.32, 'He': 0.46, 'Li': 1.2, 'Be': 0.94, 'B': 0.77,
                            'C': 0.75, 'N': 0.71, 'O': 0.63, 'F': 0.64, 'Ne': 0.67, 'Na': 1.4,
                            'Mg': 1.25, 'Al': 1.13, 'Si': 1.04, 'P': 1.1, 'S': 1.02, 'Cl': 0.99, 'Ar': 0.96,
                            'K': 1.76, 'Ca': 1.54, 'Sc': 1.33, 'Ti': 1.22, 'V': 1.21, 'Cr': 1.1, 'Mn': 1.07,
                            'Fe': 1.04, 'Co': 1.0, 'Ni': 0.99, 'Cu': 1.01, 'Zn': 1.09, 'Ga': 1.12, 'Ge': 1.09,
                            'As': 1.15, 'Se': 1.1, 'Br': 1.14, 'Kr': 1.17, 'Rb': 1.89, 'Sr': 1.67, 'Y': 1.47,
                            'Zr': 1.39, 'Nb': 1.32, 'Mo': 1.24, 'Tc': 1.15, 'Ru': 1.13, 'Rh': 1.13, 'Pd': 1.08,
                            'Ag': 1.15, 'Cd': 1.23, 'In': 1.28, 'Sn': 1.26, 'Sb': 1.26, 'Te': 1.23, 'I': 1.32,
                            'Xe': 1.31, 'Cs': 2.09, 'Ba': 1.76, 'La': 1.62, 'Ce': 1.47, 'Pr': 1.58, 'Nd': 1.57,
                            'Pm': 1.56, 'Sm': 1.55, 'Eu': 1.51, 'Gd': 1.52, 'Tb': 1.51, 'Dy': 1.5, 'Ho': 1.49,
                            'Er': 1.49, 'Tm': 1.48, 'Yb': 1.53, 'Lu': 1.46, 'Hf': 1.37, 'Ta': 1.31, 'W': 1.23,
                            'Re': 1.18, 'Os': 1.16, 'Ir': 1.11, 'Pt': 1.12, 'Au': 1.13, 'Hg': 1.32, 'Tl': 1.3,
                            'Pb': 1.3, 'Bi': 1.36, 'Po': 1.31, 'At': 1.38, 'Rn': 1.42, 'Fr': 2.01, 'Ra': 1.81,
                            'Ac': 1.67, 'Th': 1.58, 'Pa': 1.52, 'U': 1.53, 'Np': 1.54, 'Pu': 1.55
}

# Define selection schemes to use and which conformer set to apply that to
class SelectionSettings:
    '''Settings for conformer selection'''

    def __init__(self,
                 use_n: int = 1,
                 per_structure_cutoff_limit: int = 20,
                 select_higher_energy: bool = False) -> None:
        '''
        The options for selection schemes are
        select_RMSD     RMSD clustering (default as of 14 Apr 2024)
        select_MinMax   conformers min/maxing properties defined in usepropsminmax (default as of 14 Apr 2024)
        select_random   random selection within energycutoff (untested)
        select_all      all conformers within energycutoff (untested)

        use_n: int (default = 1)
            Number of conformers to select from the high and low end of
            a property distribtion.

        per_structure_cutoff_limit: int
            How many conformers to select

        select_higher_energy: bool (default = False)
            If False, conformers are selected with less than energycutoff
            relative energy. If True, the same number of conformers

        selection_schemes is a dictionary of functions: ['suffix']

        '''

        self.selection_schemes = {select_RMSD: ['noNi'], select_MinMax: ['noNi', 'Ni']}

        self.use_n = use_n

        # These appear to be the properties that are used in the min/max selection
        self.usepropsminmax = ['B1', 'B5', 'lval', 'far_vbur',
                               'far_vtot', 'max_delta_qvbur',
                               'max_delta_qvtot', 'near_vbur',
                               'near_vtot', 'ovbur_max', 'ovbur_min',
                               'ovtot_max', 'ovtot_min', 'pyr_val',
                               'qvbur_max', 'qvbur_min', 'qvtot_max',
                               'qvtot_min', 'vbur'
                              ]

        # for RMSD/random/all
        self.PerStructConfLimit = per_structure_cutoff_limit
        self.InitialRMSDcutoff = 0.5  # RMSD criterion to start the selection with. 0.5 or 1.0 are reasonable values. This is increased in 0.2 steps until the desired number of conformers is left
        self.energycutoff = 3   # select from the relative energy range up to this value in kcal/mol
        self.higher_energy = select_higher_energy

def get_conmat(elements, coords):
    '''
    Get's the connectivity matrix for a coordinate and
    element set. This is partially based on code from
    Robert Paton's Sterimol script, which was adapted
    from Grimme's D3 code.

    Parameters
    ----------
    elements: list[str]
        List of elements that represent a molecule

    coords: np.ndarray
        Array of cartesian coordinates that represent a molecule
        of shape [n_atoms][3]

    Returns
    ----------
    '''

    if type(coords) == list:
        coords = np.asarray(coords)

    natom = len(elements)
    #max_elem = 94
    k1 = 16.0
    k2 = 4.0/3.0
    conmat = np.zeros((natom,natom))
    for i in range(0,natom):
        if elements[i] not in CONFORMER_SELECTION_RCOV.keys():
            continue
        for iat in range(0,natom):
            if elements[iat] not in CONFORMER_SELECTION_RCOV.keys():
                continue
            if iat != i:
                dxyz = coords[iat]-coords[i]
                r = np.linalg.norm(dxyz)
                rco = CONFORMER_SELECTION_RCOV[elements[i]] + CONFORMER_SELECTION_RCOV[elements[iat]]
                rco = rco*k2
                rr=rco/r
                damp=1.0/(1.0+math.exp(-k1*(rr-1.0)))
                # check if threshold is good enough for general purpose
                if damp > 0.85:
                    conmat[i,iat],conmat[iat,i] = 1,1
    return conmat

def write_xyz(file: Path,
              conf,
              data_here,
              writefile: bool = True,
              inverse: bool = False):

    '''
    Writes an xyz file to the provided file Path
    '''
    if inverse:
        geometry_string = "".join([f'{atom:>3} {-data_here["confdata"]["coords"][conf][ind][0]:15f} {-data_here["confdata"]["coords"][conf][ind][1]:15f} {-data_here["confdata"]["coords"][conf][ind][2]:15f}\n' for ind,atom in enumerate(data_here["confdata"]["elements"][0][:-1])])

    else:
        geometry_string = "".join([f'{atom:>3} {data_here["confdata"]["coords"][conf][ind][0]:15f} {data_here["confdata"]["coords"][conf][ind][1]:15f} {data_here["confdata"]["coords"][conf][ind][2]:15f}\n' for ind,atom in enumerate(data_here["confdata"]["elements"][0][:-1])])
    if writefile:
        with open(file, 'w', newline='\n', encoding='utf-8') as f:
            f.write(f'{len(data_here["confdata"]["coords"][conf]) -1 }\n\n')
            f.write(geometry_string)
    return geometry_string

def select_RMSD(suffixes: list[str],
                energies: dict,
                coords_all: dict,
                elements_all: dict,
                properties_all: dict,
                selection_settings: SelectionSettings) -> dict:
    '''
    Selects conformers based on RMSD

    Parameters
    ----------
    suffxes: list[str]
        List of dataset suffixes (e.g., ['noNi', 'Ni'])

    energies: dict
        Dictionary with keys that correspond to the suffixes and the values are
        lists of floats (energies)

    coords_all: dict
        Dictionary with keys that correspond to the suffixes and the values are
        lists of lists of floats where the inner list is for an atom and the floats
        are the x, y, z position of the atom

    elements_all: dict
        Dictionary with keys that correspond to the suffixes and the values are
        lists of strings where the string is the atomic symbol

    properties_all: dict
        Dictionary of the properties (unused in this function)

    selection_settings: SelectionSettings
        Defines selection criteria

    Returns
    ----------
    dict
        A dictionary of form suffix: list[int] where the suffix is
        the original suffix passed, the list of ints are the indices
        of the conformers to be selected
    '''

    rmsdconfs = {}

    energy_cutoff = selection_settings.energycutoff
    per_struct_conf_limit = selection_settings.PerStructConfLimit

    # Iterate through the suffixes (noNi or Ni)
    for suffix in suffixes:

        # Select from lower energy range
        number_of_conformers_below_energy_cutoff = len([i for i in energies[suffix] if i < energy_cutoff])

        logger.info('There are %d conformers < %.2f kcal/mol in %s dataset', number_of_conformers_below_energy_cutoff, float(energy_cutoff), suffix)

        # Redefine the per-structure conformer limit?
        PerStructConfLimit = max((per_struct_conf_limit, int(np.log(number_of_conformers_below_energy_cutoff) ** 2)))

        # If we have more conformers than the limit
        if number_of_conformers_below_energy_cutoff > PerStructConfLimit:

            conformers = []

            # For conformer index i in number_of_conformers_below_energy_cutoff
            for i in range(number_of_conformers_below_energy_cutoff):

                # Define a list for holding the conformer data
                conf = []

                # For atom index in the molecule
                for j in range(len(elements_all[suffix][0])):

                    # Make a conformer from the array of all elements, all coordinates
                    conf.append([elements_all[suffix][0][j], coords_all[suffix][i][j][0], coords_all[suffix][i][j][1], coords_all[suffix][i][j][2]])

                conformers.append(conf)

            # Run the strictRMSDPrune
            # pruned_le are the actual geometries, pruned_indices_le contains the indices of the selected conformers
            pruned_le, pruned_indices_le, actualRMSDcutoff = StrictRMSDPrune(conformers, elements_all[suffix][0], selection_settings.InitialRMSDcutoff, PerStructConfLimit)

        else:
            # Select all conformers if the threshold of number of confs exceeds the
            # number of conformers CREST found
            pruned_indices_le = [i for i in range(number_of_conformers_below_energy_cutoff)]
            logger.info('Selected all %d conformers for %s because structure count limit was %d', len(pruned_indices_le), suffix, PerStructConfLimit)

        # If we're selecting high energy conformers
        if selection_settings.higher_energy:

            # Select from higher energy range
            num_conformers_he = len(energies[suffix]) - number_of_conformers_below_energy_cutoff
            if num_conformers_he > selection_settings.PerStructConfLimit:
                conformers = [[[elements_all[suffix][0][j],coords_all[suffix][i][j][0],coords_all[suffix][i][j][1],coords_all[suffix][i][j][2]] for j in range(len(elements_all[suffix][0]))] for i in range(number_of_conformers_below_energy_cutoff,num_conformers_he)]
                pruned_he, pruned_indices_tmp, actualRMSDcutoff = StrictRMSDPrune(conformers, elements_all[suffix][0], selection_settings.InitialRMSDcutoff, selection_settings.PerStructConfLimit)
                pruned_indices_he = [i+number_of_conformers_below_energy_cutoff for i in pruned_indices_tmp]
            else:
                pruned_indices_he = [i for i in range(number_of_conformers_below_energy_cutoff, num_conformers_he)]
            rmsdconfs[suffix] = pruned_indices_le + pruned_indices_he

        else:
            rmsdconfs[suffix] = pruned_indices_le

    return rmsdconfs

def select_MinMax(suffixes, energies, coords_all, elements_all, properties_all, Sel):
    '''
    Selects conformers that minimize or maximize a property
    Sel: kraken.Kraken_Conformer_Selection_Only.SelectionSettings object
    '''
    # Multilevel index with suffix at the first level and the conformer index at the second level
    properties_df = pd.concat([pd.DataFrame(properties_all[suffix]) for suffix in suffixes], keys=suffixes)

    # Get the series of properties that are not problematic
    nonproblematic_properties = ~properties_df[Sel.usepropsminmax].isna().any()

    # Make sure that these values are all True, otherwise there is a
    # problematic property in there
    if not all(nonproblematic_properties):
        logger.error('In select_MinMax: nonproblematic_properties: %s', str(nonproblematic_properties))

        # Make a new list of nonproblematic_props
        new_nonproblematic = []

        # Iterate through the nonproblematic_properties
        for prop, is_nonproblematic in nonproblematic_properties.items():
            if is_nonproblematic:
                new_nonproblematic.append(prop)
            else:
                logger.error('Cannot select conformers based on %s', prop)

        nonproblematic_properties_names = new_nonproblematic
    else:
        nonproblematic_properties_names = list(nonproblematic_properties.keys())

    #print(f'[DEBUG] nonproblematic_properties_names: {nonproblematic_properties_names}')

    # Get a df of the properties that are used to select conformers
    props_argsort = properties_df.copy(deep=True)
    props_argsort = props_argsort[nonproblematic_properties_names]

    # Replace the properties with the sort order
    props_argsort[nonproblematic_properties_names] = np.argsort(properties_df[nonproblematic_properties_names], axis=0)

    # This allows to pick more than one conformer minimizing/maximizing each property
    # I think this just generates 0 and -1 if you're picking the min and max property
    use = [i for i in range(Sel.use_n)] + [i for i in range(-Sel.use_n, 0)]

    # Absolute indices of the min/max conformers in the Multilevel index (what?)
    absindices = sorted(set(props_argsort.iloc[use].values.reshape(-1)))

    # Indices of the min/max conformers within each ligand set. E.g. [("Ni",4),("noNi",0)]
    setindices = [properties_df.index[i] for i in absindices]

    minmaxconfs = {i: [] for i in suffixes}

    [minmaxconfs[k].append(v) for k,v in setindices]

    return minmaxconfs

def select_all(suffixes, energies, coords_all, elements_all, properties_all,Sel):
    conformers_to_use = {}
    for suffix in suffixes:
        num_conformers_le = len([i for i in energies if i < Sel.energycutoff])
        conformers_to_use[suffix] = [i for i in range(num_conformers_le)]
    return conformers_to_use

def conformer_selection(suffixes: list[str],
                        energies: dict,
                        coords_all: dict,
                        elements_all: dict,
                        properties_all: dict,
                        selection_settings: SelectionSettings) -> dict:
    '''
    Primary function for executing the selection strategy outlined
    by the selection_settings argument.

    Parameters
    ----------
    suffxes: list[str]
        List of dataset suffixes (e.g., ['noNi', 'Ni'])

    energies: dict
        Dictionary with keys that correspond to the suffixes and the values are
        lists of floats (energies)

    coords_all: dict
        Dictionary with keys that correspond to the suffixes and the values are
        lists of lists of floats where the inner list is for an atom and the floats
        are the x, y, z position of the atom

    elements_all: dict
        Dictionary with keys that correspond to the suffixes and the values are
        lists of strings where the string is the atomic symbol

    properties_all: dict
        Dictionary of the properties that are useful for selecting based
        on min/max criteria

    selection_settings: SelectionSettings
        Defines selection criteria

    Returns
    ----------
    tuple[np.ndarray, list]
    '''

    # Get a dictionary of suffix: list for the conformers to use
    # This should end up usually being {'noNi': [], 'Ni': []}
    conformers_to_use = {suffix: [] for suffix in suffixes}

    # Iterate through the selection schema where the scheme is a function that is
    # called to get conformers based on the selection criteria
    for selection_scheme, sel_suffixes in selection_settings.selection_schemes.items():

        logger.info('Selecting conformers using %s on %s', selection_scheme, sel_suffixes)

        # This allows a set of conformers to be missing
        todo_suffixes = list(set(suffixes) & set(sel_suffixes))

        if len(todo_suffixes) == 0:
            todo_suffixes = suffixes

        newconfs = selection_scheme(todo_suffixes,
                                    energies,
                                    coords_all,
                                    elements_all,
                                    properties_all,
                                    selection_settings)

        for k, v in newconfs.items():
            conformers_to_use[k] += v

    # Remove duplicates
    conformers_to_use = {k: sorted(set(v)) for k, v in conformers_to_use.items()}

    return conformers_to_use

def delete_element_from_rdkitmol(mol,element_to_delete):
    """Delete all instances of an element in an RDKit molecule object. Arguments: an RDKit molecule object and a string of the element to be removed"""
    elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    atoms_delete = [i for i in range(len(elements)) if elements[i] == element_to_delete]
    atoms_delete.sort(reverse=True) # delete from the end to avoid index problems
    e_mol = Chem.EditableMol(mol)
    for atom in atoms_delete:
        e_mol.RemoveAtom(atom)
    new_mol = e_mol.GetMol()
    return(new_mol)

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

def write_runpy(molname, maindirectory):
    submit_template="""#!/usr/bin/env python
import os
import sys
import os
import sys
sys.path.append("../../")
import PL_dft_library_201027
import pathlib as pl

idx = int(sys.argv[1])

confname = None
for lineidx, line in enumerate(open("todo.txt","r")):
    if lineidx==idx:
        confname = line.split()[0]
        break
if confname is None:
    print("ERROR: no confname found: %i"%(idx))
    exit()
if not os.path.exists(confname):
    print("ERROR: no confname directory found: %s"%(confname))
    exit()

startdir = os.getcwd()
os.chdir(confname)
actualdir = os.getcwd()
os.system("g16 <%s/%s.com>  %s/%s.log"%(actualdir, confname, actualdir, confname))
os.system("formchk %s/%s.chk %s/%s.fchk"%(actualdir, confname, actualdir, confname))
os.system("vmin4.py %s.fchk"%(confname))   ##changed from vmin3.py 5/7/21 by EP
os.chdir(startdir)


"""
    outfile=open("{}/{}/run.py".format(maindirectory, molname), "w")
    outfile.write(submit_template)
    outfile.close()
    os.system("chmod +x {}/{}/run.py".format(maindirectory, molname))

def write_endpy(molname, maindirectory):
    submit_template="""#!/usr/bin/env python
import os
import sys
import os
import sys
sys.path.append("../../")
import PL_dft_library_201027 ##changed from PL_dft_library 5/17/21 by EP
import pathlib as pl


os.system("rm */*.chk")
os.system("rm log_file_analysis")
os.system("for i in $(find */*.log);do echo $i >> log_file_analysis;grep \\\"Normal termination\\\" $i | wc -l >> log_file_analysis;done;")

startdir = os.getcwd()
ligand_name = startdir.split("/")[-1]

os.chdir("../")
cwd = pl.Path.cwd()
confnames = PL_dft_library.main_split_logs(cwd, ligand_name)
os.chdir(startdir)

confnames_all = [i.name for i in (cwd/ligand_name).iterdir() if i.is_dir() and ligand_name in i.name]
confnames = []
for n in confnames_all:
    err=False
    for x in os.listdir("../ERR"):
        if n in x:
            err=True
            break
    if not err:
        confnames.append(n)
confnames = sorted(confnames)

cwd = pl.Path.cwd()
PL_dft_library.read_ligand(cwd, ligand_name, confnames)

if not os.path.exists("../../dft_results"):
    os.makedirs("../../dft_results")

if os.path.exists("%s_confdata.yml"%(ligand_name)):
    os.system("cp %s_confdata.yml ../../dft_results/."%(ligand_name))

if os.path.exists("%s_data.yml"%(ligand_name)):
    os.system("cp %s_data.yml ../../dft_results/."%(ligand_name))

if os.path.exists("%s_relative_energies.csv"%(ligand_name)):
    os.system("cp %s_relative_energies.csv ../../dft_results/."%(ligand_name))

os.chdir("../")
os.system("zip -rq %s %s"%(ligand_name, ligand_name))
if os.path.exists("%s.zip"%(ligand_name)):
    os.system("rm -rf %s"%(ligand_name))


"""
    outfile=open("{}/{}/end.py".format(maindirectory, molname), "w")
    outfile.write(submit_template)
    outfile.close()
    os.system("chmod +x {}/{}/end.py".format(maindirectory, molname))

def get_rmsd_ext(fn1, fn2):
    #print(fn1)
    #print(fn2)
    randomstring = uuid.uuid4()
    out = "tempfiles/rmsd_%s.out"%(randomstring)
    #print(out)
    #os.system("calculate_rmsd --reorder --no-hydrogen %s %s > %s"%(fn1, fn2, out))
    #os.system("calculate_rmsd --reorder --print %s %s > %s"%(fn1, fn2, out))
    os.system("calculate_rmsd --reorder %s %s > %s"%(fn1, fn2, out))
    rmsd=float(open(out,"r").readlines()[0].split()[0])
    #exit()
    os.system("rm %s"%(out))
    return(rmsd)

def conformer_selection_main(kraken_id: str,
                             save_dir: Path,
                             noNi_datafile: Path,
                             Ni_datafile: Path,
                             nprocs: int):
    print(f'[INFO] Beginning conformer_selection_main with {kraken_id}')

    # Make a string to contain warnings
    warnings = ''

    suffixes = ['noNi', 'Ni']

    # Define a list of files that is the combination of the two
    # separate "combined" files from Kraken
    files = [noNi_datafile, Ni_datafile]

    selection_settings = SelectionSettings()

    if not noNi_datafile.exists():
        raise FileNotFoundError(f'{noNi_datafile.absolute()} does not exist.')
    if not Ni_datafile.exists():
        raise FileNotFoundError(f'{Ni_datafile.absolute()} does not exist.')

    # Make sure the save_dir exists
    if not save_dir.exists():
        raise FileNotFoundError(f'{save_dir.absolute()} does not exist.')

    # Make a subdirectory for storing xyzs
    xyz_dir = save_dir / 'selected_conformers'
    xyz_dir.mkdir(exist_ok=True)

    # Make the tmp_dir
    tmp_dir = save_dir / 'tmp'
    tmp_dir.mkdir(exist_ok=True)

    starttime = time.time()

    # Initialize some dictionaries
    conformers_to_use = {}
    number_of_conformers = {}
    data_here = {}
    energies = {}
    coords_all = {}
    elements_all = {}
    properties_all = {}
    sdffiles = {}
    conmats_all = {}
    conmat_check = {}

    # Iterate through the nickel and no_nickel files
    for file in files:

        # Get the suffix
        #TODO Get rid of using this
        if 'noNi' in file.name:
            suffix = 'noNi'
        else:
            suffix = 'Ni'

        logger.info('Reading results from %s', file.name)

        with open(file, 'r', encoding='utf-8') as infile:
            data_here[suffix] = yaml.load(infile, Loader = yaml.CLoader)

        # Get some data from the loaded yaml file
        number_of_conformers[suffix] = data_here[suffix]["number_of_conformers"]
        energies[suffix] = [float(x) for x in data_here[suffix]["energies"]]
        coords_all[suffix] = data_here[suffix]["confdata"]["coords"]
        elements_all[suffix] = data_here[suffix]["confdata"]["elements"]
        properties_all[suffix] = data_here[suffix]["confdata"]

        # Check for consistent structures that can
        # arise from "reactions" during CREST
        # Get all the conmats
        conmats_all[suffix] = [get_conmat(elements_all[suffix][i][: -1], coords_all[suffix][i][:-1]) for i in range(number_of_conformers[suffix])]
        conmat_check[suffix] = np.zeros((len(conmats_all[suffix]), len(conmats_all[suffix])))

        for i, j in zip(range(len(conmats_all[suffix])), range(len(conmats_all[suffix]))):

            # Different number of atoms means removing Ni-fragment likely failed
            if np.shape(conmats_all[suffix][i]) != np.shape(conmats_all[suffix][j]):
                conmat_check[suffix][i, j] = 1
                conmat_check[suffix][j, i] = 1

            # Differences in connectivity matrices mean bonding changes
            elif np.abs(conmats_all[suffix][i] - conmats_all[suffix][j]).sum() != 0.0:
                conmat_check[suffix][i, j] = 1
                conmat_check[suffix][j, i] = 1

        # Ensure the sum of all checks is zero.
        if np.sum(conmat_check[suffix]) != 0:
            logger.critical('%s_%s has inconsistent structures! Check manually.', kraken_id, suffix)
            warnings += f'{kraken_id}_{suffix} has inconsistent structures! Check manually.'
            return False, warnings

    # If we're doing both the noNi and Ni datasets
    if len(suffixes) == 2:
        conmat_check_cross = np.zeros((len(conmats_all["noNi"]), len(conmats_all["Ni"])))
        for i,j in zip(range(len(conmats_all["noNi"])),range(len(conmats_all["Ni"]))):
            if np.shape(conmats_all["noNi"][i]) != np.shape(conmats_all["Ni"][j]):
                conmat_check_cross[i,j] = 1
            elif np.abs(conmats_all["noNi"][i]-conmats_all["Ni"][j]).sum() != 0.0:
                conmat_check_cross[i,j] = 1

        if np.sum(conmat_check_cross) != 0:
            logger.critical('%s has inconsistent structures between conformer sets %s. Check manually.', kraken_id, str(suffixes))
            warnings += f'{kraken_id} inconsistent structures between the conformer sets  {suffixes}! Check manually.'
            return False, warnings
    else:
        raise ValueError(f'Somehow the number of jobs performed for {kraken_id} was {len(suffixes)}')

    logger.info('Structure validation completed in %.2f seconds. All checks passed.', time.time() - starttime)
    logger.info('[INFO] Begin conformer selection.')

    # Run the function that actually does the conformer selection
    conformers_to_use = conformer_selection(suffixes,
                                            energies,
                                            coords_all,
                                            elements_all,
                                            properties_all,
                                            selection_settings)


    logger.info('Conformers selected to use: %s', str(conformers_to_use))
    logger.info('Writing temporary structure files to %s', tmp_dir.resolve())

    # Conformers selected from both conformer searches - remove possible duplicates
    if len(suffixes) > 1:

        # For each of the jobs
        for suffix in suffixes:

            sdffiles[suffix] = []

            for conf in conformers_to_use[suffix]:
                file = tmp_dir / f'{kraken_id}_{suffix}_temp_{str(conf).zfill(4)}.xyz'
                file_inv = tmp_dir / f'{kraken_id}_{suffix}_temp_{str(conf).zfill(4)}_inv.xyz'
                write_xyz(file, conf, data_here[suffix])
                write_xyz(file_inv, conf, data_here[suffix], inverse=True)

                mol = openbabel.OBMol()
                obConversion.ReadFile(mol, str(file.absolute()))
                obConversion.WriteFile(mol, str(file.with_suffix('.sdf').absolute()))
                sdffiles[suffix].append(file.stem)

        logger.info('Creating RDKit Mol objects')
        molobjects = {}
        molxyzfilenames = {}
        runrmsd = True

        for suffix in suffixes:

            # Get all the mol objects
            molobjects[suffix] = [Chem.MolFromMolFile(str(Path(tmp_dir / f'{sdf}.sdf').absolute()), removeHs=False, strictParsing=False) for sdf in sdffiles[suffix]]

            if None in molobjects[suffix]:
                logger.warning('Found None mols when running RMSD calculations for job suffix %s', suffix)

            molobjects[suffix] = [x for x in molobjects[suffix] if x is not None]

            if len(molobjects[suffix]) == 0:
                logger.warning('%s_%s failed generating RDKit Mol objects. Continuing confomer selection without duplicate detection', kraken_id, suffix)
                warnings += f'{kraken_id}_{suffix} failed generating RDKit mol objects. Continuing conformer selection without duplicate detection.'
                runrmsd = False
                break

            # Get the xyz file names
            molxyzfilenames[suffix] = [f'{sdf}.xyz' for sdf in sdffiles[suffix]]

            # FF optimization. Optional: makes potential duplicate detection more robust when comparing conformers from different origins
            [AllChem.MMFFOptimizeMolecule(m) for m in molobjects[suffix]]

            # Remove all H: also optional but speeds up RMSD calculation
            molobjects[suffix] = [Chem.RemoveHs(mol) for mol in molobjects[suffix]]

            # Don't know why this is run
            molobjects[suffix] = [delete_haloalkane_halides(mol) for mol in molobjects[suffix]]

            # Create mirror images of each conformer
            molobjects[f'{suffix}_inv'] = [mirror_mol(mol) for mol in molobjects[suffix]]
            molxyzfilenames[f'{suffix}_inv'] = [f'{sdf}_inv.xyz' for sdf in sdffiles[suffix]]

        # Remove the temporary files
        # Shutil rmtree was giving errors
        for file in tmp_dir.glob('*'):
            try:
                file.unlink()
            except Exception as _e:
                logger.error('Failed to remove %s because %s', file.absolute(), _e)

        try:
            tmp_dir.rmdir()
        except Exception as _e:
            logger.error('Failed to remove %s because %s', tmp_dir.absolute(), _e)

        if runrmsd:

            rmsd_matrix = np.zeros(([len(molobjects[i]) for i in suffixes]))
            logger.info('Computing RMSD matrix of shape %s', str(rmsd_matrix.shape))

            #rmsd_matrix2 = np.zeros(([len(molobjects[i]) for i in suffixes]))  # This code was already commented out 12 Apr 2024
            #counter = 1                                                        # This code was already commented out 12 Apr 2024
            #ntotal = len(molobjects["noNi"]) * len(molobjects["Ni"])           # This code was already commented out 12 Apr 2024

            def proc_rmsd_matrix(mo1, mo2, moi1, i, j, starttime):
                rmsd1 = rdMolAlign.GetBestRMS(mo1, mo2)
                rmsd1_inv = rdMolAlign.GetBestRMS(moi1, mo2)
                return [(i, j), min(rmsd1, rmsd1_inv)]

            logger.debug(f'There are {len(molobjects["noNi"])}x{len(molobjects["Ni"])} conformers and {len(molobjects["noNi"])*len(molobjects["Ni"])*2} rmsd combinations')

            try:
                pool = Parallel(n_jobs=nprocs, verbose=0)
                parall = pool(delayed(proc_rmsd_matrix)(molobjects["noNi"][i], molobjects["Ni"][j],molobjects["noNi_inv"][i],i,j,starttime) for i,j in itertools.product(range(len(molobjects["noNi"])),range(len(molobjects["Ni"]))))

                for results in parall:
                    rmsd_matrix[results[0]] = results[1]
            except RuntimeError:
                logger.warning('%s_conf_%s and %s_conf_%s may have the wrong structure. Check Manually', kraken_id, str(conformers_to_use["noNi"][i]).zfill(4), kraken_id, str(conformers_to_use["Ni"][j]).zfill(4))
                logger.warning('Keeping both conformers and continuing conformer selection.')
                warnings+=f'[WARNING] {kraken_id}_conf_{str(conformers_to_use["noNi"][i]).zfill(4)} and {kraken_id}_conf_{str(conformers_to_use["Ni"][j]).zfill(4)} may have the wrong structure, check manually\n[WARNING] Keeping both conformers and continuing conformer selection.'

                # Forces this pair to be kept, can be found in the xxx_rmsd_matrix.csv
                rmsd_matrix[i, j] = 100

            #for i,j in itertools.product(range(len(molobjects["noNi"])),range(len(molobjects["Ni"]))):              #*
            #    try:                                                                                                #*
            #        rmsd1 = rdMolAlign.GetBestRMS(molobjects["noNi"][i], molobjects["Ni"][j])                       #*
            #        rmsd1_inv = rdMolAlign.GetBestRMS(molobjects["noNi_inv"][i], molobjects["Ni"][j])               #*
            #        rmsd_matrix[i,j] = min(rmsd1, rmsd1_inv)                                                        #*
            #    except RuntimeError:                                                                                #*
            #        print("WARNING: %s noNi_%s, Ni_%s may have the wrong structure, check manually. Keeping both conformers and continuing conformer selection."%(molname,str(conformers_to_use["noNi"][i]).zfill(4),str(conformers_to_use["Ni"][j]).zfill(4))) #*
            #        warnings+="WARNING: %s noNi_%s, Ni_%s may have the wrong structure, check manually. Keeping both conformers and continuing conformer selection.\n"%(molname,str(conformers_to_use["noNi"][i]).zfill(4),str(conformers_to_use["Ni"][j]).zfill(4))  #*
            #        rmsd_matrix[i,j] = 100 # forces this pair to be kept, can be found in the xxx_rmsd_matrix.csv   #*
            #        # return(False, warnings)                                                                       #*
                                                                                                                    #*
                #rmsd2 = get_rmsd_ext(molxyzfilenames["noNi"][i], molxyzfilenames["Ni"][j])                         #*
                #rmsd2_inv = get_rmsd_ext(molxyzfilenames["noNi_inv"][i], molxyzfilenames["Ni"][j])                 #*
                #rmsd_matrix2[i,j] = min(rmsd2, rmsd2_inv)                                                          #*
                #print("%i of %i done: %.3f %.3f  %.3f %.3f"%(counter, ntotal, rmsd1, rmsd2, rmsd1_inv, rmsd2_inv)) #*
                #counter+=1                                                                                         #*

            # Save rmsd_matrix
            df = pd.DataFrame(rmsd_matrix, columns=conformers_to_use["Ni"], index=conformers_to_use["noNi"])
            df.to_csv(save_dir / 'rmsdmatrix.csv', sep=";")

            logger.info('Saved RMSD matrix to %s', Path(save_dir / "rmsdmatrix.csv").absolute())

            remove = set(np.where(rmsd_matrix < 0.2)[1])
            conformers_to_use["Ni"] = [i for i in conformers_to_use["Ni"] if i not in remove]

    # Iterate through the suffixes
    for suffix in suffixes:

        logger.info('Number of conformers selected from %s: %d', suffix, len(conformers_to_use[suffix]))
        logger.info('Writing conformer indices %s to confselection_minmax_%s.txt', str(conformers_to_use[suffix]), suffix)

        with open(save_dir / f'confselection_minmax_{suffix}.txt', 'a', newline='\n', encoding='utf-8') as f:
            f.write(f'{kraken_id};{";".join([str(i) for i in conformers_to_use[suffix]])}\n')

        for conf in conformers_to_use[suffix]:
            conf_number_represented_as_str = str(conf).zfill(5)
            confname = f'{kraken_id}_{suffix}_{conf_number_represented_as_str}'

            # There used to be a todo list for some reason. This has been removed (15 Apr 2024)

            conf_folder = save_dir / confname
            com_file_path = conf_folder / f'{confname}.com'
            xyz_file_path = xyz_dir / f'{confname}.xyz'

            # If writefile is False, this only returns the geometry as a
            # string as required by the next function to write the .com file
            geometry_string = write_xyz(xyz_file_path, conf, data_here[suffix], writefile=True)

            write_coms(directory=f'{save_dir.absolute()}',
                       name=confname,
                       suffix="",
                       geometry=geometry_string,
                       joboption="all",
                       num_processors=40)

    # Get all the com files in the save_dir
    coms = [x for x in save_dir.glob('*.com')]
    logger.info('A total of %d conformers were selected for DFT for Kraken ID %s', len(coms), kraken_id)

    # Define the zip file to which we will compress the .com files for convenience
    zip_file = save_dir / f'{kraken_id}_gaussian_input.zip'

    if not zip_file.exists():
        logger.info('Compressing Gaussian16 input files to %s', zip_file.absolute())
        command = f'zip {zip_file.absolute()}'
        for _ in coms:
            command = command + f' {_.name}'
        subprocess.run(args=command.split(' '), cwd=save_dir, check=False)
    else:
        logger.warning('Found zip file at %s', zip_file.absolute())
        logger.warning('This file may not have up-to-date conformers. Delete it and run the script again.')

    logger.info('Completed conformer selection. There are %d Gaussian16 input files in %s', len(coms), save_dir.absolute())
    logger.info('These files are archived into %s', zip_file.absolute())
    logger.info('Transfer the archive to a system with Gaussian16.')
    logger.info('Submit the calculations, and return the results to %s', save_dir.absolute())

    return True, warnings


if __name__ == "__main__":
    # Test to see if I get the same conformers selected from old yaml files
    conformer_selection_main('00002158', save_dir=Path('/home/sigman/krakendev/data/00002158/dft/'), noNi_datafile=Path('/home/sigman/kraken-old/results_all_noNi/00002158_noNi_combined.yml'), Ni_datafile=Path('/home/sigman/kraken-old/results_all_Ni/00002158_Ni_combined.yml'))

