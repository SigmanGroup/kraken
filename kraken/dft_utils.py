#!/usr/bin/env python3
# coding: utf-8

'''Utility functions for the DFT portion of Kraken'''

import re
import math
import shutil
import logging
import subprocess

from pathlib import Path

import pandas as pd

import numpy as np

logger = logging.getLogger(__name__)

# Define constants
HARTREE_TO_KCAL = 627.509474 # conversion to kcal/mol; taken from https://en.wikipedia.org/wiki/Hartree (20.09.2018)
BOHR_TO_ANGSTROM = 0.52917721067 # conversion from bohr to angstrom; taken from https://en.wikipedia.org/wiki/Bohr_radius (20.09.2018)
R_GAS_CONSTANT_KCAL_PER_MOL_KELVIN = 0.0019872036 #kcal mol^-1 K^-1
TEMPERATURE_IN_KELVIN = 298.15 #K

rcov = {'H': 0.32, 'He': 0.46, 'Li': 1.2, 'Be': 0.94, 'B': 0.77, 'C': 0.75, 'N': 0.71,
        'O': 0.63, 'F': 0.64, 'Ne': 0.67, 'Na': 1.4, 'Mg': 1.25, 'Al': 1.13, 'Si': 1.04,
        'P': 1.1, 'S': 1.02, 'Cl': 0.99, 'Ar': 0.96, 'K': 1.76, 'Ca': 1.54, 'Sc': 1.33,
        'Ti': 1.22, 'V': 1.21, 'Cr': 1.1, 'Mn': 1.07, 'Fe': 1.04, 'Co': 1.0, 'Ni': 0.99,
        'Cu': 1.01, 'Zn': 1.09, 'Ga': 1.12, 'Ge': 1.09, 'As': 1.15, 'Se': 1.1, 'Br': 1.14,
        'Kr': 1.17, 'Rb': 1.89, 'Sr': 1.67, 'Y': 1.47, 'Zr': 1.39, 'Nb': 1.32, 'Mo': 1.24,
        'Tc': 1.15, 'Ru': 1.13, 'Rh': 1.13, 'Pd': 1.08, 'Ag': 1.15, 'Cd': 1.23, 'In': 1.28,
        'Sn': 1.26, 'Sb': 1.26, 'Te': 1.23, 'I': 1.32, 'Xe': 1.31, 'Cs': 2.09, 'Ba': 1.76,
        'La': 1.62, 'Ce': 1.47, 'Pr': 1.58, 'Nd': 1.57, 'Pm': 1.56, 'Sm': 1.55, 'Eu': 1.51,
        'Gd': 1.52, 'Tb': 1.51, 'Dy': 1.5, 'Ho': 1.49, 'Er': 1.49, 'Tm': 1.48, 'Yb': 1.53,
        'Lu': 1.46, 'Hf': 1.37, 'Ta': 1.31, 'W': 1.23, 'Re': 1.18, 'Os': 1.16, 'Ir': 1.11,
        'Pt': 1.12, 'Au': 1.13, 'Hg': 1.32, 'Tl': 1.3, 'Pb': 1.3, 'Bi': 1.36, 'Po': 1.31,
        'At': 1.38, 'Rn': 1.42, 'Fr': 2.01, 'Ra': 1.81, 'Ac': 1.67, 'Th': 1.58, 'Pa': 1.52,
        'U': 1.53, 'Np': 1.54, 'Pu': 1.55
    }

# DO NOT MODIFY THIS LIST BECAUSE THE CODE DEPENDS ON ITS INDICES
periodictable = ['Bq', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
                 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
                 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
                 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
                 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
                 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
                 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
                 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                 'Uub', 'Uut', 'Uuq', 'Uup', 'Uuh', 'Uus', 'Uuo', 'X']

def get_filecont(file: Path) -> list[str]:
    '''
    Reads the content of a file and returns it as a list of lines.
    If the last 10 lines of the file do not contain "Normal termination",
    the function returns a list indicating job failure or abnormal termination.

    Parameters
    ----------
    file : Path
        Path to the file to be read.

    Returns
    -------
    list[str]
        List of strings, each corresponding to a line in the file, if the file indicates
        successful termination. Otherwise, returns ['failed or incomplete job'].

    Notes
    -----
    Designed primarily for parsing output logs from quantum chemical packages
    (e.g., Gaussian), where "Normal termination" signifies a successful job.
    '''

    # Read the full file content into a list of lines
    with open(file, 'r', encoding='utf-8') as f:
        filecont = f.readlines()

    # Check for "Normal termination" in the last 10 lines
    for line in filecont[-10:]:
        if "Normal termination" in line:
            return filecont

    # Return failure notice if no successful termination found
    return ["failed or incomplete job"]

def make_fchk_file(file: Path,
                   dest: Path) -> Path:
    '''
    Converts a Gaussian binary checkpoint (.chk) file into a formatted checkpoint (.fchk) file
    using the `formchk` utility.

    Parameters
    ----------
    file : Path
        Path to the input .chk file to be converted.

    Returns
    -------
    Path
        Path to the resulting .fchk file.

    Raises
    ------
    ValueError
        If the provided file does not have a `.chk` suffix, or if the `formchk` command fails.

    Notes
    -----
    This function assumes `formchk` is available in the system's PATH.
    '''

    if file.suffix != '.chk':
        raise ValueError(f'Only .chk files are accepted')

    if file.with_suffix('.fchk').exists():
        logger.warning('%s already exists.', file.with_suffix(".fchk").absolute())

    cmd = ['formchk', str(file.name), str(file.with_suffix('.fchk').name)]

    proc = subprocess.run(args=cmd, cwd=file.parent, check=False)

    if proc.returncode != 0:
        raise ValueError(f'The return code for formchk for file {file.name} was not 0')

    return file.with_suffix('.fchk')

def get_coordinates_and_elements_from_logfile(file: Path) -> tuple[np.ndarray, np.ndarray]:
    '''
    Extracts atomic Cartesian coordinates (Å) and corresponding elemental symbols
    from the final geometry table of a Gaussian output .log file.

    Parameters
    ----------
    file : Path
        Path to the Gaussian .log file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - A NumPy array of shape (n_atoms, 3) with float coordinates (X, Y, Z).
        - A NumPy array of shape (n_atoms,) with string atomic symbols.

    Raises
    ------
    ValueError
        If no geometry table is found in the file.

    Notes
    -----
    This function parses the final occurrence of the "Coordinates (Angstroms)" section.
    Atomic numbers are converted to symbols using the `periodictable` mapping.
    '''

    coords = []

    with open(file, 'r', encoding='utf-8') as _:
        text = _.read()

    GEOM_TABLE_PATTERN = re.compile(r'(?<=Coordinates \(Angstroms\)\n Number     Number       Type             X           Y           Z\n ---------------------------------------------------------------------\n)(.*?)(?=\n ---------------------------------------------------------------------)', re.DOTALL)

    tables = re.findall(GEOM_TABLE_PATTERN, text)

    if not tables:
        raise ValueError(f'No geometry table found in {file.name}')

    logger.info('Found %d geometry tables in %s. Using the last one.', len(tables), file.name)

    # Get the last table
    geom_table = tables[-1].split('\n')

    # Clean it up
    geom_table = [re.sub(r'\s+', ' ', x).strip() for x in geom_table]

    # Split it on spaces
    geom_table = [x.split(' ') for x in geom_table]

    # Make dataframe
    df = pd.DataFrame(geom_table, columns=['CENTER_NUMBER', 'ATOMIC_NUMBER', 'ATOMIC_TYPE', 'X', 'Y', 'Z'])
    elements = [periodictable[x] for x in df['ATOMIC_NUMBER'].astype(int)]

    for i, row in df.iterrows():
        coords.append([float(row['X']), float(row['Y']), float(row['Z'])])

    return np.array(coords, dtype=float), np.array(elements, dtype=str)

def get_conmat(elements: np.ndarray, coords: np.ndarray) -> np.ndarray:
    '''
    Constructs a binary connectivity matrix using interatomic distances and
    covalent radii. This is a distance-based heuristic analogous to the
    connectivity definition in Grimme's D3 dispersion correction code.

    Parameters
    ----------
    elements : np.ndarray
        An array of atomic symbols (str) corresponding to each atom.
    coords : np.ndarray
        A NumPy array of shape (n_atoms, 3) containing atomic Cartesian coordinates in Å.

    Returns
    -------
    np.ndarray
        A symmetric binary matrix of shape (n_atoms, n_atoms) where an entry of 1 indicates
        two atoms are considered bonded by the heuristic, and 0 otherwise.

    Raises
    ------
    ValueError
        If coordinate dimensions are inconsistent with the number of elements or are not 3D.
    '''
    if isinstance(coords, list):
        coords = np.asarray(coords)

    n_atoms = len(elements)

    if coords.shape != (n_atoms, 3):
        raise ValueError(f'Expected coordinates of shape ({n_atoms}, 3), got {coords.shape}')

    conmat = np.zeros((n_atoms, n_atoms), dtype=int)

    k1 = 16.0
    k2 = 4.0 / 3.0

    for i in range(n_atoms):
        ri = rcov.get(elements[i])
        if ri is None:
            logger.error('Missing covalent radius for element %s', elements[i])
            continue

        for j in range(i + 1, n_atoms):
            rj = rcov.get(elements[j])
            if rj is None:
                continue

            r = np.linalg.norm(coords[j] - coords[i])
            rco = k2 * (ri + rj)
            damp = 1.0 / (1.0 + math.exp(-k1 * (rco / r - 1.0)))

            if damp > 0.85:
                conmat[i, j] = conmat[j, i] = 1

    return conmat

def split_log(file: Path) -> list[Path]:

    logger.info('Splitting log %s', file.name)

    # Read the file
    with open(file, 'r', encoding='utf-8') as f:
        loglines = f.readlines()

    # jobs = []  # 0: route line. 1: number of start line. 2: number of termination line
    routes   = []  # route line of each subjob
    chsp     = []  # charge/spin of each subjob
    types    = []  # type of subjob
    starts   = [0,]  # index of start line
    ends     = []  # index of end line
    termination = []  # Normal/Error

    for i, log_line in enumerate(loglines):
        if " #" in log_line and "---" in loglines[i - 1] and "---" in loglines[i + 1]:
            routes.append(log_line.strip())

        if " #" in log_line and "---" in loglines[i - 1] and "---" in loglines[i + 2]:
            routes.append(log_line.strip() + loglines[i + 1].strip())

        if  re.search("Charge = [-+]?[0-9 ]+multiplicity", log_line, re.IGNORECASE):
            chsp.append([int(x) for x in re.findall("[-+]?[0-9]+", log_line)])

        if "Normal termination" in log_line:
            ends.append(i)
            starts.append(i + 1)
            termination.append("Normal")

        if "Error termination" in log_line and "Error termination" not in loglines[i - 1]:
            termination.append("Error")
            ends.append(i + 3)

    if len(ends) < len(routes):
        ends.append(-2)
        termination.append("none")

    done = True

    files_created = []

    for i, route in enumerate(routes):
        if re.search("opt", route, re.IGNORECASE):
            types.append("opt")
        elif re.search("freq", route, re.IGNORECASE) and not re.search("opt", route,re.IGNORECASE):
            types.append("freq")
        elif re.search("wfn", route, re.IGNORECASE):
            types.append("sp")
        elif re.search("nmr", route, re.IGNORECASE):
            types.append("nmr")
        elif re.search("efg", route, re.IGNORECASE):
            types.append("efg")
        elif re.search("nbo", route, re.IGNORECASE) and chsp[i] == [0, 1]:
            types.append("nbo")
        elif re.search("nbo", route, re.IGNORECASE) and chsp[i] == [-1, 2]:
            types.append("ra")
        elif re.search("nbo", route, re.IGNORECASE) and chsp[i] == [1, 2]:
            types.append("rc")
        elif re.search("scrf", route, re.IGNORECASE):
            types.append("solv")

        if len(termination) == 0:
            print("no termination found. exit.")
            raise ValueError(f'No terminations were found in {file.name}')
        elif termination[i] == "Error":
            raise ValueError(f'Error found in terminations of {file.name}.')
        elif termination[i] == "none":
            raise ValueError(f'The last job of {file.name} did not terminate.')

        file_to_write_to = file.parent / f'{file.stem}_{types[i]}.log'

        with open(file_to_write_to, 'w') as f:
            for line in loglines[starts[i]: ends[i] + 1]:
                f.write(line)

            files_created.append(file_to_write_to)

    return files_created

def run_multiwfn(file: Path,
                 multiwfn_executable: Path,
                 multiwfn_settings_file: Path) -> tuple[Path, Path, Path]:
    '''
    Runs Multiwfn on a chk file using these commands for .fchk file

    First, the function runs multiwfn on the .fchk file

    12 Quantitative analysis of molecular surface
    2 Select mapped function, current: Electrostatic potential (ESP)
    -2 (Returns to upper level???)
    3 Spacing of grid points for generating molecular surface:  0.250000
        0.25 - Change the spacing to 0.25 (despite it already being set to this value)
    0 Start analysis now!
    7 Export all surface vertices to vtx.txt in current folder
    -1 Return to upper level menu
    -1 Return to upper level menu
    100 Other functions (Part 1)
    2 Export various files (mwfn/pdb/xyz/wfn/wfx/molden/fch/47/mkl...) or generate input file of quantum chemistry programs
    5 Output current wavefunction as .wfn file
        {file.stem}.wfn
    q - Quit analysis


    returns
    tuple[Path, Path, Path]
        .wfn file, surface_vertices_file, multiwfn_outputfile
    '''
    # Define the arguments to pass into multiwfn
    if file.suffix in ['.fch', '.fchk']:

        inputargs = [file.name, '12', '2', '-2', '3', '0.25',
                     '0', '7', '-1', '-1', '100',
                     '2', '5', f'{file.stem}.wfn', 'q']

        inputargs = '\n'.join(inputargs)

    else:
        inputargs = "12\n2\n-2\n3\n0.25\n0\n7\nq\n"


    if not multiwfn_executable.exists():
        raise FileNotFoundError('Could not locate %s', multiwfn_executable.absolute())

    cmd = [str(multiwfn_executable.absolute())]

    multiwfn_output_file = file.parent / f'{file.stem}_multiwfn.output'

    # Copy the settings .ini file to workdir if it doesn't exist
    local_settings_file = file.parent / 'settings.ini'
    if not local_settings_file.exists():
        shutil.copy2(multiwfn_settings_file, file.parent / multiwfn_settings_file.name)

    # Open the output file and run the command
    with open(multiwfn_output_file, 'w', encoding='utf-8') as output:
        proc = subprocess.run(args=cmd,
                              stdout=output,
                              stderr=output,
                              encoding="ascii",
                              shell=True,
                              input=inputargs,
                              cwd=file.parent,
                              check=False)

    logger.info('Finished multiwfn with return code %d', proc.returncode)

    # Get the wfn file
    wfn = file.with_suffix('.wfn')

    # Rename the vtx.txt fiel
    old_vtx = file.parent / 'vtx.txt'
    new_vtx = file.parent / f'{file.stem}_vtx.txt'
    shutil.move(old_vtx, new_vtx)

    return wfn, new_vtx, multiwfn_output_file