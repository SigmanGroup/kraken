#!/usr/bin/env python3
# coding: utf-8

'''
Functions for reading CREST and xTB
files and extracting parameters from them.
'''

import re
import shutil
import logging
import subprocess

from pathlib import Path

import numpy as np
import pandas as pd

from .file_io import write_xyz
from morfeus import read_xyz

logger = logging.getLogger(__name__)

XTB_VERSION_PATTERN = re.compile(r'[V|v]ersion\s+\d+\.\d+\.\d+', re.DOTALL)

def call_xtb(file: Path,
             lmo: bool,
             vfukui: bool,
             esp: bool,
             vomega: bool,
             vipea: bool,
             output_file: Path,
             charge: int = 0,
             reduce_output: bool = False,
             gbsa: str | None = 'toluene',
             nprocs: int = 1) -> Path:
    '''
    Calls the xTB executable with some options.

    These are the original calculations used in the Kraken workflow

    xtb --gbsa toluene --lmo --vfukui --esp -P <nprocs> <file>
    xtb --gbsa toluene --vomega --vipea -P <nprocs> <file>

    # The VIPEA calculation may require -P 1, the
    lmo calculation may not accept any processors indicated to work

    If one of these calculations has not been performed, the program
    will attempt to run it.

    Parameters
    ----------
    file: Path
        Path to the .xyz file

    reduce_output: bool
        If True, deletes several files from each calculation

    nprocs: int
        Number of processors to use

    Returns
    ----------
    None
    '''
    # Format the command
    command = f'xtb --gbsa toluene --lmo --vfukui --esp --chrg 0 {file.name}'
    cmd = ['xtb']

    # Add keywords
    if gbsa is not None:
        cmd.extend(['--gbsa', str(gbsa)])

    if lmo:
        cmd.append('--lmo')

    if vfukui:
        cmd.append('--vfukui')

    if esp:
        cmd.append('--esp')

    if vomega:
        cmd.append('--vomega')

    if vipea:
        cmd.append('--vipea')

    # Add the charge and file
    cmd.extend(['-P', str(nprocs), '--chrg', str(charge), file.name])

    logger.debug('Running %s', ' '.join(cmd))

    # Set the output files
    xtb_log_file = output_file
    xtb_error_file = output_file.with_suffix('.error')

    if not xtb_log_file.exists():
        with open(xtb_log_file, 'a', encoding='utf-8') as mystdout:
            with open(xtb_error_file, 'a', encoding='utf-8') as mystderror:
                process = subprocess.run(args=cmd, stdout=mystdout, stderr=mystderror, cwd=file.parent, check=False)
    else:
        logger.info('Found xTB log file at %s', xtb_log_file.absolute())

    # Reduce file output
    if reduce_output:
        for item in file.parent.glob('*'):
            item_name = item.name
            if item_name == 'wbo' or item_name == 'xtbrestart' or item_name == 'xtbscreen.xyz':
                if item.is_dir():
                    shutil.rmtree(item)
                elif item.is_file():
                    item.unlink()
                else:
                    pass

    return xtb_log_file

def _get_xtb_version_from_file(file: Path) -> str:
    '''
    Determines the version of xtb used to generate
    an xtb log file
    '''
    with open(file, 'r', encoding='utf-8') as infile:
        text = infile.read()

    version = re.findall(XTB_VERSION_PATTERN, text)

    if len(version) == 0:
        raise ValueError(f'Could not determine xTB version from {file.absolute()}')

    return str(version[0]).lower()

def _get_xtb_version() -> str | None:
    '''
    Get's the version of xtb from the command line.
    '''
    try:
        proc = subprocess.run(args=['xtb', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        exit('[FATAL] xtb is not callable. Make sure it is on your PATH environment variable.')
    version = proc.stdout.decode('utf-8')
    version = re.findall(XTB_VERSION_PATTERN, version)

    if len(version) == 0:
        logger.warning('xTB is callable, but the version could not be checked.')
        return None

    return str(version[0]).lower()

def get_results_conformer(regular_xtb_calculation_log: Path,
                          vipea_xtb_calculation_log: Path,
                          natoms: int,
                          debug: bool = False) -> tuple:
    '''
    Wrapper function for reading properties from specific xtb versions

    Returns
    ----------
    bool (whether xTB calculation is done)
    muls
    alphas
    wils

    dip : list[float, float, float]
        Molecular dipole x, y, and z

    alpha

    fukui

    HOMO_LUMO_gap

    IP_delta_SCC

    EA_delta_SCC
    global_electrophilicity_index
    esp_profile
    esp_points
    occ_energies
    virt_energies
    nucleophilicity
    '''

    if not regular_xtb_calculation_log.exists():
        raise FileNotFoundError(f'Could not find {regular_xtb_calculation_log.absolute()}')

    if not vipea_xtb_calculation_log.exists():
        raise FileNotFoundError(f'Could not find {vipea_xtb_calculation_log.absolute()}')

    # Get the xtb version
    xtb_version = _get_xtb_version_from_file(regular_xtb_calculation_log)

    if '6.2.2' in xtb_version:
        logger.info('Reading xTB log files for version %s (should be original version)', str(xtb_version))
        # Get results from regular XTB calculation
        muls, alphas, wils, dip, alpha, fukui, HOMO_LUMO_gap, esp_profile, esp_points, occ_energies, virt_energies = read_xtb_log1_orig_xtb(regular_xtb_calculation_log,
                                                                                                                                            natoms=natoms)
        # Get results from vipea XTB calculation
        IP_delta_SCC, EA_delta_SCC, global_electrophilicity_index, nucleophilicity = read_xtb_log2_orig_xtb(vipea_xtb_calculation_log,
                                                                                                            natoms=natoms)

    else:
        logger.info('Reading xTB log files for version %s', str(xtb_version))
        # Get results from regular XTB calculation
        muls, alphas, wils, dip, alpha, fukui, HOMO_LUMO_gap, esp_profile, esp_points, occ_energies, virt_energies = read_xtb_log1(regular_xtb_calculation_log, natoms=natoms, debug=debug)

        # Get results from vipea XTB calculation
        IP_delta_SCC, EA_delta_SCC, global_electrophilicity_index, nucleophilicity = read_xtb_log2(vipea_xtb_calculation_log, natoms=natoms, debug=debug)

    return True, muls, alphas, wils, dip, alpha, fukui, HOMO_LUMO_gap, IP_delta_SCC, EA_delta_SCC, global_electrophilicity_index, esp_profile, esp_points, occ_energies, virt_energies, nucleophilicity


def read_xtb_log1_orig_xtb(regular_xtb_calculation_log: Path,
                           natoms: int):
    '''
    Reads the properties from the
    xtb --gbsa toluene --lmo --vfukui --esp
    calculation.

    This function is currently built for calculations with xTB Version 6.2.2
    '''

    if not regular_xtb_calculation_log.exists():
        raise FileNotFoundError(f'Could not locate {regular_xtb_calculation_log.absolute()} in read_xtb_log1.')

    # Get the raw text for regex
    with open(regular_xtb_calculation_log, 'r', encoding='utf-8') as infile:
        text = infile.read()

    if 'convergence criteria cannot be satisfied within' in text:
        logger.warning('%s did not converged.', regular_xtb_calculation_log.absolute())

    # Get the Mulliken charges and alpha values, capture everything between alpha(0) and Mol. C6AA
    COVCN_Q_C6AA_ALPHA_TABLE_PATTERN = re.compile(r'(?<=α\(0\)\n)(.*?)(?=Mol. C6AA /au)', re.DOTALL)
    matches = re.findall(COVCN_Q_C6AA_ALPHA_TABLE_PATTERN, text)
    if len(matches) == 1:
        COVCN_Q_C6AA_ALPHA_TABLE = matches[0]
        COVCN_Q_C6AA_ALPHA_TABLE = [x.strip() for x in COVCN_Q_C6AA_ALPHA_TABLE.split('\n')]
        COVCN_Q_C6AA_ALPHA_TABLE = [re.sub('\s+', ' ', x).split(' ') for x in COVCN_Q_C6AA_ALPHA_TABLE if x != '']
        COVCN_Q_C6AA_ALPHA_TABLE = pd.DataFrame(COVCN_Q_C6AA_ALPHA_TABLE, columns=['ATOM_NUM', 'ATOMIC_NUMBER', 'SYMBOL', 'covCN', 'q', 'C6AA', 'alpha(0)'])
        muls = COVCN_Q_C6AA_ALPHA_TABLE['q'].astype(float).to_list()
        alphas = COVCN_Q_C6AA_ALPHA_TABLE['alpha(0)'].astype(float).to_list()
    else:
        logger.warning('\tCound not extract alphas and Mulliken charges from %s', regular_xtb_calculation_log.absolute())
        alphas = []
        muls = []

    # Get the molecular dipole, finding everything between molecular dipole: and molecular dipole (traceless)
    MOL_DIPOLE_PATTERN = re.compile(r'(?<=molecular dipole:)(.*?)(?=\nmolecular quadrupole \(traceless\):)', re.DOTALL)
    matches = re.findall(MOL_DIPOLE_PATTERN, text)
    if len(matches) == 1:
        MOL_DIPOLE = matches[0].strip()
        MOL_DIPOLE = re.sub('\s+', ' ', MOL_DIPOLE)
        x, y, z, total = [float(x) for x in MOL_DIPOLE.split('full:')[1].strip().split(' ')]
        dip = [x, y, z]
    else:
        logger.warning('\tCould not extract molecule dipole from %s', regular_xtb_calculation_log.absolute())
        dip = [0.0,0.0,0.0]

    # Get molecular polarizability
    MOL_POLARIZABILITY_PATTERN = re.compile(r'(?<=Mol. α\(0\) /au        :)(.*?)(?=\n)', re.DOTALL)
    matches = re.findall(MOL_POLARIZABILITY_PATTERN, text)
    if len(matches) == 1:
        alpha = float(matches[0])
    else:
        logger.warning('\tCould not extract molecular polarizability (alpha) from %s', regular_xtb_calculation_log.absolute())
        alpha = None

    # Get fukui
    fukui=[]
    FUKUI_TABLE_PATTERN = re.compile(r'(?<= f\(0\))(.*?)(?=------)', re.DOTALL) #TODO This matching pattern is pretty bad
    matches = re.findall(FUKUI_TABLE_PATTERN, text)
    if len(matches) == 1:
        FUKUI_TABLE = [x.strip() for x in matches[0].strip().split('\n')]
        FUKUI_TABLE = [re.sub('\s+', ' ', x).split(' ') for x in FUKUI_TABLE if x != '']
        FUKUI_TABLE = pd.DataFrame(FUKUI_TABLE, columns=['#', 'ATOMIC_SYMBOL', 'f(+)', 'f(-)', 'f(0)'])
        for i, row in FUKUI_TABLE.iterrows():
            fukui.append([float(row['f(+)']), float(row['f(-)']), float(row['f(0)'])])
    else:
        logger.warning('\tCould not extract Fukui from %s', regular_xtb_calculation_log.absolute())
        fukui = []

    # Get HOMO-LUMO gap (one of two extractions, for some reason it's done in )
    HOMO_LUMO_GAP_PATTERN = re.compile(r'(?<=:: HOMO-LUMO gap)(.*?)(?=eV)', re.DOTALL)
    matches = re.findall(HOMO_LUMO_GAP_PATTERN, text)
    if len(matches) == 1:
        HOMO_LUMO_gap = float(matches[0])
    else:
        logger.warning('\tould not extract HOMO-LUMO gap from %s', regular_xtb_calculation_log.absolute())
        HOMO_LUMO_gap = None

    # Get Wiberg/Mayer (AO) data (get the total bond order for each atom)
    WILS_TABLE_ATOM_PATTERN = re.compile(r'(?<=WBO to atom ...\n)(.*?)(?=molecular dipole)', re.DOTALL)
    matches = re.findall(WILS_TABLE_ATOM_PATTERN, text)
    if len(matches) == 1:
        wils = matches[0]
    else:
        logger.warning('\tCould not extract Wiberg/Mayer (AO) data from %s', regular_xtb_calculation_log.absolute())
        wils = None

    # Making these operations really explicit for myself later
    wils = wils.strip().split('\n')
    wils = [re.sub('\s+', ' ', x) for x in wils]
    wils = [re.sub('\n', '', x) for x in wils]
    wils = [x.strip() for x in wils]
    wils = [x.split(' ') for x in wils]
    wils = [float(x[2]) for x in wils]

    # Get the occupied and virtual orbital energies
    # This should get the occupied orbital energies from the eV column
    # of the table for the orbitals surrounding the FMOs (i.e., the ~13 or
    # so orbitals between the "..." characters in the printed table)
    occ_energies = []
    virt_energies = []
    ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE_PATTERN = re.compile(r'(?<=\* Orbital Energies and Occupations\n\n         #    Occupation            Energy/Eh            Energy/eV\n      -------------------------------------------------------------\n)(.*?)(?=\n\s+-------------------------------------------------------------\n                  HL-Gap)', re.DOTALL)
    matches = re.findall(ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE_PATTERN, text)
    if len(matches) == 1:
        ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE = matches[0].strip()
        ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE = [re.sub('\s+', ' ', x) for x in ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE.split('\n')]

        # Remove the first two and last two entries of the table
        ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE = ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE[2:-2]
        ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE = [x.split(' ') for x in ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE]
        for line in ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE:
            if len(line) == 5 and '(LUMO)' not in line:
                # Add the last item in the lines (energy in eV)
                occ_energies.append(float(line[-1]))
            elif len(line) == 5 and '(LUMO)' in line:
                virt_energies.append(float(line[-2]))
            elif len(line) == 6 and '(HOMO)' in line:
                occ_energies.append(float(line[-2]))
            elif len(line) == 4:
                virt_energies.append(float(line[-1]))
            else:
                raise ValueError(f'There was an issue parsing a line orbital energies in {regular_xtb_calculation_log.absolute()}'
                                 f'\n{line}')
    else:
        logger.warning('\tCould not extract orbital energies from %s', regular_xtb_calculation_log.absolute())

    logger.debug('\talpha: %s', alpha)
    logger.debug('\tdip: %s', dip)
    logger.debug('\tlen(muls): %d', len(muls))
    logger.debug('\tlen(alphas): %d', len(alphas))
    logger.debug('\tlen(fukui): %d', len(fukui))
    logger.debug('\tlen(wils): %d', len(wils))
    logger.debug('\tlen(occ_energies): %d', len(occ_energies))
    logger.debug('\tlen(virt_energies): %d', len(virt_energies))
    logger.debug('\tHOMO_LUMO_gap: %s', HOMO_LUMO_gap)

    ESP_PROFILE_PATH = Path(regular_xtb_calculation_log.parent / 'xtb_esp_profile.dat')
    ESP_DAT_PATH = Path(regular_xtb_calculation_log.parent / 'xtb_esp.dat')

    esp_profile = []
    if not ESP_PROFILE_PATH.exists():
        print(f'\t[WARNING] Could not find ESP_PROFILE_PATH {ESP_PROFILE_PATH}')
    else:
        for line in open(ESP_PROFILE_PATH, 'r', encoding='utf-8'):
            potential_esp_data = list(re.split('\s+', line.strip()))
            if len(potential_esp_data) == 2:
                esp_profile.append([float(potential_esp_data[0]), float(potential_esp_data[1])])

    # Read in the ESP_DAT_PATH
    esp_points = []
    if not ESP_DAT_PATH.exists():
        logger.warning('Could not find ESP_DAT_PATH', ESP_DAT_PATH)

    else:
        for line in open(ESP_DAT_PATH, 'r', encoding='utf-8'):
            potential_esp_data = list(re.split('\s+', line.strip()))

            if len(potential_esp_data) == 4:
                esp_points.append([float(potential_esp_data[0]), float(potential_esp_data[1]), float(potential_esp_data[2]), float(potential_esp_data[3])])
            else:
                raise ValueError('ESP points were not of length 4')

    # If something went wrong with the parameter extraction, return None
    if len(occ_energies) == 0:
        logger.warning('\tSetting occ_energies to None.')
        occ_energies = None
    if len(virt_energies) == 0:
        logger.warning('\tSetting virt_energies to None.')
        virt_energies = None
    if len(fukui) != natoms:
        logger.warning('\tSetting fukui to None.')
        fukui = None
    if len(wils) != natoms:
        logger.warning('\tSetting wils to None.')
        wils = None
    if len(alphas) != natoms:
        logger.warning('\tSetting alphas to None.')
        alphas = None
    if len(esp_profile) == 0:
        logger.warning('\tCould not extract esp profile data from %s.', ESP_PROFILE_PATH.absolute())
        esp_profile = None
    if len(esp_points) == 0:
        logger.warning('\tCould not extract esp point data from %s.', ESP_DAT_PATH.absolute())
        esp_points = None

    return muls, alphas, wils, dip, alpha, fukui, HOMO_LUMO_gap, esp_profile, esp_points, occ_energies, virt_energies

def read_xtb_log2_orig_xtb(logfile: Path,
                           natoms: int) -> tuple[float, float, float, float]:
    '''
    Reads the logfile from an XTB --vipea
    calculation and returns the following values:

    IP_delta_SCC
    EA_delta_SCC
    global_electrophilicity_index
    nucleophilicity

    Reads log files from xtb version 6.7.0 (08769fc)
    compiled by 'albert@albert-system' on 2024-03-04

    Parameters
    ----------
    logfile: Path
        Path to the xTB logfile.

    Returns
    ----------
    tuple[float, float, float, float]
    '''

    # Make sure the logfile exists
    if not logfile.exists():
        raise FileNotFoundError(f'{logfile.absolute()} does not exist.')

    # Read in the file
    with open(logfile, 'r', encoding='utf-8') as infile:
        text = infile.read()

    # Check that it's the right logfile
    if '--vipea' not in text:
        raise ValueError(f'It looks like {logfile.absolute()} was not run with --vipea.')

    # If the calculation failed, raise an error
    if 'convergence criteria cannot be satisfied within' in text:
        print(f'\t[WARNING] {logfile.absolute()} did not converge.')
        return None, None, None, None

    # Get the EA_delta_SCC
    EA_DELTA_SCC_PATTERN = re.compile(r'(?<=delta SCC EA \(eV\):)(.*?)(?=\n)', re.DOTALL)
    EA_delta_SCC = [x.strip() for x in re.findall(EA_DELTA_SCC_PATTERN, text)]
    if len(EA_delta_SCC) != 1:
        print(f'[WARNING] Found {len(EA_delta_SCC)} EA_delta_SCC when expecting 1. Setting EA_delta_SCC to None.')
        EA_delta_SCC = None
    else:
        EA_delta_SCC = float(EA_delta_SCC[0])

    # Get the IP_delta_SCC
    IP_DELTA_SCC_PATTERN = re.compile(r'(?<=delta SCC IP \(eV\):)(.*?)(?=\n)', re.DOTALL)
    IP_delta_SCC = [x.strip() for x in re.findall(IP_DELTA_SCC_PATTERN, text)]
    if len(IP_delta_SCC) != 1:
        print(f'[WARNING] Found {len(IP_delta_SCC)} IP_delta_SCC when expecting 1. Setting IP_delta_SCC to None.')
        IP_delta_SCC = None
    else:
        IP_delta_SCC = float(IP_delta_SCC[0])

    # Also set nucleophilicity to -IP_delta_SCC
    if IP_delta_SCC is not None:
        nucleophilicity = -IP_delta_SCC
    else:
        print(f'[WARNING] Nucleophilicity is calculated from IP_delta_SCC. Setting nucleophilicity to None.')
        nucleophilicity = None

    # Get the global_electrophilicity_index
    GLOBAL_ELECTROPHILICITY_INDEX = re.compile(r'(?<=Global electrophilicity index \(eV\):)(.*?)(?=\n)', re.DOTALL)
    global_electrophilicity_index = [x.strip() for x in re.findall(GLOBAL_ELECTROPHILICITY_INDEX, text)]
    if len(global_electrophilicity_index) != 1:
        print(f'Found {len(global_electrophilicity_index)} global_electrophilicity_index when expecting 1. Setting to None.')
        global_electrophilicity_index = None
    else:
        global_electrophilicity_index = float(global_electrophilicity_index[0])

    # Get the empirical_EA_shift
    EMPRICAL_EA_SHIFT_PATTERN = re.compile(r'(?<=empirical EA shift \(eV\):)(.*?)(?=\n)', re.DOTALL)
    empirical_EA_shift = [x.strip() for x in re.findall(EMPRICAL_EA_SHIFT_PATTERN, text)]
    if len(empirical_EA_shift) != 1:
        print(f'[WARNING] Found {len(empirical_EA_shift)} empirical_EA_shift when expecting 1. Setting to None.')
        empirical_EA_shift = None
    else:
        empirical_EA_shift = float(empirical_EA_shift[0])

    # Get the empirical_EA_shift
    EMPRICAL_IP_SHIFT_PATTERN = re.compile(r'(?<=empirical IP shift \(eV\):)(.*?)(?=\n)', re.DOTALL)
    empirical_IP_shift = [x.strip() for x in re.findall(EMPRICAL_IP_SHIFT_PATTERN, text)]
    if len(empirical_IP_shift) != 1:
        print(f'[WARNING] Found {len(empirical_IP_shift)} empirical_IP_shift when expecting 1. Setting to None.')
        empirical_IP_shift = None
    else:
        empirical_IP_shift = float(empirical_IP_shift[0])

    return IP_delta_SCC, EA_delta_SCC, global_electrophilicity_index, nucleophilicity

def _get_orbital_energies_and_occupations_table(file_text: str) -> pd.DataFrame:
    '''
    Extracts the orbital energies and occupations table from an XTB single-point calculation output.

    This function identifies and parses the orbital energies and occupations table from the provided
    XTB calculation text. The extracted data includes the orbital index, occupancy, energy in Hartrees
    (Eh), energy in electron volts (eV), and orbital designation (occupied or virtual).

    Parameters
    ----------
    file_text : str
        The raw text output from an XTB single-point calculation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the parsed orbital energies and occupations with columns:
        ['ORB#', 'OCCUPANCY', 'Energy(Eh)', 'Energy(Ev)', 'DESIGNATION'].
    '''

    ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE_PATTERN = re.compile(r'(?<=\* Orbital Energies and Occupations\n\n         #    Occupation            Energy/Eh            Energy/eV\n      -------------------------------------------------------------\n)(.*?)(?=\n\s+-------------------------------------------------------------\n                  HL-Gap)', re.DOTALL)

    # Extract the table content from the file text
    matches = re.findall(ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE_PATTERN, file_text)

    # If exactly one table is found, proceed with parsing
    if len(matches) == 1:
        table_text = matches[0].strip()

        # Split the table based on the new line and replace multiple spaces with a single space
        table_lines = [re.sub(r'\s+', ' ', x).strip() for x in table_text.split('\n')]

        # Remove placeholder lines containing "..." which indicate omitted orbitals in the output
        table_lines = [line.split(' ') for line in table_lines if '...' not in line]

        # Container for the parsed table
        parsed_table = []

        # Initial designation for orbitals (starts with occupied)
        designation = 'occ'

        for entry in table_lines:

            # Check if we need to change the designation
            if any(['LUMO' in x for x in entry]):
                designation = 'virt'

            # Adjust entries based on missing columns to standardize the format
            if len(entry) == 4 and '(LUMO)' not in entry:
                entry.append(designation)  # Ensure consistency with HOMO/LUMO lines
            if len(entry) == 4 and '(LUMO)' in entry:
                entry.insert(1, np.nan)  # Insert NaN for missing occupancy column
            elif len(entry) == 3:
                entry.insert(1, np.nan)  # Insert NaN for missing occupancy column
                entry.append(designation)
            elif len(entry) != 5:
                raise ValueError('Could not parse ORBITAL_ENERGIES_AND_OCCUPATIONS_TABLE')

            parsed_table.append(entry)

        # Convert parsed data into a DataFrame with appropriate column names
        orbital_table = pd.DataFrame(parsed_table, columns=['ORB#', 'OCCUPANCY', 'Energy(Eh)', 'Energy(Ev)', 'DESIGNATION'])

        return orbital_table

    else:
        raise ValueError(f'No orbital energies and occupations table was found.')

def read_xtb_log1(regular_xtb_calculation_log: Path,
                  natoms: int,
                  debug: bool = False):
    '''
    Reads the properties from the
    xtb --gbsa toluene --lmo --vfukui --esp
    calculation.

    Reads log files from * xtb version 6.4.0 (d4b70c2)
    compiled by 'ehlert@majestix' on 2021-02-01
    '''

    if not regular_xtb_calculation_log.exists():
        raise FileNotFoundError(f'Could not locate {regular_xtb_calculation_log.absolute()} in read_xtb_log1.')

    # Get the raw text for regex
    with open(regular_xtb_calculation_log, 'r', encoding='utf-8') as infile:
        text = infile.read()

    if 'convergence criteria cannot be satisfied within' in text:
        logger.warning('%s did not converged', regular_xtb_calculation_log.absolute())

    # Get the Mulliken charges and alpha values, capture everything between alpha(0) and Mol. C6AA
    COVCN_Q_C6AA_ALPHA_TABLE_PATTERN = re.compile(r'(?<=α\(0\)\n)(.*?)(?=Mol. C6AA /au)', re.DOTALL)

    matches = re.findall(COVCN_Q_C6AA_ALPHA_TABLE_PATTERN, text)
    if len(matches) == 1:
        COVCN_Q_C6AA_ALPHA_TABLE = matches[0]
        COVCN_Q_C6AA_ALPHA_TABLE = [x.strip() for x in COVCN_Q_C6AA_ALPHA_TABLE.split('\n')]
        COVCN_Q_C6AA_ALPHA_TABLE = [re.sub('\s+', ' ', x).split(' ') for x in COVCN_Q_C6AA_ALPHA_TABLE if x != '']
        COVCN_Q_C6AA_ALPHA_TABLE = pd.DataFrame(COVCN_Q_C6AA_ALPHA_TABLE, columns=['ATOM_NUM', 'ATOMIC_NUMBER', 'SYMBOL', 'covCN', 'q', 'C6AA', 'alpha(0)'])
        muls = COVCN_Q_C6AA_ALPHA_TABLE['q'].astype(float).to_list()
        alphas = COVCN_Q_C6AA_ALPHA_TABLE['alpha(0)'].astype(float).to_list()
    else:
        logger.warning('Could not extract alphas and Mulliken charges from %s', regular_xtb_calculation_log.absolute())
        alphas = []
        muls = []

    # Get the molecular dipole, finding everything between molecular dipole: and molecular dipole (traceless)
    MOL_DIPOLE_PATTERN = re.compile(r'(?<=molecular dipole:)(.*?)(?=\nmolecular quadrupole \(traceless\):)', re.DOTALL)
    matches = re.findall(MOL_DIPOLE_PATTERN, text)
    if len(matches) == 1:
        MOL_DIPOLE = matches[0].strip()
        MOL_DIPOLE = re.sub('\s+', ' ', MOL_DIPOLE)
        x, y, z, total = [float(x) for x in MOL_DIPOLE.split('full:')[1].strip().split(' ')]
        dip = [x, y, z]
    else:
        logger.warning('Could not extract molecular dipole from %s', regular_xtb_calculation_log.absolute())
        dip = [0.0,0.0,0.0]

    # Get molecular polarizability
    MOL_POLARIZABILITY_PATTERN = re.compile(r'(?<=Mol. α\(0\) /au        :)(.*?)(?=\n)', re.DOTALL)
    matches = re.findall(MOL_POLARIZABILITY_PATTERN, text)
    if len(matches) == 1:
        alpha = float(matches[0])
    else:
        logger.warning('Could not extract molecular polarizablity (alpha) from from %s', regular_xtb_calculation_log.absolute())
        alpha = None

    # Get fukui
    fukui=[]
    FUKUI_TABLE_PATTERN = re.compile(r'(?<= f\(0\))(.*?)(?=------)', re.DOTALL) #TODO This matching pattern is pretty bad
    matches = re.findall(FUKUI_TABLE_PATTERN, text)
    if len(matches) == 1:
        FUKUI_TABLE = [x.strip() for x in matches[0].strip().split('\n')]
        FUKUI_TABLE = [re.sub('\s+', ' ', x).split(' ') for x in FUKUI_TABLE if x != '']
        FUKUI_TABLE = pd.DataFrame(FUKUI_TABLE, columns=['#', 'f(+)', 'f(-)', 'f(0)'])
        for i, row in FUKUI_TABLE.iterrows():
            fukui.append([float(row['f(+)']), float(row['f(-)']), float(row['f(0)'])])
    else:
        logger.warning('Could not extract fukui from from %s', regular_xtb_calculation_log.absolute())
        fukui = []

    # Get HOMO-LUMO gap (one of two extractions, for some reason it's done in )
    HOMO_LUMO_GAP_PATTERN = re.compile(r'(?<=:: HOMO-LUMO gap)(.*?)(?=eV)', re.DOTALL)
    matches = re.findall(HOMO_LUMO_GAP_PATTERN, text)
    if len(matches) == 1:
        HOMO_LUMO_gap = float(matches[0])
    else:
        logger.warning('Could not extract molecular HOMO-LUMO gap from from %s', regular_xtb_calculation_log.absolute())
        HOMO_LUMO_gap = None

    # Get Wiberg/Mayer (AO) data (get the total bond order for each atom)
    WILS_TABLE_ATOM_PATTERN = re.compile(r'\d+\s+\d+ [a-zA-Z]+\s+\d+\.\d+ --', re.DOTALL)
    matches = re.findall(WILS_TABLE_ATOM_PATTERN, text)
    # check if the number of atom patterns we found is the same as the number of atoms in the calculation or file
    wils = [float(x.strip().split()[3]) for x in matches]

    # Get the table of energies and occupations
    try:
        orbital_energies_and_occupation_table = _get_orbital_energies_and_occupations_table(file_text=text)
        occ_energies = orbital_energies_and_occupation_table.loc[orbital_energies_and_occupation_table['DESIGNATION'].isin(['(HOMO)', 'occ']), 'Energy(Ev)'].astype(float).to_list()
        virt_energies = orbital_energies_and_occupation_table.loc[orbital_energies_and_occupation_table['DESIGNATION'].isin(['(LUMO)', 'virt']), 'Energy(Ev)'].astype(float).to_list()
    except ValueError as e:
        logger.error('Could not get occ_energies and virt_energies because %s. Setting values to None.', e)
        occ_energies = None
        virt_energies = None

    if debug:
        print(f'\t[DEBUG] alpha: {alpha}')
        print(f'\t[DEBUG] dip: {dip}')
        print(f'\t[DEBUG] len(muls): {len(muls)}')
        print(f'\t[DEBUG] len(alphas): {len(alphas)}')
        print(f'\t[DEBUG] len(fukui): {len(fukui)}')
        print(f'\t[DEBUG] len(wils): {len(wils)}')
        print(f'\t[DEBUG] len(occ_energies): {len(occ_energies)}')
        print(f'\t[DEBUG] len(virt_energies): {len(virt_energies)}')
        print(f'\t[DEBUG] HOMO_LUMO_gap: {HOMO_LUMO_gap}')

    ESP_PROFILE_PATH = Path(regular_xtb_calculation_log.parent / 'xtb_esp_profile.dat')
    ESP_DAT_PATH = Path(regular_xtb_calculation_log.parent / 'xtb_esp.dat')

    esp_profile = []
    if not ESP_PROFILE_PATH.exists():
        logger.warning('Could not find ESP_PROFILE_PATH %s', ESP_PROFILE_PATH.absolute())
    else:
        for line in open(ESP_PROFILE_PATH, 'r', encoding='utf-8'):
            potential_esp_data = list(re.split('\s+', line.strip()))
            if len(potential_esp_data) == 2:
                esp_profile.append([float(potential_esp_data[0]), float(potential_esp_data[1])])

    # Read in the ESP_DAT_PATH
    esp_points = []
    if not ESP_DAT_PATH.exists():
        logger.warning('Could not find ESP_DAT_PATH %s', ESP_DAT_PATH.absolute())
    else:
        for line in open(ESP_DAT_PATH, 'r', encoding='utf-8'):
            potential_esp_data = list(re.split('\s+', line.strip()))

            if len(potential_esp_data) == 4:
                esp_points.append([float(potential_esp_data[0]), float(potential_esp_data[1]), float(potential_esp_data[2]), float(potential_esp_data[3])])
            else:
                raise ValueError('ESP points were not of length 4')

    # If something went wrong with the parameter extraction, return None
    if occ_energies is not None:
        if len(occ_energies) == 0:
            logger.warning('\tSetting occ_energies to None')
            occ_energies = None
    if virt_energies is not None:
        if len(virt_energies) == 0:
            logger.warning('\tSetting virt_energies to None')
            virt_energies = None

    if len(fukui) != natoms:
        logger.warning('\tSetting fukui to None.')
        fukui = None
    if len(wils) != natoms:
        logger.warning('\tSetting wils to None.')
        wils = None
    if len(alphas) != natoms:
        logger.warning('\tSetting alphas to None.')
        alphas = None
    if len(esp_profile) == 0:
        logger.warning('\tCould not extract esp profile data from %s', ESP_PROFILE_PATH.absolute())
        esp_profile = None
    if len(esp_points) == 0:
        logger.warning('\tCould not extract esp point data from %s', ESP_DAT_PATH.absolute())
        esp_points = None

    return muls, alphas, wils, dip, alpha, fukui, HOMO_LUMO_gap, esp_profile, esp_points, occ_energies, virt_energies

def read_xtb_log2(logfile: Path,
                  natoms: int,
                  debug: bool = False) -> tuple[float, float, float, float]:
    '''
    Reads the logfile from an XTB --vipea
    calculation and returns the following values:

    IP_delta_SCC
    EA_delta_SCC
    global_electrophilicity_index
    nucleophilicity

    Reads log files from xtb version 6.7.0 (08769fc)
    compiled by 'albert@albert-system' on 2024-03-04

    Parameters
    ----------
    logfile: Path
        Path to the xTB logfile.

    Returns
    ----------
    tuple[float, float, float, float]
    '''

    # Make sure the logfile exists
    if not logfile.exists():
        raise FileNotFoundError(f'{logfile.absolute()} does not exist.')

    # Read in the file
    with open(logfile, 'r', encoding='utf-8') as infile:
        text = infile.read()

    # Check that it's the right logfile
    if '--vipea' not in text:
        raise ValueError(f'It looks like {logfile.absolute()} was not run with --vipea.')

    # If the calculation failed, raise an error
    if 'convergence criteria cannot be satisfied within' in text:
        logger.warning('%s did not converge. Returning None.', logfile.absolute())
        return None, None, None, None

    # Get the EA_delta_SCC
    EA_DELTA_SCC_PATTERN = re.compile(r'(?<=delta SCC EA \(eV\):)(.*?)(?=\n)', re.DOTALL)
    EA_delta_SCC = [x.strip() for x in re.findall(EA_DELTA_SCC_PATTERN, text)]
    if len(EA_delta_SCC) != 1:
        raise ValueError(f'Found {len(EA_delta_SCC)} EA_delta_SCC when expecting 1.')
    EA_delta_SCC = float(EA_delta_SCC[0])

    # Get the IP_delta_SCC
    IP_DELTA_SCC_PATTERN = re.compile(r'(?<=delta SCC IP \(eV\):)(.*?)(?=\n)', re.DOTALL)
    IP_delta_SCC = [x.strip() for x in re.findall(IP_DELTA_SCC_PATTERN, text)]
    if len(IP_delta_SCC) != 1:
        raise ValueError(f'Found {len(EA_delta_SCC)} IP_delta_SCC when expecting 1.')
    IP_delta_SCC = float(IP_delta_SCC[0])

    # Also set nucleophilicity to -IP_delta_SCC
    nucleophilicity = -IP_delta_SCC

    # Get the global_electrophilicity_index
    GLOBAL_ELECTROPHILICITY_INDEX = re.compile(r'(?<=Global electrophilicity index \(eV\):)(.*?)(?=\n)', re.DOTALL)
    global_electrophilicity_index = [x.strip() for x in re.findall(GLOBAL_ELECTROPHILICITY_INDEX, text)]
    if len(global_electrophilicity_index) != 1:
        raise ValueError(f'Found {len(global_electrophilicity_index)} global_electrophilicity_index when expecting 1.')
    global_electrophilicity_index = float(global_electrophilicity_index[0])

    # Get the empirical_EA_shift
    EMPRICAL_EA_SHIFT_PATTERN = re.compile(r'(?<=empirical EA shift \(eV\):)(.*?)(?=\n)', re.DOTALL)
    empirical_EA_shift = [x.strip() for x in re.findall(EMPRICAL_EA_SHIFT_PATTERN, text)]
    if len(empirical_EA_shift) != 1:
        raise ValueError(f'Found {len(empirical_EA_shift)} empirical_EA_shift when expecting 1.')
    empirical_EA_shift = float(empirical_EA_shift[0])

    # Get the empirical_EA_shift
    EMPRICAL_IP_SHIFT_PATTERN = re.compile(r'(?<=empirical IP shift \(eV\):)(.*?)(?=\n)', re.DOTALL)
    empirical_IP_shift = [x.strip() for x in re.findall(EMPRICAL_IP_SHIFT_PATTERN, text)]
    if len(empirical_IP_shift) != 1:
        raise ValueError(f'Found {len(empirical_IP_shift)} empirical_IP_shift when expecting 1.')
    empirical_IP_shift = float(empirical_IP_shift[0])

    return IP_delta_SCC, EA_delta_SCC, global_electrophilicity_index, nucleophilicity

def xtb_opt(coords,
            elements,
            smiles: str,
            scratch_dir: Path,
            charge=0,
            nprocs: int | None = None,
            freeze=[]):

    '''
    nprocs: int | None
        Number of processors to request. If None, xtb will respond
        to environment variable OMP_NUM_THREADS.
    '''

    # Export the xyzfile

    write_xyz(destination=scratch_dir / 'xtb_tmp.xyz',
              coords=coords,
              elements=elements,
              comment='xtb_optimization_input',
              mask=[])

    if len(freeze) > 0:
        with open(scratch_dir / "xcontrol", 'w', encoding='utf-8') as outfile:
            outfile.write("$fix\n")
            outfile.write(" atoms: ")
            for counter,i in enumerate(freeze):
                if (counter+1)<len(freeze):
                    outfile.write(f"{i+1},")
                else:
                    outfile.write(f"{i+1}\n")
        add=" -I xcontrol "
    else:
        add=""

    logfile = scratch_dir / 'xtb.log'
    cmd = f"xtb xtb_tmp.xyz --chrg {charge} --opt{add}"

    # Add nprocs if requested
    if nprocs is not None:
        cmd = cmd + f' -P {nprocs}'

    logger.debug('Running command in xtb_opt: %s', cmd)

    with open(logfile, 'w') as stdout:
        subprocess.run(args=cmd.split(), stdout=stdout, stderr=subprocess.PIPE, cwd=scratch_dir, check=False)

    if not Path(scratch_dir / 'xtbopt.xyz').exists():
        logger.warning('xTB geometry optimization failed for %s', smiles)
        return coords, elements

    elements_new, coords_new = read_xyz(scratch_dir / 'xtbopt.xyz')

    for item in scratch_dir.glob('*'):
        item.unlink()

    return coords_new, elements_new


