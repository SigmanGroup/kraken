#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Dispersion related code'''

import re
import json
import subprocess

from pathlib import Path
import pandas as pd


D3_TABLE_PATTERN = re.compile(r'(?<=#               XYZ \[au\]               R0\(AA\) \[Ang.\]  CN       C6\(AA\)     C8\(AA\)   C10\(AA\) \[au\])(.*?)(?=molecular)', re.DOTALL)

# This is for the old D4 (2.5.0 I think)
# D4_TABLE_PATTERN = re.compile(r'(?<=#   Z        covCN         q      C6AA      C8AA      α\(0\))(.*?)(?=molecular)', re.DOTALL)
D4_TABLE_PATTERN = re.compile(r'(?<=#    Z              CN          q     C6\(AA\)     C8\(AA\))(.*?)(?=Molecular)', re.DOTALL)

def run_dftd3(xyz_file: Path,
              dftd3_executable: Path,
              charge: int = 0) -> list[list[float]]:
    '''
    Executes Grimme’s DFT-D3 dispersion correction (Becke-Johnson damping) and extracts
    atom-specific C₆ and C₈ dispersion coefficients.

    Parameters
    ----------
    xyz_file: Path
        Path to the input XYZ file containing the molecular geometry.

    dftd3_executable: Path
        Path to the dftd3 executable

    charge: int, optional
        Molecular charge. Currently unused but preserved for interface completeness. Default is 0.

    Returns
    -------
    list[list[float]]
        A nested list where each sublist contains the [C₆, C₈] coefficients for an individual atom.

    Raises
    ------
    ValueError
        If the DFTD3 execution fails or the output does not contain a single table of coefficients.

    Notes
    -----
    The functional used is B3LYP with Becke-Johnson damping (-bjm flag).
    '''

    if not dftd3_executable.exists():
        raise FileNotFoundError(f'Could not locate {dftd3_executable.absolute()}')

    # Construct the DFTD3 execution command using B3LYP with BJ damping
    cmd = [str(dftd3_executable.absolute()), str(xyz_file.name), '-func', 'b3-lyp', '-bjm']

    # Define the output file name and path
    output_file = xyz_file.parent / f'{xyz_file.stem}_d3.out'

    # Run the DFTD3 command, redirecting stdout and stderr to the output file
    with open(output_file, 'w', encoding='utf-8') as o:
        proc = subprocess.run(args=cmd, stdout=o, stderr=o, check=False, cwd=output_file.parent)

    # Raise an error if the subprocess failed
    if proc.returncode != 0:
        raise ValueError(f'Issue with running dftd3 on {xyz_file.absolute()}')

    # Read the DFTD3 output text
    with open(output_file, 'r', encoding='utf-8') as infile:
        d3_out = infile.read()

    # Extract the atom-specific dispersion coefficient table using a precompiled regex
    table = re.findall(D3_TABLE_PATTERN, d3_out)

    # Ensure exactly one match was found
    if len(table) != 1:
        raise ValueError(f'Found {len(table)} DFTD3 tables when expecting 1 for file {xyz_file.name}')

    # Parse and clean the matched table
    table = table[0].strip().split('\n')
    table = [re.split(r'\s+', x.strip()) for x in table]

    # Construct DataFrame with relevant column headers
    df = pd.DataFrame(table, columns=['#', 'X', 'Y', 'Z', 'SYMBOL',
                                      'R0(AA) [Ang.]', 'CN', 'C6(AA)', 'C8(AA)', 'C10(AA) [au]'], dtype=object)

    # Extract C6 and C8 coefficients as float values
    d3 = list(zip(df['C6(AA)'].astype(float).to_list(), df['C8(AA)'].astype(float).to_list()))

    # Convert tuples to lists and return
    d3 = [list(x) for x in d3]

    return d3

def run_dftd4(xyz_file: Path,
              dftd4_executable: Path,
              charge: int = 0) -> list[list[float]]:
    '''
    Executes Grimme’s DFT-D4 dispersion correction (Becke-Johnson damping) and extracts
    atom-specific C6AA and α(0) values

    Parameters
    ----------
    xyz_file : Path
        Path to the input XYZ file containing the molecular geometry.

    dftd4_executable: Path
        Path to the dftd4 executable

    charge : int, optional
        Molecular charge. Currently unused but preserved for interface completeness. Default is 0.

    Returns
    -------
    list[list[float]]
        A nested list where each sublist contains the [C6AA, α(0)] coefficients for an individual atom.

    Raises
    ------
    ValueError
        If the DFTD3 execution fails or the output does not contain a single table of coefficients.

    Notes
    -----
    The functional used is B3LYP with Becke-Johnson damping (-bjm flag).
    '''

    if not dftd4_executable.exists():
        raise FileNotFoundError(f'Could not locate {dftd4_executable.absolute()}')

    # Define the output file name and path
    output_file = xyz_file.parent / f'{xyz_file.stem}_d4.out'
    json_file = xyz_file.parent / f'{xyz_file.stem}_d4.json'

    # Construct the DFTD3 execution command using B3LYP with BJ damping
    cmd = [str(dftd4_executable.absolute()), str(xyz_file.name), '-c', str(charge), '--property', '--json', str(json_file.name)]


    # Run the DFTD3 command, redirecting stdout and stderr to the output file
    with open(output_file, 'w', encoding='utf-8') as o:
        proc = subprocess.run(args=cmd, stdout=o, stderr=o, check=False, cwd=output_file.parent)

    # Raise an error if the subprocess failed
    if proc.returncode != 0:
        raise ValueError(f'Issue with running dftd4 on {xyz_file.absolute()}')

    # Read the DFTD4 json file for α(0) values
    # We can probably get the c6_coefficients here too, but they are in
    # a full matrix format and probably don't want to mess with that
    with open(json_file, 'r', encoding='utf-8') as handle:
        properties = json.load(handle)

    static_polarizabilities = properties['polarizibilities']

    with open(output_file, 'r', encoding='utf-8') as infile:
        d4out = infile.read()

    d4_table = re.findall(D4_TABLE_PATTERN, d4out)

    if len(d4_table) != 1:
        raise ValueError(f'Found {len(d4_table)} DFT-D4 tables when expecting 1 for {xyz_file.name}')

    # Process the table
    d4_table = d4_table[0].split('\n')
    d4_table = [re.sub(r'\s+', ' ', x) for x in d4_table]

    # Remove the empty lines and dashes
    d4_table = [x for x in d4_table if x != '']
    d4_table = [x.strip().split(' ') for x in d4_table if not all([z == '-' for z in x])]

    df = pd.DataFrame(d4_table, columns=['#', 'Z', 'SYMBOL', 'CN', 'q', 'C6(AA)', 'C8(AA)'])
    df['STATIC_POLARIZABILITY'] = static_polarizabilities

    # Extract C6 and static_polarizability  as float values
    d4 = list(zip(df['C6(AA)'].astype(float).to_list(), df['STATIC_POLARIZABILITY'].astype(float).to_list()))

    # Convert tuples to lists and return
    d4 = [list(x) for x in d4]

    return d4

def read_disp(file: Path,
              disp: str) -> list[None | float]:
    '''
    Only supports DFTD3 V3.1 Rev 1

    If it works successfully, it appears to return

    [None, None, C6(AA), C8(AA), None]

    '''

    if disp.casefold() == 'd4':
        raise ValueError(f'This function is not designed to read in D4. Only DFTD3 V3.1 Rev 1')

    with open(file, 'r', encoding='utf-8') as f:
        disp_cont = f.readlines()

    if disp.casefold() == "d4":
        P_pat = "15 P"
        start_pat = "#   Z        covCN         q      C6AA      C8AA      α(0)"
    elif disp.casefold() == "d3":
        P_pat = " p "
        start_pat = "XYZ [au]"
    else:
        raise ValueError(f'Could not find {start_pat} in {file.name}')

    for ind, line in enumerate(disp_cont[::-1]):

        if start_pat in line:

            for line_ in disp_cont[:-ind-1:-1]:

                if P_pat in line_:

                    if disp == "d4":
                        dispres = [float(i) for i in line_.split()[3:]]

                    elif disp == "d3":
                        dispres = [None, None]+[float(i) for i in line_.split()[-3:-1]]+[None]
                    return dispres

    return [None, None, None, None, None]