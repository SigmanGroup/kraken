#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Custom functions for reading and writing files
'''

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def readXYZs(file: Path):
    with open(file, 'r') as infile:
        lines = infile.readlines()

    coords=[[]]
    elements=[[]]

    for line in lines:

        if len(line.split())==1 and len(coords[-1])!=0:
            coords.append([])
            elements.append([])
        elif len(line.split())==4:
            elements[-1].append(line.split()[0].capitalize())
            coords[-1].append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])

    return coords, elements


def exportXYZs(coords,elements,filename):
    outfile=open(filename,"w")
    for idx in range(len(coords)):
        outfile.write("%i\n\n"%(len(elements[idx])))
        for atomidx,atom in enumerate(coords[idx]):
            outfile.write("%s %f %f %f\n"%(elements[idx][atomidx].capitalize(),atom[0],atom[1],atom[2]))
    outfile.close()


def read_xyz(file: Path) -> tuple[np.ndarray, list]:
    '''
    Reads a .xyz using custom xyz parsing and returns an array of
    cartesian coordinates and a list of corresponding elements.

    You should probably just use morfeus.read_xyz

    Parameters
    ----------
    file: Path
        Path to the .xyz logfile.

    Returns
    ----------
    tuple[np.ndarray, list]
    '''

    if isinstance(file, str):
        file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f'File {file.absolute()} does not exist')

    coords = []
    elements = []
    with open(file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    if len(lines) < 3:
        raise Exception(f'No coordinates were found in {file.absolute()}')

    for line in lines[2:]:
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])

    return np.array(coords), elements

def exportXYZ(file: str | Path,
              coords: np.ndarray,
              elements: list,
              mask=[]) -> Path:
    '''
    Export an XYZ file containing atomic coordinates and element symbols.

    Parameters
    ----------
    file: str | Path
        Destination path for the XYZ file.

    coords: np.ndarray
        Array of shape (N, 3) with atomic coordinates, where N is the number of atoms.

    elements: list
        List of length N containing element symbols (strings) corresponding to each coordinate.

    mask: list, optional
        List of integer indices specifying which atoms to include. If empty, all atoms are written.
        Defaults to [].

    Returns
    -------
    path: Path
        The Path object corresponding to the written XYZ file.
    '''

    if len(coords) != len(elements):
        raise ValueError(f'Number of coords {len(coords)} does not match number of elements {len(elements)}')

    with open(file, "w", encoding='utf-8') as outfile:

        # If no mask is provided, write out everything
        if len(mask)==0:

            # Write out length of list of elements (number of atoms)
            outfile.write(f'{len(elements)}\n\n')
            for atomidx,atom in enumerate(coords):
                outfile.write("%s %f %f %f\n"%(elements[atomidx].capitalize(),atom[0],atom[1],atom[2]))

        else:
            outfile.write("%i\n\n"%(len(mask)))
            for atomidx in mask:
                atom = coords[atomidx]
                outfile.write("%s %f %f %f\n"%(elements[atomidx].capitalize(),atom[0],atom[1],atom[2]))

    return Path(file)

def write_xyz(destination: str | Path,
              coords: NDArray,
              elements: NDArray | list,
              comment: str = '',
              mask: list[int] = []) -> Path:
    '''
    Writes an XYZ-format file to the specified destination.

    Parameters
    ----------
    destination: str | Path
        Path to the output file where the XYZ content will be written.

    coords: NDArray
        A (n_atoms, 3) array of Cartesian coordinates in Ångstrom.

    elements : NDArray
        A (n_atoms,) array of atomic symbols corresponding to the coordinates.

    comment: str
        Comment to place into the second line of the file

    mask: list[int]
        List of atom indices (0-indexed) that will be written to the output
        file instead of the full set of elements/coordinates

    Returns
    -------
    Path
        The absolute path to the written XYZ file.

    Raises
    ------
    ValueError
        If the number of elements and coordinates are inconsistent.
    '''
    if len(coords) != len(elements):
        raise ValueError(f'Number of coords {len(coords)} does not match number of elements {len(elements)}')

    with open(destination, 'w', encoding='utf-8') as outfile:

        # Write only the data for the mask if present
        if len(mask) != 0:
            outfile.write(f'{len(mask)}\n{comment}\n')
            for _idx in mask:
                _crd = coords[_idx]
                _ele = elements[_idx]
                outfile.write(f'{_ele:<3}  {_crd[0]:12.6f}  {_crd[1]:12.6f}  {_crd[2]:12.6f}\n')

        # Write everything if no mask specified
        else:
            outfile.write(f'{len(elements)}\n{comment}\n')
            for element, crd in zip(elements, coords):
                outfile.write(f'{element:<3}  {crd[0]:12.6f}  {crd[1]:12.6f}  {crd[2]:12.6f}\n')

    return Path(destination)

def log_to_xyz(file: Path) -> Path:
    '''
    Converts a quantum chemistry output log file to an XYZ-format geometry file using Open Babel.

    Parameters
    ----------
    file : Path
        Path to the input log file (e.g., Gaussian, ORCA, etc.) containing atomic coordinates.

    Returns
    -------
    Path
        Path to the resulting `.xyz` file written in Cartesian coordinates.

    Raises
    ------
    ValueError
        If the Open Babel conversion process returns a nonzero exit code, indicating failure.

    Notes
    -----
    - Requires Open Babel (`obabel`) to be installed and accessible in the system environment.
    - The output `.xyz` file is written in the same directory as the input log file with the `.xyz` extension.
    '''
    xyz_file = file.with_suffix('.xyz')

    cmd = ['obabel', '-ilog', str(file.name), '-oxyz', f'-O{xyz_file.name}']
    proc = subprocess.run(args=cmd, cwd=file.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    if proc.returncode != 0:
        raise ValueError(f'Could not convert {file.name} to a .xyz file.')

    return xyz_file

def xyz_to_sdf(xyz_file: Path,
               destination: Path) -> None:
    '''
    Converts a .xyz file to a .sdf file with openbabel.
    obabel command must be accessible from the commandline

    Parameters
    ----------
    xyz_file: Path
        Path to the .xyz file

    destination: Path
        Path to the destination .sdf file

    Returns
    ----------
    None
    '''
    if not xyz_file.exists():
        raise FileNotFoundError(f'{xyz_file.absolute()} does not exist.')

    if xyz_file.suffix != '.xyz':
        raise ValueError(f'xyz_to_sdf requires a .xyz file to convert')

    if destination.suffix != '.sdf':
        raise ValueError(f'xyz_to_sdf requires a .sdf file to convert to')

    cmd = ['obabel', '-ixyz', str(xyz_file.absolute()), '-osdf', f'-O{destination.absolute()}']

    proc = subprocess.run(args=cmd, cwd=xyz_file.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def get_outstreams(file: Path) -> list[str] | str:
    '''
    Extracts the compressed stream information blocks
    from a Gaussian 16output file.

    Parameters
    ----------
    file : Path
        Path to the G16 file (.log)

    Returns
    -------
    list[str] | str
        List of parsed stream blocks if the job completed normally.
        Returns an error message (str) if the function determines
        the job did not complete correctly.
    '''

    # Make lists for holding the results
    streams = []
    starts = []
    ends = []
    norm_terms = []

    # Default unless 'Normal termination' is in the file
    error = 'failed or incomplete job'

    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if '1\\1\\' in line:
            starts.append(i)
        if '@' in line:
            ends.append(i)
        if 'Normal termination' in line:
            norm_terms.append(i)
            error = ''

    if (len(starts) != len(ends))or (len(starts) == 0) or (error != '') or (len(norm_terms) != len(starts)):
        error = 'failed or incomplete job'
        return error

    for line_start_idx, line_end_index in zip(starts, ends):
        streams.append(''.join([line.strip() for line in lines[line_start_idx:line_end_index+1]]).split('\\'))

    return streams
