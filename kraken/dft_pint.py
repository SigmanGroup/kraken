#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path

import numpy as np

from .dft_utils import HARTREE_TO_KCAL

class pint_pdb:
    def __init__(self,
                 xyz: Path,
                 data: Path,
                 head=None,
                 col=None):

        self.xyz = xyz
        self.data = data
        self.path = self.xyz.parents[0]
        self.name = self.data.stem

        if head == None:
            self.head = 0
        else:
            self.head = head

        if col == None:
            self.col = 0
        else:
            self.col = col

        self.read(self.head, self.col)

    def read(self, head, col):
        self.xyz = np.genfromtxt(self.xyz, delimiter=None, usecols = (0, 1, 2), skip_header = head)
        self.values = np.genfromtxt(self.data, delimiter=None, usecols = (col), skip_header = 0)
        self.npoints = len(self.xyz)
        return

    def dump(self, pdb_file: Path):
        with open(pdb_file, 'w', encoding='utf-8') as f:
            f.write('REMARK   ' + str(self.npoints) + '\n')
            for ni in range(self.npoints):
                f.write('HETATM')
                f.write('{:>{length}s}'.format(str(ni+1), length = 5))
                f.write('{:>{length}s}'.format('C', length = 3))
                f.write('{:>{length}s}'.format('MOL', length = 6))
                f.write('{:>{length}s}'.format('A', length = 2))
                f.write('{:>{length}s}'.format('1', length = 4))
                f.write('{:>{length}s}'.format('', length = 4))
                f.write('{:>{length}s}'.format(str(format(self.xyz[ni][0], '.3f')), length = 8))
                f.write('{:>{length}s}'.format(str(format(self.xyz[ni][1], '.3f')), length = 8))
                f.write('{:>{length}s}'.format(str(format(self.xyz[ni][2], '.3f')), length = 8))
                f.write('{:>{length}s}'.format('1.00', length = 6))
                f.write('{:>{length}s}'.format(str(format(self.values[ni], '.2f')), length = 6))
                f.write('{:>{length}s}'.format('C', length = 11))
                f.write('\n')
        return

def compute_pint(number_of_points: int,
                 number_of_atoms: int,
                 coords_bohr: np.ndarray,
                 surface_points: np.ndarray,
                 dispersion_descriptors,
                 coefficient_selection: int) -> np.ndarray:
    '''
    DOCSTRING
    '''

    # Define pint array
    pint_array = np.zeros(number_of_points)

    for j in range(number_of_points):

        for i in range(number_of_atoms):

            ij_dist = np.sqrt(np.sum((coords_bohr[i] - surface_points[j]) ** 2))

            pint_array[j] += (np.sqrt(dispersion_descriptors[i][0]) / ((ij_dist) ** 3)) * np.sqrt(HARTREE_TO_KCAL)

            if coefficient_selection > 1:

                pint_array[j] += (np.sqrt(dispersion_descriptors[i][1]) / ((ij_dist) ** 4)) * np.sqrt(HARTREE_TO_KCAL)

                if coefficient_selection > 2:
                    # include abc dispersion here
                    pass

    return pint_array
