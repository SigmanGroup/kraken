#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import subprocess
import logging

from pathlib import Path

import numpy as np
import pandas as pd

# The old vmin4 script used its own rcov, but all the values were
# the same as the currently formatted one in utils
from .dft_utils import BOHR_TO_ANGSTROM, rcov

logger = logging.getLogger(__name__)

# parameters
numbers_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")

# box dimensions:
# standard presets for Phosphines
class Grid_def:
    def __init__(self,size):
        if size == 0:
            self.dxy = 0.85 / BOHR_TO_ANGSTROM   # dimensions perpendicular to P-LP
            self.dz = 0.8 / BOHR_TO_ANGSTROM   #0.5 dimension in P-LP direction
            self.d_lp = 2.0 / BOHR_TO_ANGSTROM #1.9# distance of grid center from P
            self.npoints = [25,25,40]     # number of grid points in x,y,z directions
        elif size == 1:
            self.dxy = 1.9 / BOHR_TO_ANGSTROM   # dimensions perpendicular to P-LP
            self.dz = 1.0 / BOHR_TO_ANGSTROM   #0.5 dimension in P-LP direction
            self.d_lp = 2.15 / BOHR_TO_ANGSTROM #1.9# distance of grid center from P
            self.npoints = [50,50,50]     # number of grid points in x,y,z directions

dxy = 0.85 / BOHR_TO_ANGSTROM   # dimensions perpendicular to P-LP
dz = 0.8 / BOHR_TO_ANGSTROM   #0.5 dimension in P-LP direction
d_lp = 2.0 / BOHR_TO_ANGSTROM #1.9# distance of grid center from P
npoints = [25,25,40]     # number of grid points in x,y,z directions

# alternative presets for very electron-poor phosphines
# dxy = 1.5 / BOHR_TO_ANGSTROM   # dimensions perpendicular to P-LP
# dz = 1.50 / BOHR_TO_ANGSTROM    # dimension in P-LP direction
# d_lp = 2.5/BOHR_TO_ANGSTROM  # distance of grid center from P
# npoints = [30,30,50]     # number of grid points in x,y,z directions
# dxy = 1.9/BOHR_TO_ANGSTROM   # dimensions perpendicular to P-LP
# dz = 1.0/BOHR_TO_ANGSTROM    # dimension in P-LP direction
# d_lp = 2.0/BOHR_TO_ANGSTROM  # distance of grid center from P
# npoints = [50,50,50]     # number of grid points in x,y,z directions

# for N ligands
# dxy = 0.9/BOHR_TO_ANGSTROM   # dimensions perpendicular to P-LP
# dz = 0.5/BOHR_TO_ANGSTROM    # dimension in P-LP direction
# d_lp = 1.4/BOHR_TO_ANGSTROM  # distance of grid center from P
# npoints = [25,25,25]     # number of grid points in x,y,z directions

# Renamed this to vmin_elements since "elements" is used frequently in the code
vmin_elements = {"1": "H", "5": "B", "6": "C", "7": "N", "8": "O", "9": "F", "14": "Si",
            "15": "P", "16": "S", "17": "Cl", "26": "Fe", "33": "As", "34": "Se",
            "35": "Br", "44": "Ru", "46": "Pd", "51": "Sb", "53": "I",
}

class Vminob:
    def __init__(self,
                 name,
                 ext,
                 status):
        self.name = name
        self.ext = ext
        if status == "" or status == "stdcube":
            self.dxy = 0.85 / BOHR_TO_ANGSTROM   # dimensions perpendicular to P-LP
            self.dz = 0.8 / BOHR_TO_ANGSTROM   #0.5 dimension in P-LP direction
            self.d_lp = 2.0 / BOHR_TO_ANGSTROM #1.9# distance of grid center from P
            self.npoints = [25,25,40]     # number of grid points in x,y,z directions
        elif status == "follow_edge":
            self.dxy = 1.9 / BOHR_TO_ANGSTROM   # dimensions perpendicular to P-LP
            self.dz = 1.0 / BOHR_TO_ANGSTROM   #0.5 dimension in P-LP direction
            self.d_lp = 2.15 / BOHR_TO_ANGSTROM #1.9# distance of grid center from P
            self.npoints = [50,50,50]     # number of grid points in x,y,z directions

    def p_lp(self,iteration):
        # expects a three-coordinate P containing compound. Adds a valence to P so that the angle to the three previous substituents is maximized and resorts the coordinate output for convenience

        for i,atom in enumerate(self.coords):
            if (atom[0] == "P" or atom[0] == "As" or atom[0] == "Sb") and sum(self.conmat[i]) <=3:
                self.nop = i
                self.coordp = np.array(self.coords[i][1:])
                break

        vec = np.array([0.0,0.0,0.0])
        bonded = []
        for atom in range(len(self.coords)):
            if self.conmat[self.nop][atom]:
                bonded.append(atom)
                coorda = np.array(self.coords[atom][1:])
                vec += self.coordp - coorda
        self.coordlp = self.d_lp*vec/np.linalg.norm(vec) + self.coordp  # coordinates of grid center

        atomno = max(iteration-1,0)
        dir_bond1 = np.array((self.coords[bonded[atomno]][1:]))-np.array((self.coords[self.nop][1:])) # direction of first P-R bond
        dirx = np.cross(dir_bond1,vec)  # defines the x-direction of the grid
        diry = np.cross(vec,dirx)       # defines the y-direction of the grid

        self.dirx = dirx/np.linalg.norm(dirx)    # normalization of the grid-coordinate system
        self.diry = diry/np.linalg.norm(diry)
        self.dirz = vec/np.linalg.norm(vec)
        self.grid_coords = [self.dirx*self.dxy/self.npoints[0],self.diry*self.dxy/self.npoints[1],self.dirz*self.dz/self.npoints[2]]
    #    grid_coords = [2*dirx/npoints[0],2*diry/npoints[1],dirz/2*npoints[2]]
        self.grid_or = self.coordlp - self.dz*0.5 * self.dirz - 0.5*self.dxy * self.diry - 0.5*self.dxy * self.dirx # grid_origin
        return()

    def copy_info(self,prev):
        self.coords = prev.coords
        self.conmat = prev.conmat
        self.nop = prev.nop
        self.coordp = prev.coordp
        self.coordlp = prev.coordlp
        self.dirx = prev.dirx
        self.diry = prev.diry
        self.dirz = prev.dirz
        self.grid_coords = prev.grid_coords
        self.grid_or = prev.grid_or

    def follow_vmin(self,coords_vmin_prev):
        vec_P_vmin_prev = coords_vmin_prev - self.coordp
        grid_center = self.coordp + vec_P_vmin_prev / np.linalg.norm(vec_P_vmin_prev) * (np.linalg.norm(vec_P_vmin_prev)+0.5)
        self.dirx = np.cross(self.coordlp,vec_P_vmin_prev)
        self.diry = np.cross(vec_P_vmin_prev,self.dirx)
        self.dirx = self.dirx/np.linalg.norm(self.dirx)    # normalization of the grid-coordinate system
        self.diry = self.diry/np.linalg.norm(self.diry)
        self.dirz = vec_P_vmin_prev/np.linalg.norm(vec_P_vmin_prev)
        self.grid_coords = [self.dirx*self.dxy/self.npoints[0],self.diry*self.dxy/self.npoints[1],self.dirz*self.dz/self.npoints[2]]
        self.grid_or = grid_center - self.dz*0.5 * self.dirz - 0.5*self.dxy * self.diry - 0.5*self.dxy * self.dirx # grid_origin

        # print("coords_vmin_prev: {:.2f},{:.2f},{:.2f}\ngrid_center: {:.2f},{:.2f},{:.2f}\ngrid_or: {:.2f},{:.2f},{:.2f}\n".format(*coords_vmin_prev,*grid_center,*self.grid_or))

    def read_cubtxt(self, file: Path) -> None:
        '''
        Sets:
        self.esp
        self.v_min
        self.vmin_ind
        self.coords_min
        '''
        get_permission(filepath=file)

        with open(file, 'r', encoding='utf-8') as f:
            cubtxt = f.readlines()

        self.esp = np.array(([float(line.split()[-1]) for line in cubtxt]))
        self.v_min = np.amin(self.esp)
        self.vmin_ind = int(np.where(self.esp==self.v_min)[0][0])
        self.coords_min = np.array(([float(i) for i in cubtxt[self.vmin_ind].split()[:3]]))  # find index of v_min, get coordinates in Bohr
        return

    def analyze_vmin(self) -> str:
        '''

        '''
        npoints = self.npoints

        self.r_min = np.linalg.norm(self.coords_min - self.coordp) * BOHR_TO_ANGSTROM
        self.line_pos = self.vmin_ind + 1

        if self.line_pos % npoints[2] in [0,1]:
            self.on_edge = True
        elif (self.line_pos <= npoints[2] * npoints[1]) or (npoints[2] * npoints[1] * npoints[0] - self.line_pos <= npoints[2] * npoints[1]):
            self.on_edge = True
        elif (self.line_pos % (npoints[2] * npoints[1]) <= npoints[2]) or (npoints[2] * npoints[1] - (self.line_pos % (npoints[2] * npoints[1])) <= npoints[2]):
            self.on_edge = True
        else:
            self.on_edge = False

        self.vminatom,self.vmindist = "",""
        rmin_other = {(i[0]+str(j+1)): np.linalg.norm(self.coords_min - np.array(i[1:])) * BOHR_TO_ANGSTROM for j,i in enumerate(self.coords) if i[0] not in  ['H', 'P']}
        rmin_other_S = pd.Series(rmin_other)

        if rmin_other_S.min() < self.r_min / 1.1: # scaled P to account for larger radius vs. most other elements
            self.wrongatom = True
            # print("{};Vmin of other atom;{}".format(self.name,rmin_other_S.idxmin()))
            self.vminatom = rmin_other_S.idxmin()
            self.vmindist = rmin_other_S.min()
        else:
            self.wrongatom = False

        self.coords_min_A = self.coords_min * BOHR_TO_ANGSTROM

        #with open(name + "_vmin_results2.txt","a") as f:

        return f'{self.name,};{self.suffix};{self.v_min};{self.r_min:.5f};{self.on_edge};{self.wrongatom};{self.coords_min_A[0]:.4f};{self.coords_min_A[1]:.4f};{self.coords_min_A[2]:.4f}\n'

    def get_geom_cub(self, file: Path):
        '''
        Reads the geometry from a .cub file in Bohr

        Updates:
        self.coords
        self.grid_coords
        self.grid_or
        self.dirx
        self.diry
        self.dirz

        '''
        # reads the geometry out of a cube file (in Bohr)


        self.coords = []
        self.grid_coords = [[],[],[]]

        get_permission(file)

        with open(file, 'r') as f:
            fcont = f.readlines()
        natoms = int(re.findall(numbers_pattern,fcont[2])[0])
        self.grid_or = np.asarray([float(i) for i in re.findall(numbers_pattern,fcont[2])[1:]])
        for i in range(3):
            self.npoints[i] = int(re.findall(numbers_pattern,fcont[3+i])[0])
            self.grid_coords[i] = np.asarray([float(j) for j in re.findall(numbers_pattern,fcont[3+i])[1:]])
        self.dirx = self.grid_coords[0]/np.linalg.norm(self.grid_coords[0])    # normalization of the grid-coordinate system
        self.diry = self.grid_coords[1]/np.linalg.norm(self.grid_coords[1])
        self.dirz = self.grid_coords[2]/np.linalg.norm(self.grid_coords[2])

        for line in fcont[6:6+natoms]:
            linesplit = re.findall(numbers_pattern,line)
            if len(linesplit) != 5:
                break
            if linesplit[0] not in vmin_elements.keys():
                print("Element not implemented in code: " + linesplit[0])
                continue
            else:
                self.coords.append([vmin_elements[linesplit[0]]]+[float(i) for i in linesplit[2:]])
        return

def get_permission(filepath: str | Path) -> None:
    '''
    Ensures that a file is readable. If not, sets permissions to 0o777.
    '''
    path = Path(filepath)
    if not os.access(path, os.R_OK):
        path.chmod(0o777)

def run_cubegen(n_in: str,
                fchk_file: Path,
                cube_infile: Path,
                cube_outfile: Path,
                nprocs: int = 1) -> int:

    nproc = str(nprocs)

    if not (fchk_file.parent == cube_infile.parent == cube_outfile.parent):
        raise ValueError('All files must be in the same directory for cubegen.')

    cmd = ['cubegen', nproc, 'potential=scf', str(fchk_file.name), str(cube_outfile.name), '-1', 'h', str(cube_infile.name)]

    logger.info('Running cubegen on %s', fchk_file.name)
    logger.debug('cmd: %s', str(cmd))
    proc = subprocess.run(args=cmd,
                          cwd=fchk_file.parent)

    logger.info('Finished cubegen on %s\treturncode %d', fchk_file.name, proc.returncode)

    return proc.returncode

def get_conmat_for_vmin(_rcov: dict, _coords: list[list]):
    '''
    Silly way to get the conmat in a particular way for the vmin object
    # partially based on code from Robert Paton's Sterimol script, which based this part on Grimme's D3 code
    '''
    natom = len(_coords)
    max_elem = 94
    k1 = 16.0
    k2 = 4.0/3.0
    conmat = np.zeros((natom,natom))
    for i in range(0,natom):
        if _coords[i][0] not in _rcov.keys():
            continue
        for iat in range(0,natom):
            if _coords[iat][0] not in _rcov.keys():
                continue
            if iat != i:
                dx = _coords[iat][1] - _coords[i][1]
                dy = _coords[iat][2] - _coords[i][2]
                dz = _coords[iat][3] - _coords[i][3]
                r = np.linalg.norm([dx,dy,dz]) * BOHR_TO_ANGSTROM # with conversion Bohr - Angstrom
                rco = _rcov[_coords[i][0]]+_rcov[_coords[iat][0]]
                rco = rco*k2
                rr=rco/r
                damp=1.0/(1.0 + math.exp(-k1*(rr-1.0)))
                if damp > 0.85:
                    conmat[i,iat],conmat[iat,i] = 1,1
    return conmat



def get_vmin(fchk: Path,
             nprocs: int,
             runcub=False) -> Vminob:
    '''
    Pass the .fchk file
    runcub should be True in the read_conformer script
    '''

    # Don't know what this shit was supposed to do
    if "_Pesp_out" in fchk.name:
        name =  fchk.name.split("_Pesp_out.")[0]
        ext = fchk.name.split("_Pesp_out.")[1]
    else:
        name = fchk.name.split(".")[0]
        ext = fchk.name.split(".")[1]

    vminobjects = []

    output_file = fchk.parent / f'{fchk.stem}_vmin_results2.txt'
    final_output_file = fchk.parent / f'{fchk.stem}_vmin_results.txt'

    with open(output_file, 'w') as f:
        f.write("Name;Suffix;Vmin;R(Vmin);On_edge;Wrong_atom;X_Vmin;Y_Vmin;Z_Vmin\n")

    status = ""

    # status:
    #       "": normal/first pass
    #       "follow_edge": previous vmin was on edge of cube
    #       "stdcube": no vmin was found at P; generate three cubes around expected P lone pair
    #       "done": finished procedure

    iteration = 0

    while status != "done":
        vminob = Vminob(name, ext, status)

        # If this is the first iteration, read in the file and get the conmat
        if iteration == 0 and "fch" in ext:
            vminob.coords = get_geom_fch(vminob, file=fchk.parent / f'{name}.{ext}')
            vminob.conmat = get_conmat_for_vmin(_rcov=rcov, _coords=vminob.coords)

        # This part of the code is untested
        elif iteration == 0 and ext == "cub":
            _cub_file = fchk.parent / f'{name}_Pesp_out.cub"'
            vminob.get_geom_cub(file=_cub_file)
            vminob.conmat = get_conmat_for_vmin(rcov, vminob.coords)
        else:
            vminob.coords = vminobjects[0].coords
            vminob.conmat = vminobjects[0].conmat

        if status == "follow_edge":
            vminob.copy_info(vminobjects[0])
            vminob.follow_vmin(vminobjects[-1].coords_min)
        else:
            vminob.p_lp(iteration)

        vminob.suffix = ("_" + status+str(iteration)) * bool(len(status))

        cubegen_input_file = fchk.parent / f'{fchk.stem}_Pesp_in{vminob.suffix}.cub'
        cubegen_output_file = fchk.parent / f'{fchk.stem}_Pesp_out{vminob.suffix}.cub'

        # Write the input file regardless
        if not cubegen_input_file.exists():
            write_inpcube(vminob=vminob, destination=cubegen_input_file)

        # If the cubegen output file doesn't exist
        if not cubegen_output_file.exists():

            # If the current extension is cub
            if vminob.ext == "cub":

                # If the output file doesn't exist and we're trying to
                # run the cub file, then we error out
                if runcub == False:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write("{};{};Problem encountered, run additional cubegens\n".format(vminob.name,vminob.suffix))

                    logger.error('%s;%s;Problem encountered, run additional cubegens', vminob.name, vminob.suffix)
                    break

                # If runcub is on
                elif fchk.exists():

                    # Change vminob to fchk mode
                    vminob.ext = "fchk"

                    run_cubegen(n_in=name,
                                fchk_file=fchk,
                                cube_infile=cubegen_input_file,
                                cube_outfile=cubegen_output_file,
                                nprocs=nprocs)


                else:
                    with open(output_file, 'a') as f:
                        f.write("{};{};Missing .fchk for further analysis\n".format(vminob.name,vminob.suffix))

                    logger.error('%s;%s;Missing .fchk for further analysis\n', vminob.name, vminob.suffix)
                    break

            # If the mode is not cub, run cubegen anyways
            else:
                run_cubegen(n_in=name,
                            fchk_file=fchk,
                            cube_infile=cubegen_input_file,
                            cube_outfile=cubegen_output_file,
                            nprocs=nprocs)


        cubman_output_file = fchk.parent / f'{fchk.stem}_Pesp_out{vminob.suffix}.txt'

        if not cubman_output_file.exists():
            logger.info('Running cubman')
            cubman_output_file = run_cubman(cube_file=cubegen_output_file)
        else:
            logger.info('Found output .txt file from cubman at %s', str(cubman_output_file.absolute()))

        logger.info('Reading .cub ASCII %s', cubman_output_file.name)
        vminob.read_cubtxt(file=cubman_output_file)
        analysis_str = vminob.analyze_vmin()

        # Append the analysis to the file
        with open(output_file, 'a') as f:
            f.write(analysis_str)

        # Add our current obj to the list
        vminobjects.append(vminob)

        if vminob.on_edge == False and vminob.wrongatom == False:
            status = "done"

            with open(final_output_file, "a") as f:
                f.write("%s;%s;%s\n" %(name,vminob.v_min,vminob.r_min))
            return vminob

        elif vminob.on_edge == True and iteration <3 and status != "stdcube":
            status = "follow_edge"
            logger.debug('get_vmin: changed status to %s', status)
        else:
            if status != "stdcube":
                iteration = 0
                status = "stdcube"
                logger.debug('get_vmin: changed status to %s', status)

            elif iteration == 3:
                status = "done"
                logger.debug('get_vmin: changed status to %s after %d iterations', status, iteration)

                # Check if any of the three stdcubes found an actual Vmin
                foundmin = np.argwhere(np.array([not(i.on_edge or i.wrongatom) for i in vminobjects[-3:]]) == True )

                if len(foundmin) > 0:
                    mincubeno = np.argmin([vminobjects[-3 + i].v_min for i in foundmin])
                else:
                    mincubeno = np.argmin([i.v_min for i in vminobjects[-3:]])

                mincube = vminobjects[-3+mincubeno]

                with open(output_file, "a") as f:
                    f.write("{};{};{};{:.5f};{};{};{:.4f};{:.4f};{:.4f}\n".format(mincube.name,mincube.suffix,mincube.v_min,mincube.r_min,mincube.on_edge,mincube.wrongatom,*mincube.coords_min_A))

                with open(final_output_file, "a") as f:
                    f.write("%s;%s;%s\n" %(mincube.name,mincube.v_min,mincube.r_min))
                return(mincube)

        iteration += 1

    return vminobjects[-1]

def write_inpcube(vminob: Vminob,
                  destination: Path) -> Path:

    writecont = " " + vminob.name + " potential=scf\n Electrostatic potential from Total SCF Density\n"
    writecont += "{0:>5}{1[0]:>12.6f}{1[1]:>12.6f}{1[2]:>12.6f}\n".format(len(vminob.coords), vminob.grid_or)
    for i in range(3):
        writecont += "{0:>5}{1[0]:>12.6f}{1[1]:>12.6f}{1[2]:>12.6f}\n".format(vminob.npoints[i], vminob.grid_coords[i])

    with open(destination, "w", newline="\n") as f:
        f.write(writecont)
    return destination

def run_cubman(cube_file: Path) -> Path:
    '''
    Runs cubman and returns the .txt formatted ASCII file
    '''
    output_file = cube_file.with_suffix('.txt')
    a = subprocess.Popen(args='cubman',
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         cwd=cube_file.parent)

    inputargs = f"to\n{cube_file.name}\ny\n{output_file.name}"

    # Args for testing
    # to y 90000001_noNi_00000_Pesp_out.cub y 90000001_noNi_00000_Pesp_out.txt
    a_out = a.communicate(input=inputargs.encode())

    return output_file

def get_geom_fch(vminob: Vminob, file: Path):

    if '.fch' not in file.suffix:
        raise ValueError(f'This function is meant to be called on a .fch or .fchk file not {file.name}')

    if not file.exists():
        raise FileNotFoundError(f'{file.absolute()} does not exist')

    get_permission(file)

    with open(file) as f:
        natoms = 250
        content = []
        for line in f:
            content.append(line)
            if len(content) == 3:
                natoms = int(content[-1].split()[-1])
            elif "Atomic numbers" in line:
                atomnumbersstart = len(content)-1
            elif "Nuclear charges" in line:
                atomnumbersend = len(content)-1
            elif "Current cartesian coordinates" in line:
                coordsstart = len(content)-1
            elif "Number of symbols" in line:
                coordsend = len(content)-1
                break
            elif "Force Field" in line:
                coordsend = len(content)-1
                break

    atoms,atomnos = [],[]
    for line in range(atomnumbersstart+1,atomnumbersend):
        atomnos += content[line].split()
    for atom in atomnos:
        if atom in vmin_elements:
            atoms.append(elements[atom]),
        else:
            atoms.append(atom)

    coords = []
    coordsstream = []
    for line in range(coordsstart+1,coordsend):
        coordsstream += content[line].split()
    coordsstream = [float(i) for i in coordsstream]
    for atom in range(natoms):
        coords.append([atoms[atom]]+coordsstream[atom*3:atom*3+3])

    return coords

