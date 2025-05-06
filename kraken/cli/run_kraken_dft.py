#!/usr/bin/env python3
# coding: utf-8

import re
import sys
import yaml
import shutil
import logging
import argparse
import itertools

from yaml import CLoader as Loader
from yaml import CDumper as Dumper

from pathlib import Path
import pandas as pd
import numpy as np

from numpy.typing import NDArray

from pprint import PrettyPrinter

import morfeus
import morfeus.utils

from kraken.dft_properties import get_e_hf, get_homolumo, get_enthalpies, get_nimag, get_nbo
from kraken.dft_properties import get_nbo_orbsP, get_nmr, get_dipole, get_quadrupole
from kraken.dft_properties import get_efg, get_nuesp, get_edisp, get_ecds, get_time

from kraken.file_io import get_outstreams
from kraken.file_io import write_xyz, log_to_xyz, xyz_to_sdf

from kraken.dft_pint import pint_pdb, compute_pint

from kraken.dft_utils import get_filecont, make_fchk_file, get_coordinates_and_elements_from_logfile
from kraken.dft_utils import get_conmat, split_log, run_multiwfn
from kraken.dft_utils import HARTREE_TO_KCAL, BOHR_TO_ANGSTROM, R_GAS_CONSTANT_KCAL_PER_MOL_KELVIN, TEMPERATURE_IN_KELVIN

from kraken.dft_vmin import get_vmin

from kraken.dft_dispersion import run_dftd3, run_dftd4, read_disp

from kraken.dft_duplicate_detection import get_rmsd_matrix

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
    datefmt='%m/%d/%Y:%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

DESCRIPTION = r'''
╔══════════════════════════════════════╗
║   | |/ / _ \  /_\ | |/ / __| \| |    ║
║   | ' <|   / / _ \| ' <| _|| .` |    ║
║   |_|\_\_|_\/_/ \_\_|\_\___|_|\_|    ║
╚══════════════════════════════════════╝
Kolossal viRtual dAtabase for moleKular dEscriptors
of orgaNophosphorus ligands.


This is the second script required to run the Kraken
workflow. This script accepts a Kraken ID and a directory
that contains a subdirectory with the specified Kraken ID.
The <KRAKEN_ID>/dft/ folder must contain the .log and .chk
files required to compute DFT properties.

The script can be called directly in the terminal or
called in a shell script that is submitted to SLURM
or some other job scheduler.
'''

DESCRIPTION = '\n'.join(line.center(80) for line in DESCRIPTION.strip('\n').split('\n'))

def get_args() -> argparse.Namespace:
    '''Gets the arguments for running Kraken'''

    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, 2, 80),
        allow_abbrev=False,
        usage=argparse.SUPPRESS,
        add_help=False)

    parser.add_argument('-h', '--help',
                        action='help',
                        default=argparse.SUPPRESS,
                        help='Show this help message and exit.\n\n')

    parser.add_argument('-k', '--kid',
                        dest='kraken_id',
                        required=True,
                        type=str,
                        help='Input Kraken ID for which the DFT processing is done\n\n',
                        metavar='INT')

    parser.add_argument('-d', '--dir',
                        dest='datadir',
                        required=True,
                        type=str,
                        help='Directory in which the Kraken ID folder exists\n\n',
                        metavar='DIR')

    parser.add_argument('-n', '--nprocs',
                        dest='nprocs',
                        default=4,
                        type=int,
                        help='Number of processors to use\n\n',
                        metavar='INT')

    parser.add_argument('--force',
                        action='store_true',
                        help='Forces the recalculation instead of reading potentially incomplete results from file\n\n')

    args = parser.parse_args()

    args.datadir = Path(args.datadir)
    if not args.datadir.exists():
        raise FileNotFoundError(f'Could not locate {args.datadir.absolute()}')

    return args

# Custom pretty printer
PRINTER = PrettyPrinter(indent=2)

MULTIWFN_EXECUTABLE = Path('/uufs/chpc.utah.edu/common/home/u6053008/kraken/kraken/executables/Multiwfn_3.7_bin_Linux_noGUI/Multiwfn')
MULTIWFN_SETTINGS_FILE = Path('/uufs/chpc.utah.edu/common/home/u6053008/kraken/kraken/executables/Multiwfn_3.7_bin_Linux_noGUI/settings.ini')

DFTD3_EXECUTABLE = Path('/uufs/chpc.utah.edu/common/home/u6053008/kraken/kraken/executables/dftd3')
DFTD4_EXECUTABLE = Path('/uufs/chpc.utah.edu/common/home/u6053008/kraken/kraken/executables/dftd4')

numbers_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")

# which energies to read from which log-file
ENERGY_LOG_DICT = {'freq': 'e_dz',
                   'nbo': 'e_tz_gas',
                   'sp': 'e_tz_gas',
                   'solv': 'e_tz_solv',
                   'ra': 'e_tz_ra',
                   'rc': 'e_tz_rc'
}

# which properties to read from which log-file
PROP_LOG_DICT = {'freq':['nimag','g','t'],
                 'sp'  :['dipole','homo','qpole','t'],
                 'ra'  :['homo','nbo','t'],
                 'rc'  :['homo','nbo','t'],
                 'nbo' :['nbo','nborbsP','t'],
                 'nmr' :['nmr','t'],
                 'efg' :['efg','nuesp','t'],
                 'solv':['ecds','t'],
                 }

# Properties to Boltzmann weight
# Potentiall other properties: "vv_total_visible_volume",
# "vv_proximal_visible_volume","vv_distal_visible_volume",
# "vv_ratio_visible_total","vv_ratio_proxvis_total",
BOLTZPROPERTIES = ['vmin_vmin', 'vmin_r', 'dipolemoment', 'fmo_e_homo',
                   'fmo_e_lumo', 'fmo_mu', 'fmo_eta', 'fmo_omega',
                   'somo_ra', 'somo_rc', 'qpole_amp', 'qpoletens_xx',
                   'qpoletens_yy', 'qpoletens_zz', 'nbo_P', 'nbo_P_ra',
                   'spindens_P_ra', 'nbo_P_rc', 'spindens_P_rc', 'nmr_P',
                   'nmrtens_sxx_P', 'nmrtens_syy_P', 'nmrtens_szz_P',
                   'efg_amp_P', 'efgtens_xx_P', 'efgtens_yy_P', 'efgtens_zz_P',
                   'nuesp_P', 'E_solv_cds', 'nbo_lp_P_percent_s', 'nbo_lp_P_occ',
                   'nbo_lp_P_e', 'nbo_bd_e_max', 'nbo_bd_e_avg', 'nbo_bds_e_min',
                   'nbo_bds_e_avg', 'nbo_bd_occ_min', 'nbo_bd_occ_avg', 'nbo_bds_occ_max',
                   'nbo_bds_occ_avg', 'E_solv_total', 'E_solv_elstat', 'E_oxidation',
                   'E_reduction', 'fukui_p', 'fukui_m', 'pyr_P', 'pyr_alpha', 'vbur_vbur',
                   'vbur_vtot', 'vbur_ratio_vbur_vtot', 'vbur_qvbur_min', 'vbur_qvbur_max',
                   'vbur_qvtot_min', 'vbur_qvtot_max', 'vbur_max_delta_qvbur',
                   'vbur_max_delta_qvtot', 'vbur_ovbur_min', 'vbur_ovbur_max',
                   'vbur_ovtot_min', 'vbur_ovtot_max', 'vbur_near_vbur', 'vbur_far_vbur',
                   'vbur_near_vtot', 'vbur_far_vtot', 'sterimol_B1', 'sterimol_B5', 'sterimol_L',
                   'sterimol_burB1', 'sterimol_burB5', 'sterimol_burL', 'Pint_P_int', 'Pint_dP',
                   'Pint_P_min', 'Pint_P_max', 'volume', 'surface_area', 'sphericity']

# Condensed properties list
# Potential other properties to compute condensed values:
# "vv_total_visible_volume", "vv_proximal_visible_volume",
# "vv_distal_visible_volume", "vv_ratio_visible_total", "vv_ratio_proxvis_total",
MMPROPERTIES = ['dipolemoment', 'qpole_amp', 'qpoletens_xx', 'qpoletens_yy', 'qpoletens_zz',
                'pyr_P', 'pyr_alpha', 'vbur_vbur', 'vbur_vtot', 'vbur_qvbur_min', 'vbur_qvbur_max',
                'vbur_qvtot_min', 'vbur_qvtot_max', 'vbur_max_delta_qvbur', 'vbur_max_delta_qvtot',
                'vbur_ovbur_min', 'vbur_ovbur_max', 'vbur_ovtot_min', 'vbur_ovtot_max', 'vbur_near_vbur',
                  'vbur_far_vbur', 'vbur_near_vtot', 'vbur_far_vtot', 'sterimol_B1', 'sterimol_B5',
                  'sterimol_L', 'sterimol_burB1', 'sterimol_burB5', 'sterimol_burL']


jobtypes = {'e':          [get_e_hf, 'streams'],
            'homo':       [get_homolumo, 'filecont'],
            'g':          [get_enthalpies, 'filecont'],
            'nimag':      [get_nimag, 'streams'],
            'nbo':        [get_nbo, 'filecont'],
            'nborbsP':    [get_nbo_orbsP, 'filecont'],
            'nmr':        [get_nmr, 'filecont'],
            'dipole':     [get_dipole, 'streams'],
            'qpole':      [get_quadrupole, 'streams'],
            'efg':        [get_efg, 'filecont'],
            'nuesp':      [get_nuesp, 'filecont'],
            'edisp':      [get_edisp, 'filecont'],
            'ecds':       [get_ecds, 'filecont'],
            't':          [get_time, 'filecont'],
}

# assign names to each descriptor
propoutput = {'freq_g':      ['', 'g'],
              'freq_nimag':  ['nimag'],
              'sp_dipole': ['dipolemoment',],
              'sp_homo':   ['fmo_e_homo','fmo_e_lumo','fmo_mu','fmo_eta','fmo_omega'],
              'ra_homo':['somo_ra','','','',''],
              'rc_homo':['somo_rc','','','',''],
              'sp_qpole':  ['qpole_amp','qpoletens_xx','qpoletens_yy','qpoletens_zz'],
              'nbo_nbo':    ['nbo_P'],
              'ra_nbo': ['nbo_P_ra','spindens_P_ra'],
              'rc_nbo': ['nbo_P_rc','spindens_P_rc'],
              'nmr_nmr':    ['nmr_P','nmrtens_sxx_P','nmrtens_syy_P','nmrtens_szz_P',],
              'efg_efg':    ['efg_amp_P','efgtens_xx_P','efgtens_yy_P','efgtens_zz_P'],
              'efg_nuesp':  ['nuesp_P',],
              'solv_ecds':  ['E_solv_cds'],
              'nbo_dipole': ['dipolemoment',],
              'nbo_homo':   ['fmo_e_homo','fmo_e_lumo','fmo_mu','fmo_eta','fmo_omega'],
              'nbo_qpole':  ['qpole_amp','qpoletens_xx','qpoletens_yy','qpoletens_zz'],
}

def add_valence(elements: np.ndarray,
                coords: np.ndarray,
                conmat: np.ndarray,
                base_idx: int,
                add_element="Pd") -> tuple[np.ndarray, np.ndarray]:

    '''
    Adds a valence to base so that the angle to the previous substituents is maximized
    and REORDERS the coordinate output for "convenience"

    add_element: add any of the following elements:
        O, Se, Pd, and X

    Returns
    ----------
    tuple[np.ndarray, np.ndarray]
        The elements and coordinates of the new complex
    '''

    # tTypicalypical bond distances to P
    distpx = {"O": 1.5,
              "Se": 2.12,
              "Pd": 2.28,
              "X": 1.8}

    # Get number of atoms
    num_atoms = len(elements)

    # Get the coordinates of the specific atom
    # to which we will add the element
    coord_base = coords[base_idx]
    base_element = elements[base_idx]

    # Define the vectory as just zeroes for now
    vec = np.array([0.0, 0.0, 0.0])

    bonded = []
    for atom_index in range(num_atoms):

        if conmat[base_idx][atom_index]:
            logger.debug('Atom index %d (%s) is bound to %d (%s)', atom_index, elements[atom_index], base_idx, elements[base_idx])

            bonded.append(atom_index)

            vec += coord_base - coords[atom_index]

    # Bond distance type the normal vector plus the coords of the base atom
    coordox = distpx[add_element] * vec / np.linalg.norm(vec) + coord_base

    atoms = [x for x in range(num_atoms + 1)]

    coords_temp = np.vstack((coords,coordox))
    elements_temp = np.concatenate((elements, np.array([add_element])))

    test_conmat = get_conmat(elements=elements_temp, coords=coords_temp)

    if sum(test_conmat[-1]) != 1.0:
        logger.warning('Possible collision when adding Pd coordination in add_valence!')

    # Sort coordinates so that base is first
    # add_element is second,
    # and the other atoms bonded to base are next
    elements_new = [base_element,add_element] + [elements[a] for a in bonded] + [a for i,a in enumerate(elements) if i not in [base_idx]+bonded]
    coords_new = np.vstack((coord_base, coordox, coords[bonded], coords[[i for i,a in enumerate(elements) if i not in [base_idx]+bonded]]))

    return np.array(elements_new), coords_new

def gp_properties(sublogs: list[Path],
                  conformer_name: str,
                  p_idx: int):
    '''
    Gets all the properties found in Gaussian 16 log files
    '''
    logger.info('Getting G16 properties for conformer %s', conformer_name)
    # reads gaussian log files
    gpdict = {}
    gpdict["properties"] = {}

    # Define a dict for the contents
    contents = {'streams': {},
                'filecont': {},
    }

    for _sublog in sublogs:

        # Get the sublog type
        type_of_sublog = _sublog.stem.split('_')[-1]

        # Consistent with the old workflow, only certain enregies
        # are collected and the others are ignored
        if type_of_sublog not in ENERGY_LOG_DICT.keys():
            #logger.warning('Energy of sublog suffix %s is not collected', type_of_sublog)
            continue
        energy_to_retrieve = ENERGY_LOG_DICT[type_of_sublog]

        contents['streams'][type_of_sublog] = get_outstreams(file=_sublog)

        if contents['streams'][type_of_sublog] == 'failed or incomplete job':
            logger.error('Failed or incomplete job for %s', _sublog.name)
            return {'error': True}
        else:
            gpdict[energy_to_retrieve] = get_e_hf(contents["streams"][type_of_sublog])

    # Set the error to False at this point
    gpdict['error'] = False

    # Going through each log file, get the relevant properties
    for _sublog in sublogs:

        # Get the type of the sublog
        type_of_sublog = _sublog.stem.split('_')[-1]

        # Consistent with the old workflow, no properties are
        # collected from the optimization portion
        if type_of_sublog not in PROP_LOG_DICT.keys():
            #logger.warning('Properties of sublog suffix %s are not collected', type_of_sublog)
            continue

        # Gets a list of properties like nimag, g, t, etc...
        properties_to_retrieve = PROP_LOG_DICT[type_of_sublog]

        contents['filecont'][type_of_sublog] = get_filecont(_sublog)

        for prop in properties_to_retrieve:

            # This is calling the function jobtypes[prop] which is a function
            gpresults = jobtypes[prop][0](contents[jobtypes[prop][1]][type_of_sublog], p_idx)

            logger.debug('log: %-8s\tproperty: %-8s\tval: %s', type_of_sublog, prop, PRINTER.pformat(gpresults))

            if prop == "nborbsP":   # NBO orbital analysis returns a dictionary with the proper labels
                gpdict["properties"].update(gpresults)
            elif prop == "t":       # Time to complete one of the "subjobs"
                gpdict[f"{type_of_sublog}_t"] = gpresults
            elif prop in ["e_dz","g", "e_tz_gas", "e_tz_solv", "e_tz_ra", "e_tz_rc", "nimag"]:
                gpdict.update({propoutput[f"{type_of_sublog}_{prop}"][i]: float(gpresults[i]) for i in range(len(gpresults))})
            else: # all other functions return a list. This is assigned into a dict with proper names here
                gpdict["properties"].update({propoutput[f"{type_of_sublog}_{prop}"][i]: float(gpresults[i]) for i in range(len(gpresults))})

    gpdict["g_tz_gas"]  = gpdict["g"] - gpdict["e_dz"] + gpdict["e_tz_gas"] # in Hartree
    gpdict["g_tz_solv"] = gpdict["g"] - gpdict["e_dz"] + gpdict["e_tz_solv"] # in Hartree
    gpdict["properties"]["E_solv_total"] = (gpdict["e_tz_solv"] - gpdict["e_tz_gas"]) * HARTREE_TO_KCAL # in kcal/mol
    gpdict["properties"]["E_solv_elstat"] = gpdict["properties"]["E_solv_total"] - gpdict["properties"]["E_solv_cds"] # in kcal/mol
    gpdict["properties"]["E_oxidation"] = gpdict["e_tz_rc"] - gpdict["e_tz_gas"] # in Hartree
    gpdict["properties"]["E_reduction"] = gpdict["e_tz_ra"] - gpdict["e_tz_gas"] # in Hartree
    gpdict["properties"]["fukui_p"] = gpdict["properties"]["nbo_P"]-gpdict["properties"]["nbo_P_ra"] # fukui electrophilicity
    gpdict["properties"]["fukui_m"] = gpdict["properties"]["nbo_P_rc"]-gpdict["properties"]["nbo_P"] # fukui nucleophilicity

    gpdict["t_total"] = sum([gpdict[f"{log}_t"] for log in PROP_LOG_DICT.keys()])
    if "" in gpdict.keys():
        del gpdict[""]
    if "" in gpdict["properties"].keys():
        del gpdict["properties"][""]
    return gpdict

def get_morfeus_props(elements: np.ndarray,
                      coordinates: np.ndarray,
                      phosphorus_valence: int) -> dict:
    '''
    Gets MORFEUS properties on two arrays: one of elements and one
    of coordinates. In the original Kraken workflow, this was done
    on coordinates_pd which are the coordinates containing an additional
    Pd atom bound to phosphorus.
    '''

    # Define a dictionary to hold the MORFEUS properties
    morfdict = {}

    # Create a dictionary that we will write to a file and read
    # to skip calculations if the calculations are already completed.

    if phosphorus_valence == 3:

        logger.info('Computing pyramidalization values')

        # Check that we have the correct set of coordinates consistent with old code
        if elements[0] != 'P':
            raise ValueError(f'Found element {elements[0]} instead of P in elements index 0 for MORFEUS properties')
        if elements[1] != 'Pd':
            raise ValueError(f'Found element {elements[1]} instead of Pd in elements index 1 for MORFEUS properties')

        # Note that we're passing in 1-indexed atom
        # numbers here for phosphorus (1) and Pd (2)
        pyr = morfeus.Pyramidalization(coordinates=coordinates,
                                       elements=elements,
                                       atom_index=1,
                                       excluded_atoms=[2])

        # Add the pyramidalization parameters
        morfdict['pyr_P'] = float(pyr.P)
        morfdict['pyr_alpha'] = float(pyr.alpha)
    else:
        logger.error('phosphorus_valence was %d not 3. Setting pyramidalization values to None', int(phosphorus_valence))
        morfdict['pyr_P'] = None
        morfdict['pyr_alpha'] = None

    # Compute vbur_vbur
    # Buried volume - get quadrant volumes and distal volume
    # Iterate through P-substituents, aligning the quadrants parallel to each once (= xz_plane definition)
    # Metal/point of reference should be 2.28 A away from P
    # z_axis_atoms: P
    # xz_plane_atoms: each of the substituents once
    # keep lowest and highest quadrant and octant volume across all three orientations of the coordinate system
    # keep highest difference of any neighboring quadrant volume
    # keep volume in each of the two hemispheres

    # Arrays and lists for holding all the data
    qvbur_all = np.array([])
    qvdist_all = np.array([])
    qvtot_all = np.array([])
    max_delta_qvbur_all = []
    max_delta_qvtot_all = []
    ovbur_all = np.array([])
    ovtot_all = np.array([])

    logger.info('Computing buried volume')

    # Iterate over the phosphorus substituents
    for phos_substituent_index in range(3):

        # Make the volume calculator
        volume = morfeus.BuriedVolume(coordinates=coordinates,
                                      elements=elements,
                                      metal_index=2,
                                      excluded_atoms=[2],
                                      z_axis_atoms=[1],
                                      xz_plane_atoms=[3 + phos_substituent_index],
                                      radii_type='bondi',
                                      radii_scale=1.17,
                                      include_hs=False)

        # Perform octant_analysis and distal volume measurements
        volume.octant_analysis()
        volume.compute_distal_volume(method='buried_volume', octants=True)

        # Get the vbur of the complex
        vbur = volume.buried_volume   # vbur should be identical for each iteration
        vdist = volume.distal_volume  #
        vtot = vbur + vdist           #

        qvbur = np.asarray(list(volume.quadrants["buried_volume"].values()))
        qvdist = np.asarray(list(volume.quadrants["distal_volume"].values()))
        qvtot = qvbur + qvdist

        qvbur_all = np.append(qvbur_all,qvbur)
        qvtot_all = np.append(qvtot_all,qvtot)

        max_delta_qvbur_all.append(max([abs(qvbur[j] - qvbur[j - 1]) for j in range(4)]))
        max_delta_qvtot_all.append(max([abs(qvtot[j] - qvtot[j - 1]) for j in range(4)]))

        ovbur = np.asarray(list(volume.octants["buried_volume"].values()))
        ovdist = np.asarray(list(volume.octants["distal_volume"].values()))
        ovtot = ovbur + ovdist

        ovbur_all = np.append(ovbur_all, ovbur)
        ovtot_all = np.append(ovtot_all, ovtot)

        near_vbur = ovbur[4:].sum()   # these are identical for each iteration
        far_vbur = ovbur[:4].sum()    #
        near_vtot = ovtot[4:].sum()   #
        far_vtot = ovtot[:4].sum()    #

        # Add everything to MORFEUS dictionary
        morfdict["vbur_vbur"] = vbur
        morfdict["vbur_vtot"] = float(vtot)
        morfdict["vbur_ratio_vbur_vtot"] = float(vbur / vtot)

        morfdict["vbur_qvbur_min"] = float(min(qvbur_all))
        morfdict["vbur_qvbur_max"] = float(max(qvbur_all))
        morfdict["vbur_qvtot_min"] = float(min(qvtot_all))
        morfdict["vbur_qvtot_max"] = float(max(qvtot_all))

        morfdict["vbur_max_delta_qvbur"] = float(max(max_delta_qvbur_all))
        morfdict["vbur_max_delta_qvtot"] = float(max(max_delta_qvtot_all))

        morfdict["vbur_ovbur_min"] = float(min(ovbur_all))
        morfdict["vbur_ovbur_max"] = float(max(ovbur_all))
        morfdict["vbur_ovtot_min"] = float(min(ovtot_all))
        morfdict["vbur_ovtot_max"] = float(max(ovtot_all))

        morfdict["vbur_near_vbur"] = float(near_vbur)
        morfdict["vbur_far_vbur"]  = float(far_vbur)
        morfdict["vbur_near_vtot"] = float(near_vtot)
        morfdict["vbur_far_vtot"]  = float(far_vtot)
    '''
    else:

        morfdict["vbur_vbur"] = None
        morfdict["vbur_vtot"] = None
        morfdict["vbur_ratio_vbur_vtot"] = None

        morfdict["vbur_qvbur_min"] = None
        morfdict["vbur_qvbur_max"] = None
        morfdict["vbur_qvtot_min"] = None
        morfdict["vbur_qvtot_max"] = None

        morfdict["vbur_max_delta_qvbur"] = None
        morfdict["vbur_max_delta_qvtot"] = None

        morfdict["vbur_ovbur_min"] = None
        morfdict["vbur_ovbur_max"] = None
        morfdict["vbur_ovtot_min"] = None
        morfdict["vbur_ovtot_max"] = None

        morfdict["vbur_near_vbur"] = None
        morfdict["vbur_far_vbur"]  = None
        morfdict["vbur_near_vtot"] = None
        morfdict["vbur_far_vtot"]  = None
    '''

    logger.info('Computing Sterimol descriptors')

    # Compute Sterimol matching Rob Paton's implementation
    patonradii = morfeus.utils.get_radii(elements, radii_type="bondi")
    patonradii = np.array(patonradii)

    # Make this correction?
    patonradii[patonradii == 1.2] = 1.09

    # Create the Sterimol object
    sterimol = morfeus.Sterimol(elements=elements,
                                coordinates=coordinates,
                                dummy_index=2,
                                attached_index=1,
                                radii=patonradii,
                                n_rot_vectors=3600)

    morfdict["sterimol_B1"] = float(sterimol.B_1_value)
    morfdict["sterimol_B5"] = float(sterimol.B_5_value)
    morfdict["sterimol_L"]  = float(sterimol.L_value)

    # Get the buried Sterimol
    sterimol_bur = morfeus.Sterimol(elements=elements,
                                    coordinates=coordinates,
                                    dummy_index=2,
                                    attached_index=1,
                                    calculate=False,
                                    radii=patonradii,
                                    n_rot_vectors=3600)

    sterimol_bur.bury(sphere_radius=5.5, method="delete", radii_scale=0.5)

    morfdict["sterimol_burB1"] = float(sterimol_bur.B_1_value)
    morfdict["sterimol_burB5"] = float(sterimol_bur.B_5_value)
    morfdict["sterimol_burL"]  = float(sterimol_bur.L_value)

    return morfdict

def get_conformer_properties(main_logfile: Path,
                             optim_log: Path,
                             freq_log: Path,
                             sp_log: Path,
                             nmr_log: Path,
                             efg_log: Path,
                             nbo_log: Path,
                             solv_log: Path,
                             rc_log: Path,
                             ra_log: Path,
                             formatted_chk_file: Path,
                             parent_error_file: Path,
                             dispersion_model: str,
                             ded_output_file: Path,
                             multiwfn_output_file: Path,
                             dispersion_output_file: Path,
                             nprocs: int = 1) -> tuple[dict, list]:
    '''
    Gets all the properties for a particular conformer
    '''

    # Check validity of args
    dispersion_model = dispersion_model.upper()
    if dispersion_model not in ['D3', 'D4']:
        raise ValueError(f'dispersion_model must be either D3 or D4 not {dispersion_model}')

    # Make a dictionary to hold all of the conformer data
    confdata = {}

    # List for holding misc errors
    errors = []

    # Get the elements and coords
    coords, elements = get_coordinates_and_elements_from_logfile(file=optim_log)

    # Get the conmat
    conmat = get_conmat(elements=elements, coords=coords)

    # Get the index of the phosphorus atom
    # This check removes quaternary phosphorus atoms (phosphonium, phosphate, etc.) but allows
    p_idx = [i for i in range(len(elements)) if elements[i] == "P" and sum(conmat[i]) <= 3][0]

    logger.info('The phosphorus index is %d', p_idx)

    # Add "Pd" at the reference position in the P-lone pair region
    elements_pd, coordinates_pd = add_valence(elements=elements,
                                              coords=coords,
                                              conmat=conmat,
                                              base_idx=p_idx,
                                              add_element="Pd")

    # Make a file for the palladated coordinates
    palladated_species_file = Path(optim_log.parent / f'{optim_log.stem}_Pd.xyz')
    if not palladated_species_file.exists():
        logger.info('Writing palladated species to %s', palladated_species_file.absolute())
        write_xyz(destination=palladated_species_file,
                  coords=coordinates_pd,
                  elements=elements_pd)
    else:
        logger.warning('Using existing palladated species file %s', palladated_species_file.absolute())

    # Convert the optimized file to a .sdf file
    optimized_xyz_file = Path(optim_log.parent / f'{optim_log.stem}.xyz')
    optimized_sdf_file = Path(optim_log.parent / f'{optim_log.stem}.sdf')

    write_xyz(destination=optimized_xyz_file,
              coords=coords,
              elements=elements)

    xyz_to_sdf(xyz_file=optimized_xyz_file,
                destination=optimized_sdf_file)

    # Save some info to the confdata dict
    confdata['coords'] = coords
    confdata['coords_pd'] = coordinates_pd.tolist()
    confdata['elements'] = elements
    confdata['elements_pd'] = elements_pd
    confdata['conmat'] = conmat.tolist()
    confdata['p_idx'] = p_idx
    confdata['p_val'] = int(sum(conmat[p_idx]))  # how many substituents at phosphorus

    # Make another key for properties
    confdata['properties'] = {}

    gp_props = gp_properties(sublogs=[optim_log, freq_log,sp_log, nmr_log, efg_log, nbo_log, solv_log, ra_log, rc_log],
                             conformer_name=main_logfile.stem,
                             p_idx=p_idx)

    confdata.update(gp_props)

    # If there is some conformer-level error
    if confdata['error']:
        logger.error('Error for logfile %s: %s', main_logfile.name, str(confdata['error']))
        errors.append(f'Error from conformer {main_logfile.name}. Check .log files.')

        with open(parent_error_file, 'a', encoding='utf-8') as o:
            o.write(f'{parent_error_file.stem};{main_logfile.name};{errors[-1]}')

        # Dump the yaml at this point
        with open(main_logfile.parent / f'{main_logfile.stem}_data.yml', 'w', encoding='utf-8') as o:
            yaml.dump(confdata, o, Dumper=Dumper)

        return confdata, errors

    logger.info('Computing MORFEUS properties on %s', main_logfile.name)

    morfeus_props = get_morfeus_props(elements=elements_pd,
                                      coordinates=coordinates_pd,
                                      phosphorus_valence=confdata['p_val'])

    confdata['properties'].update(morfeus_props)

    # In the old code, Pint was read in from a file here and
    # placed into confdata, but because Pint was computed seperately
    # This reads in 10 values but we only assign 7 (for some reason)
    pint_read = read_dedout(file=ded_output_file) + read_multiwfnout(multiwfn_output_file) + read_disp(file=dispersion_output_file, disp='d3')

    pint_descriptor_labels = ['Pint_P_int', 'Pint_dP', 'Pint_P_min', 'Pint_P_max', 'volume', 'surface_area', 'sphericity']

    assert len(pint_read[:7]) == len(pint_descriptor_labels)

    for label, value in zip(pint_descriptor_labels, pint_read[:7]):
        confdata['properties'][label] = float(value)

    # Get the Vmin
    logger.info('Computing vmin')
    vmin_object = get_vmin(fchk=formatted_chk_file, nprocs=nprocs, runcub=True)
    confdata['properties']["vmin_vmin"] = float(vmin_object.v_min)
    confdata['properties']["vmin_r"] = float(vmin_object.r_min)

    return confdata, errors

def read_dedout(file: Path) -> list[float]:
    with open(file, 'r', encoding='utf-8') as f:
        ded_cont = f.readlines()
    ded_results = [float(ded_cont[i].split()[-1]) for i in range(2, 6)]
    return ded_results

def read_multiwfnout(file: Path) -> list[float | None]:

    '''
    Reads something like "{name}_Multiwfn_out.txt"

    Returns the volume (angstrom^3), area (angstrom^2), and sphericity
    '''
    multiwfn_float_pattern = re.compile(r"[-+]?\d*\.\d+")

    try:
        with open(file, 'r', encoding='utf-8') as f:
            mwfn_cont = f.readlines()

        mwfn_pat = "================= Summary of surface analysis ================="
        for ind, line in enumerate(mwfn_cont[::-1]):
            if mwfn_pat in line:
                vol = float(multiwfn_float_pattern.findall(mwfn_cont[-ind + 1])[-1])  # in Angstrom^3
                area = float(multiwfn_float_pattern.findall(mwfn_cont[-ind + 4])[-1]) # in Angstrom^2
                sph = np.round(np.pi ** (1/3) * (6 * vol) ** (2/3) / area, 6) # Sphericity
                return [vol, area, sph]
    except:
        return [None, None, None]

    return [None, None, None]

def run_conformer_properties(logfile: Path,
                             dispersion_model: str,
                             coefficient_selection_value: int,
                             parent_error_file: Path,
                             charge: int,
                             nprocs: int,
                             force: bool) -> Path:
    '''
    Runs the property collection and computations for a single conformer

    Returns the path of the conformer .yml file
    '''

    # Make a bunch of files
    chkfile = logfile.with_suffix('.chk')

    fchkfile = chkfile.with_suffix('.fchk')
    if not fchkfile.exists() or force:
        logger.info('Making .fchk file from %s', chkfile.name)
        fchkfile = make_fchk_file(file=chkfile, dest=fchkfile)
    else:
        logger.info('Reading existing .fchk file from %s', fchkfile.name)

    conformer_yml_file = logfile.parent / f'{logfile.stem}_data.yml'

    # Names for wavefunction, vertex, and output files from multiwfn
    wfn = logfile.with_suffix('.wfn')
    vtx = logfile.parent / f'{logfile.stem}_vtx.txt'
    multiwfn_output_file = logfile.parent / f'{logfile.stem}_multiwfn.output'

    logger.info('Running multiwfn')
    wfn, vtx, multiwfn_output_file = run_multiwfn(file=fchkfile,
                                                  multiwfn_executable=MULTIWFN_EXECUTABLE,
                                                  multiwfn_settings_file=MULTIWFN_SETTINGS_FILE)

    # Read in the logfile
    with open(logfile, 'r', encoding='utf-8') as _:
        logtext = _.read()

    # Split the logfiles just like the old workflow
    logger.info('Splitting parent logfile %s', logfile.name)
    split_log_files = split_log(logfile)

    # Check that all of the logfiles are there
    optim_log = logfile.parent / f'{logfile.stem}_opt.log'
    freq_log = logfile.parent / f'{logfile.stem}_freq.log'
    sp_log = logfile.parent / f'{logfile.stem}_sp.log'
    nmr_log = logfile.parent / f'{logfile.stem}_nmr.log'
    efg_log = logfile.parent / f'{logfile.stem}_efg.log'
    nbo_log = logfile.parent / f'{logfile.stem}_nbo.log'
    solv_log = logfile.parent / f'{logfile.stem}_solv.log'
    rc_log = logfile.parent / f'{logfile.stem}_rc.log'
    ra_log = logfile.parent / f'{logfile.stem}_ra.log'

    for _sublog in [optim_log, freq_log, sp_log, nmr_log, efg_log, nbo_log, solv_log, rc_log, ra_log]:
        if not _sublog.exists():
            raise FileNotFoundError(f'Could not find {_sublog.absolute()}')

    xyz = log_to_xyz(logfile)

    # Pint related files for dftd3
    pint_surface_data_file_d3 = logfile.parent / f'{logfile.stem}_ded_surf_D3.out'
    pint_pdb_file_d3 = logfile.parent / f'{logfile.stem}_ded_surf_D3.pdb'
    ded_output_file_d3 = logfile.parent / f'{logfile.stem}_ded_D3.txt'

    # Run dftd3 then dftd4
    logger.info('Running dftd3')
    d3_output_file = logfile.parent / f'{logfile.stem}_d3.out'
    d3 = run_dftd3(xyz_file=xyz, charge=charge, dftd3_executable=DFTD3_EXECUTABLE)

    logger.info('Running dftd4')
    d4 = run_dftd4(xyz_file=xyz, charge=charge, dftd4_executable=DFTD4_EXECUTABLE)

    # Do some sanity checks
    if len(d3) != len(d4):
        raise ValueError(f'The number of atoms in the d3 and d4 descriptors are not the same ({len(d3)} != {len(d4)})')

    # Get some stuff from the files above
    atoms = np.genfromtxt(fname=xyz,
                          delimiter=None,
                          usecols=(0,),
                          skip_header=2,
                          dtype=str)

    xyz_coords_bohr = np.genfromtxt(xyz, delimiter=None, usecols=(1, 2, 3), skip_header = 2) / BOHR_TO_ANGSTROM
    surf = np.genfromtxt(vtx, delimiter=None, usecols=(0, 1, 2), skip_header=1)

    # Get npoints from the surface
    npoints = len(surf)

    # Remove anchor group (this is old code from a previous version)
    #if args.anchor != None:
    #    anchor = np.genfromtxt(pl.Path(args.anchor).resolve(), delimiter=None, usecols = (0), skip_header = 0, dtype=int)
    #    anchor -= 1 # numbering in python starts with 0
    #    anchor.tolist()
    #    atoms, xyz, surf, cn = rm_anch(atoms, xyz, surf, anchor, cn)

    # Save the coefficients to a file
    output_coefficient_file_d3 = Path(logfile.parent / f'{logfile.stem}_ded_C6_D3.out')
    with open(output_coefficient_file_d3, 'w', encoding='utf-8') as o:
        for atom, d3_values in zip(atoms, d3):
            o.write(f'{str(atom):>3}  {d3_values[0]:>12.4f}')

    # Save the coefficients to a file
    output_coefficient_file_d4 = Path(logfile.parent / f'{logfile.stem}_ded_C6_D4.out')
    with open(output_coefficient_file_d4, 'w', encoding='utf-8') as o:
        for atom, d4_values in zip(atoms, d4):
            o.write(f'{str(atom):>3}  {d4_values[0]:>12.4f}')

    if dispersion_model == 'D3':
        cn = d3
    elif dispersion_model == 'D4':
        cn = d4
    else:
        raise ValueError(f'Select either D3 or D4 for dispersion model not {dispersion_model}')

    # Perform pint calculations (old code)
    # Make a file for storing raw Pint array
    pint_array_file = logfile.parent / f'{logfile.stem}_Pint.npy'

    if pint_array_file.exists():
        logger.info('Found Pint array at %s', pint_array_file.absolute())
        pint = np.load(pint_array_file)

        # Check that it's the right one
        if pint.shape[0] != npoints:
            raise ValueError(f'Found pint array of shape {pint.shape} when expecting {npoints}')
        else:
            logger.info('Pint array points %s are equal to the number of expected points %d', str(pint.shape), npoints)
    else:
        logger.info('Performing Pint calculation')
        pint = compute_pint(number_of_points=npoints,
                            number_of_atoms=len(atoms),
                            coords_bohr=xyz_coords_bohr,
                            surface_points=surf,
                            dispersion_descriptors=cn,
                            coefficient_selection=coefficient_selection_value)
        np.save(pint_array_file, pint)

    # Get the summary properties
    pint_ave = np.mean(pint)
    pint_std = np.std(pint)
    pint_max = np.mean(np.sort(pint)[-10:])  # to increase robustness
    pint_min = np.median(np.sort(pint)[:100])  # to increase robustness

    logger.info('Pint descriptors\tavg: %.6f\tstd: %f\tmax: %.6f\tmin: %.6f', pint_ave, pint_std, pint_max, pint_min)

    # Save Pint surface data
    logger.info('Writing surface data for %s', logfile.name)
    with open(pint_surface_data_file_d3, 'w', encoding='utf-8') as o:
        for ni in range(npoints):
            o.write(format(format(surf[ni][0] * BOHR_TO_ANGSTROM, '.7f'), '>14s') + "  " + format(format(surf[ni][1] * BOHR_TO_ANGSTROM, '.7f'), '>14s') + "  " + format(format(surf[ni][2] * BOHR_TO_ANGSTROM, '.7f'), '>14s') + "  " + format(format(pint[ni], '.7f'), '>14s') + "\n")

    # Don't know what this does
    fpdb = pint_pdb(xyz=pint_surface_data_file_d3, data=pint_surface_data_file_d3, head=0, col=3)
    fpdb.dump(pdb_file=pint_pdb_file_d3)

    # Save pint report (also referred to as dedout or ded_output_file)
    with open(ded_output_file_d3, 'w', encoding='utf-8') as f:
        f.write('P PARAMETERS')
        f.write('\n')
        f.write('\n')
        f.write('Pint ' + format(format(pint_ave, '.2f'), '>6s'))
        f.write('\n')
        f.write('dP ' + format(format(pint_std, '.2f'), '>6s'))
        f.write('\n')
        f.write('Pmin ' + format(format(pint_min, '.2f'), '>6s'))
        f.write('\n')
        f.write('Pmax ' + format(format(pint_max, '.2f'), '>6s'))
        f.write('\n')

    # Now we run the code from read_ligand() from PL_dft_library_201027
    confdata, errors = get_conformer_properties(logfile,
                                                optim_log,
                                                freq_log,
                                                sp_log,
                                                nmr_log,
                                                efg_log,
                                                nbo_log,
                                                solv_log,
                                                rc_log,
                                                ra_log,
                                                formatted_chk_file=fchkfile,
                                                parent_error_file=parent_error_file,
                                                dispersion_model=dispersion_model,
                                                ded_output_file=ded_output_file_d3,
                                                multiwfn_output_file=multiwfn_output_file,
                                                dispersion_output_file=d3_output_file,
                                                nprocs=nprocs)


    logger.info('Errors found when processing %s: %s', logfile.name, errors)
    # Now we have to change several properties so they are read out correctly in the yaml (mostly numpy arrays)
    #for k, v in confdata.items():
    #    if type(k)
    confdata['elements'] = [str(x) for x in confdata['elements']]
    confdata['elements_pd'] = [str(x) for x in confdata['elements_pd']]

    confdata['coords'] = [[float(z) for z in x] for x in confdata['coords']]
    confdata['coords_pd'] = [[float(z) for z in x] for x in confdata['coords_pd']]

    with open(conformer_yml_file, 'w', encoding='utf-8') as f:
        yaml.dump(data=confdata, stream=f, Dumper=Dumper)

    logger.info('Finished confomer %s', logfile.name)

    return conformer_yml_file

def run_end(kraken_id: str,
            number_of_processors: int,
            parent_folder: Path,
            force_recalculation: bool = False):
    '''

    '''

    if not parent_folder.exists():
        raise FileNotFoundError(f'Could not locate {parent_folder.absolute()}')

    if not parent_folder.stem == kraken_id:
        raise ValueError(f'The data directory stem {parent_folder.stem} does not match the kraken_id {kraken_id}')

    dft_folder = parent_folder / 'dft'

    if not dft_folder.exists():
        raise ValueError(f'DFT folder {dft_folder.absolute()} does not exist.')

    # Define an error file that will be written to if certain errors are encountered
    _parent_error_file = dft_folder / f'{kraken_id}_errors.txt'
    _charge = 0

    # Looks like the default is D3 for the old code
    # but this could be set to D4, may break code
    _dispersion_model = 'D3'

    # Set to 2 by default, forget what this does
    _coefficient_selection_value = 2

    # Get all the logfiles
    logfiles = sorted([x for x in dft_folder.glob(f'{kraken_id}*.log') if x.is_file()])

    # Make directories to hold the many files
    for logfile in logfiles:
        conf_directory = dft_folder / logfile.stem
        conf_directory.mkdir()

        for _ in dft_folder.glob(f'{logfile.stem}*'):
            if _.is_dir():
                continue
            shutil.move(_, conf_directory / _.name)

    # Get all the dirs we just made
    _dirs = sorted([x for x in dft_folder.glob(f'{kraken_id}*') if x.is_dir()])

    # Create dcit for holding the different yaml files created arranged by name
    conformer_yamls = {}

    # Define the number of conformers
    number_of_conformers = len(_dirs)

    logger.info('Found %d conformers for %s', number_of_conformers, kraken_id)

    # Iterate through the conf dirs
    for i, _dir in enumerate(_dirs):

        logger.info('Computing properties for %s with charge %d (conf %d of %d)', _dir.stem, _charge, i + 1, number_of_conformers)

        # Define the name of the conformer
        conformer_name = _dir.stem

        # Get the logfile
        g16_logfile = _dir / f'{_dir.name}.log'

        # Raise Exception if the logfile does not exist
        if not g16_logfile.exists():
            raise ValueError(f'The main logfile {g16_logfile.absolute()} does not exist')

        # Check if the conf yaml exists and skip getting properties again if it does
        conf_yaml = _dir / f'{_dir.stem}_data.yml'

        if not conf_yaml.exists() or force_recalculation:

            # Get the conformer properties
            conf_yaml = run_conformer_properties(logfile=g16_logfile,
                                                 dispersion_model=_dispersion_model,
                                                 coefficient_selection_value=_coefficient_selection_value,
                                                 parent_error_file=_parent_error_file,
                                                 charge=_charge,
                                                 nprocs=number_of_processors,
                                                 force=force_recalculation)
        else:
            logger.info('Using %s data file for conformer %s', conf_yaml.name, conformer_name)

        # Append it to this big list
        conformer_yamls[conformer_name] = conf_yaml

    # Process all the conformers into some master dictionary for the ligand
    # Make a dictionary to hold the data across all conformers and whatnot
    ligand_data = {
        'conformers_all': [x.stem for x in _dirs],
        'conformers': [x.stem for x in _dirs],  # Duplicates and computations with errors (including nimag=1) will be removed from this list
        'number_of_conformers': number_of_conformers,
        'removed_duplicates': [],
        'confdata': {},
        'boltzmann_averaged_data': {},
        'min_data': {},
        'max_data': {},
        'delta_data': {},
        'vburminconf_data': {},

    }

    # This is a weird "status" dictionary holding errors that occur at the ligand level?
    status = {"ligandlevel": [],}

    # Track the number of new confomers
    new_confs = 0

    # Read in all the yamls
    for conformer_name, conf_yaml in conformer_yamls.items():

        logger.info('Processing %s', conf_yaml.name)

        # Check if the conformer is flagged for removal or read in the yaml
        if conformer_name in ligand_data["removed_duplicates"]:
            logger.warning('Skipping %s because it is in removed_duplicates', conformer_name)
            continue
        else:
            with open(conf_yaml, 'r', encoding='utf-8') as f:
                ligand_data["confdata"][conformer_name] = yaml.load(f, Loader=Loader)

            new_confs += 1

    # If we have conformers read in
    if new_confs == 0:
        raise ValueError(f'No conformers were read in from {len(conformer_yamls)} conformer .yml files.')

    # Get the conformers with errors
    ligand_data["conformers_w_error"] = [_conf_name for _conf_name in ligand_data["conformers"] if ligand_data["confdata"][_conf_name]["error"]]

    # Get the conformers without errors
    ligand_data["conformers"] = [_conf_name for _conf_name in ligand_data["conformers"] if _conf_name not in ligand_data["conformers_w_error"]]

    # Add number of successfully completed conformer calculations
    ligand_data["number_of_conformers"] = len(ligand_data["conformers"])

    # Define a list of energies to process
    energies = ["e_dz",
                "g","e_tz_gas",
                "g_tz_gas",
                "e_tz_solv",
                "g_tz_solv"]

    # Add new entries for the ligand for the energies
    ligand_data["energies"] = {}
    ligand_data["relative_energies"] = {}

    # For each particular energy we are trying to evaluate
    for energy_type in energies:

        # Add the energy to the energy type key
        ligand_data["energies"][energy_type] = {_conf_name: ligand_data["confdata"][_conf_name][energy_type] for _conf_name in ligand_data["conformers"]}

        # Get the min and minconf
        ligand_data[f"{energy_type}_min"] = min(ligand_data["energies"][energy_type].values())
        ligand_data[f"{energy_type}_minconf"] = list(ligand_data["energies"][energy_type].keys())[np.argmin(list(ligand_data["energies"][energy_type].values()))]

        # Compute the relative energies
        ligand_data["relative_energies"][f"{energy_type}_rel"] = {_conf_name: (ligand_data["energies"][energy_type][_conf_name] - ligand_data[f"{energy_type}_min"]) * HARTREE_TO_KCAL for _conf_name in sorted(ligand_data["conformers"])}


    # Get a relative energy dataframe
    erel_df = pd.DataFrame([ligand_data["relative_energies"][f"{_energy_type}_rel"] for _energy_type in energies], index=energies).T

    # Add a dictionary for the relative energies to the yaml
    ligand_data["relative_energies_dict"] = erel_df.to_dict()


    # Find duplicates using this procedure
    #  1) Find pairs of confs within E_rel < 0.1 kcal/mol
    #       Note: relative energies seem to be much more reliable than relative free energies
    #  2) Check these pairs to also have RMSD < 0.2 A
    #  3) Remove the conformer with higher relative free energy

    duplicates_candidates = [(i, j) for i, j in itertools.combinations(ligand_data["conformers"], 2) if abs(erel_df["e_dz"].loc[i] - erel_df["e_dz"].loc[j]) < 0.1]

    try:

        # Make a list for holding RMSD values
        rmsd_values = []

        for candidate_pair in duplicates_candidates:

            # Get the Paths to the candidate pairs
            _conf1 = dft_folder / f'{candidate_pair[0]}/{candidate_pair[0]}_opt.sdf'
            _conf2 = dft_folder / f'{candidate_pair[1]}/{candidate_pair[1]}_opt.sdf'

            if not _conf1.exists():
                raise FileNotFoundError(f'Could not locate {_conf1.absolute()}')
            if not _conf2.exists():
                raise FileNotFoundError(f'Could not locate {_conf2.absolute()}')

            rmsd_matrix = get_rmsd_matrix(conformers=[_conf1, _conf2])

            # The upper right corner of a 2x2 rmsd matrix is the rmsd between two conformers
            # Cast to float to be compatible with yaml formatting instead of numpy float
            rmsd = float(rmsd_matrix[0, 1])

            logger.debug('Running RMSD duplicate detection on %s and %s\t\tRMSD:\t%.8f', candidate_pair[0], candidate_pair[1], float(rmsd))

            rmsd_values.append(rmsd)

        ligand_data["rmsd_candidates"] = {k: v for k, v in zip(duplicates_candidates, rmsd_values)}
        ligand_data["duplicates"] = [_pair for _pair in ligand_data["rmsd_candidates"] if ligand_data["rmsd_candidates"][_pair] < 0.2]

    # RDkit failed to generate Mol objects and thus could not compute RMSD, or
    # some of the internal structures in those mol files are different despite
    # actually being the same. Default to duplicate detection based on dipole
    # moment and chemical shift similarity
    except Exception as e:
        logger.error('Error in RMSD duplicate detection because %s. Trying dipole method', str(e))


        # Log this on ligand level for double-checking
        err = "Warning: RDKit error at duplicate RMSD testing. Please double check."
        status["ligandlevel"].append(err)

        with open(_parent_error_file, 'a', encoding='utf-8') as f:
            f.write(f"{kraken_id};ligandlevel;{err}\n")

        # Generate the candidates
        dipole_candidates = set([(i, j) for i,j in duplicates_candidates if abs(ligand_data["confdata"][i]["properties"]["dipolemoment"] - ligand_data["confdata"][j]["properties"]["dipolemoment"]) < 0.025])
        nmr_candidates = set([(i,j) for i,j in duplicates_candidates if abs(ligand_data["confdata"][i]["properties"]["nmr_P"] - ligand_data["confdata"][j]["properties"]["nmr_P"]) < 0.1])

        # Append these to duplicates
        ligand_data["duplicates"] = sorted(dipole_candidates & nmr_candidates)

    # Remove the duplicates
    ligand_data["removed_duplicates"] = [erel_df.loc[list(pair)]["g_tz_gas"].idxmax() for pair in ligand_data["duplicates"]]
    logger.info('Initial number of conformers: %d', len(ligand_data["conformers"]))
    ligand_data["conformers"] = [c for c in ligand_data["conformers"] if c not in ligand_data["removed_duplicates"]]
    ligand_data["number_of_conformers"] = len(ligand_data["conformers"])

    logger.info('Removed duplicates: %s', str(ligand_data["removed_duplicates"]))
    logger.info('Remaining number of conformers: %d', ligand_data["number_of_conformers"])
    logger.info('Remaining conformers: %s', str(ligand_data["conformers"]))

    # Begin Boltzmann weighting
    # Generate the Boltzmann factors
    boltzfacs = {conformer: np.exp(-erel_df["g_tz_gas"].loc[conformer] / (R_GAS_CONSTANT_KCAL_PER_MOL_KELVIN * TEMPERATURE_IN_KELVIN)) for conformer in ligand_data["conformers"]}

    # Get the sum of the factors
    Q = sum(boltzfacs.values())

    # Get the weight for the conformers
    ligand_data["boltzmann_weights"] = {conformer: float(boltzfacs[conformer] / Q) for conformer in ligand_data["conformers"] }

    # Iterate over properties we will Boltzmann weight
    for prop in BOLTZPROPERTIES:

        #logger.debug('Boltzmann weighting property %s', prop)

        # List of
        confsmissingprop = [conf for conf in ligand_data["conformers"] if prop not in ligand_data["confdata"][conf]["properties"].keys()]

        if len(confsmissingprop) == 0:

            ligand_data["boltzmann_averaged_data"][prop] = sum([ligand_data["boltzmann_weights"][conf] * ligand_data["confdata"][conf]["properties"][prop] for conf in ligand_data["conformers"]])

        else:

            logger.warning('At least 1 missing a value for %s. Setting %s_boltz  to None', prop, prop)

            for _conformer_name in confsmissingprop:
                logger.warning('%s was missing %s', _conformer_name, prop)

            #! log this as a ligand-level error with prop and confsmissingprop
            err = f"Warning: {len(confsmissingprop)}/{len(ligand_data['conformers'])} conformers missing values for property {prop}: {','.join(confsmissingprop)}."
            status["ligandlevel"].append(err)
            with open(_parent_error_file, 'a', encoding='utf-8') as f:
                f.write(f"{kraken_id};ligandlevel;{err}\n")

            ligand_data["boltzmann_averaged_data"][prop] = None

            continue

    # Generate "Condensed" properties
    ligand_data["vburminconf"] = ligand_data["conformers"][np.argmin([ligand_data["confdata"][conf]["properties"]["vbur_vbur"] for conf in ligand_data["conformers"]])]

    for prop in MMPROPERTIES:

        #logger.debug('Computing condensed properties for property %s', prop)

        proplist = [ligand_data["confdata"][conf]["properties"][prop] for conf in ligand_data["conformers"] if prop in ligand_data["confdata"][conf]["properties"].keys()]

        # if a single conformer is missing a property value, still perform min/max analysis
        # (Boltzmann-average will be None to indicate missing value(s))
        # if all confs are missing this prop, set min/max/delta to None

        if len(proplist) == 0:
            ligand_data["min_data"][prop] = None
            ligand_data["max_data"][prop] = None
            ligand_data["delta_data"][prop] = None
            ligand_data["vburminconf_data"][prop] = None
        else:
            ligand_data["min_data"][prop] = min(proplist)
            ligand_data["max_data"][prop] = max(proplist)
            ligand_data["delta_data"][prop] = ligand_data["max_data"][prop] - ligand_data["min_data"][prop]
            ligand_data["vburminconf_data"][prop] = ligand_data["confdata"][ligand_data["vburminconf"]]["properties"][prop]

    ligand_data["time_all"] = sum([ligand_data["confdata"][conf]["t_total"] for conf in ligand_data["conformers_all"] if "t_total" in ligand_data["confdata"][conf].keys()])

    # Write the results to file
    summarized_data_file = parent_folder / f'{kraken_id}_data.yml'
    conformers_data_file = parent_folder / f'{kraken_id}_confdata.yml'
    relative_energy_csv = parent_folder / f'{kraken_id}_relative_energies.csv'

    logger.info('Writing summary data file to %s', summarized_data_file.absolute())
    with open(summarized_data_file, 'w', encoding='utf-8') as f:
        yaml.dump({k:v for k,v in ligand_data.items() if k != "confdata"}, stream=f , Dumper=Dumper)

    logger.info('Writing conformer data file to %s', conformers_data_file.absolute())
    with open(conformers_data_file, 'w', encoding='utf-8') as f:
        yaml.dump(ligand_data["confdata"], stream=f, Dumper=Dumper)

    erel_df.to_csv(relative_energy_csv, sep=";")

    logger.info('Finished writing DFT data for Kraken ID %s', kraken_id)

def main():
    '''Main entrypoint'''
    args = get_args()

    run_end(kraken_id=args.kraken_id,
            number_of_processors=args.nprocs,
            parent_folder=args.datadir / args.kraken_id,
            force_recalculation=args.force)

if __name__ == "__main__":
    main()


