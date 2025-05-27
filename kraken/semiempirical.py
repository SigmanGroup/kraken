#!/usr/bin/env python3
# coding: utf-8

'''
Functions for reading CREST and xTB
files and extracting parameters from them.
'''

import re
import time
import shlex
import shutil
import logging
import subprocess

from pathlib import Path
from importlib.resources import files

import numpy as np
import pandas as pd

from .file_io import readXYZs, write_xyz
from .utils import get_dummy_positions, get_ligand_indices
from .xtb import call_xtb, _get_xtb_version_from_file, get_results_conformer

XTB_PARAM_FILE = Path(files("kraken") / "param_ipea-xtb.txt")

logger = logging.getLogger(__name__)

def _get_crest_version() -> str | None:
    '''
    Get's the version of CREST from the command line.
    '''
    VERSION_PATTERN = re.compile(r'[V|v]ersion \d+\.\d+', re.DOTALL)
    try:
        proc = subprocess.run(args=['crest', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        exit('[FATAL] CREST is not callable. Make sure it is on your PATH environment variable.')
    version = proc.stdout.decode('utf-8')
    version = re.findall(VERSION_PATTERN, version)
    if len(version) == 0:
        print('[WARNING] CREST is callable, but the version could not be checked.')
        return None
    return str(version[0]).lower()

def call_crest(file: Path,
               nprocs: int,
               reduce_output: bool,
               debug: bool = False,
               noreftopo: bool = False,
               nocross: bool = False
               ) -> None:
    '''
    Executes the CREST command
    '''

    command=f'crest {file.absolute()} --gbsa toluene -metac -nozs -T {nprocs}' # -mquick --gfnff

    # These are old crest commands?
    #command="crest %s --gbsa toluene -metac"%(filename)
    #command="crest %s -ethr %f -pthi %f -metac"%(filename, settings["max_E"], settings["max_p"])
    # crest -chrg %i is used for charges

    if debug:
        print(f'[DEBUG] In call_crest: Running {command}')

    args = shlex.split(command)

    # Run the crest command
    with open(file.parent / 'crest.log', 'a') as crest_log_file:
        proc = subprocess.run(args, stdout=crest_log_file, stderr=subprocess.PIPE, cwd=file.parent)
        stderr = proc.stderr.decode('utf-8')

        logger.debug('stderr in call_crest. Return code %d', proc.returncode)
        logger.debug('-'*80)
        logger.debug(stderr)
        logger.debug('-'*80)

    # Sleep for some god damned reason
    #time.sleep(5)

    # Remove files if requested
    if reduce_output:
        for item in file.parent.glob('*'):
            first_letter = item.name[0]
            item_name = item.name
            if first_letter == 'M' or first_letter == 'N' or item_name == "wbo" or item_name == "coord" or "_rotamers_" in item_name or ".tmp" in item_name or first_letter == '.' or item_name == "coord.original":
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    elif item.is_file():
                        item.unlink()
                except Exception as e:
                    logger.error('Could not remove %s because %s', str(item.absolute()), str(e))

    return

def run_crest(file: Path,
              nprocs: int,
              reduce_output: bool,
              smiles: str,
              metal_char: str,
              add_Pd_Cl2: bool,
              add_Pd_Cl2_PH3: bool,
              add_Ni_CO_3: bool,
              ) -> tuple[bool, bool, list, list, list, list, list[dict], list]:
    '''
    Primary function for running CREST/xTB and
    extracting relevant information.
    '''

    #crest_version = _get_crest_version()

    if not file.exists():
        raise FileNotFoundError(f'{file.absolute()} does not exist. Make the file before calling run_crest')

    crest_start_time = time.time()

    CREST_LOG_FILE = file.parent / 'crest.log'
    CREST_BEST_FILE = file.parent / 'crest_best.xyz'

    # Check if a CREST calculation is already complete for this molecule
    crest_complete = False
    if CREST_LOG_FILE.exists() and CREST_BEST_FILE.exists():
        with open(CREST_LOG_FILE, 'r', encoding='utf-8') as infile:
            text = infile.read()
            if 'CREST terminated normally.' in text:
                crest_complete = True

    # If CREST did not run, run it or else read in the CREST results
    if not crest_complete:
        call_crest(file=file,
                   nprocs=nprocs,
                   reduce_output=reduce_output)
    else:
        logger.info('Found existing CREST calculation at %s. Reading output.', CREST_LOG_FILE.absolute())

    # Extract the information from the CREST calculation
    crest_done, coords_all, elements_all, boltzmann_data = get_crest_results(file=file)

    logger.debug('crest_done: %s', crest_done)
    logger.debug('len(coords_all): %d', len(coords_all))
    logger.debug('len(elements_all): %d', len(elements_all))
    logger.debug('len(boltzmann_data): %d', len(boltzmann_data))
    logger.debug('n_atoms for first conformer: %d', len(coords_all[0]))

    # Perform some sanity checks
    if len(elements_all) == 0:
        raise ValueError(f'No conformers were found for {file.name}')

    if 'P' not in elements_all[0]:
        raise ValueError(f'No phosphorus atom found in first conformer of {file.name}')

    if not all([len(q) == len(elements_all[0]) for q in elements_all]):
        raise ValueError(f'Not all the conformers have the same number of atoms measured by elements_all!')

    if not all([len(q) == len(coords_all[0]) for q in coords_all]):
        raise ValueError(f'Not all the conformers have the same number of atoms measured by coords_all!')

    # Get the number of atoms
    natoms = int(len(coords_all[0]))

    # Get the P atom index
    #TODO This is used in some other functions and was previously stored in "settings" dictionary
    P_index=elements_all[0].index("P")
    logger.debug('run_crest: found P atom index %s', str(P_index))

    # Set xtb_done to True
    xtb_done = True

    crest_end_time = time.time()
    total_crest_time = crest_end_time - crest_start_time
    logger.info('Completed CREST in %.2f seconds', total_crest_time)

    coords_all_used = []
    elements_all_used = []
    boltzmann_data_used = []
    conf_indeces_used = []

    # If the CREST calculations are complete
    if crest_done:

        electronic_properties_conformers=[]

        # Coords all is a list of list of lists
        # The outer list is the conformer that can be accessed with conf_idx
        # The middle list is the atom of the molecule
        # The inner list made of the X, Y, and Z coordinates
        # In the Kraken revision, coords_all[conf_idx] was changed to coord_set after using the enumerate class for running the loop

        for conf_idx, coord_set in enumerate(coords_all):

            conformer_stem = f'conf_{conf_idx}'
            moldir2= file.parent / conformer_stem
            moldir2.mkdir(exist_ok=True)

            conformer_file_path = moldir2 / f'{conformer_stem}.xyz'

            skip_this_conformer = False

            logger.info('Running xTB calculation of %s/conf_%d.xyz (%d of %d)', moldir2.name, conf_idx, conf_idx + 1, len(coords_all))

            if add_Pd_Cl2_PH3 or add_Pd_Cl2 or add_Ni_CO_3:

                mask, done = get_ligand_indices(coords=np.array(coord_set),
                                                elements=elements_all[conf_idx],
                                                P_index=P_index,
                                                smiles=smiles,
                                                metal_char=metal_char)

                if not done:
                    exit()

                if add_Ni_CO_3 and len(mask) != len(coord_set) - 7:
                    logger.error('Expected mask length of 7 but got %d. Skipping this conformer.', int(len(coord_set) - len(mask)))
                    skip_this_conformer=True
            else:
                mask = []

            if not skip_this_conformer:

                done = False

                if not done:

                    # Export the coordinates as an xyz file
                    write_xyz(destination=conformer_file_path,
                              coords=coord_set,
                              elements=elements_all[conf_idx],
                              mask=mask)

                    # Run xTB for the lmo (this is the first read_xtb_file input)
                    #TODO add charge
                    xtb_lmo_log_file = conformer_file_path.parent / 'xtb.log'
                    xtb_lmo_log_file = call_xtb(file=conformer_file_path,
                                                lmo=True,
                                                vfukui=True,
                                                esp=True,
                                                vomega=False,
                                                vipea=False,
                                                output_file=xtb_lmo_log_file,
                                                charge=0,
                                                reduce_output=True,
                                                gbsa='toluene',
                                                nprocs=nprocs)

                    # Make the vipea dir
                    vipea_dir = conformer_file_path.parent / 'xtb_ipea'
                    vipea_dir.mkdir(exist_ok=True)

                    # Copy the file to the vipea dir
                    vipea_xyz_file = vipea_dir / conformer_file_path.name
                    shutil.copy2(conformer_file_path, vipea_xyz_file)

                    # Copy the local XTB_IPEA_PARAM file
                    shutil.copy2(XTB_PARAM_FILE, vipea_dir / XTB_PARAM_FILE.name)

                    xtb_vipea_log_file = vipea_dir / 'xtb_ipea.log'
                    xtb_vipea_log_file = call_xtb(file=vipea_xyz_file,
                                                  lmo=False,
                                                  vfukui=False,
                                                  esp=False,
                                                  vomega=True,
                                                  vipea=True,
                                                  output_file=xtb_vipea_log_file,
                                                  charge=0,
                                                  reduce_output=True,
                                                  gbsa='toluene',
                                                  nprocs=nprocs)

                # Get the properties we need
                xtb_done_here, muls, alphas, wils, dip, alpha, fukui, HOMO_LUMO_gap, IP_delta_SCC, EA_delta_SCC, global_electrophilicity_index, esp_profile, esp_points, occ_energies, virt_energies, nucleophilicity = get_results_conformer(regular_xtb_calculation_log=conformer_file_path.parent / 'xtb.log',
                                                                                                                                                                                                                                              vipea_xtb_calculation_log=conformer_file_path.parent/ 'xtb_ipea/xtb_ipea.log',
                                                                                                                                                                                                                                              natoms=natoms)

                # Get the lmocent file (localized molecular orbital center file)
                lmocent_file = conformer_file_path.parent / 'lmocent.coord'

                if not lmocent_file.exists():
                    logger.error('%s  (localized molecular orbitals) does not exist!', lmocent_file.absolute())
                    logger.error('Setting dummy_positions to None')
                    dummy_positions = None

                dummy_positions = get_dummy_positions(lmocent_coord_file=lmocent_file)

                logger.debug('len(dummy_positions): %d', len(dummy_positions))

                electronic_properties_conformers.append({'muls': muls,
                                                         'alphas': alphas,
                                                         'wils': wils,
                                                         'dip': dip,
                                                         'alpha': alpha,
                                                         'dummy_positions': dummy_positions,
                                                         'fukui': fukui,
                                                         'HOMO_LUMO_gap': HOMO_LUMO_gap,
                                                         'IP_delta_SCC': IP_delta_SCC,
                                                         'EA_delta_SCC': EA_delta_SCC,
                                                         'global_electrophilicity_index': global_electrophilicity_index,
                                                         'esp_profile': esp_profile,
                                                         'esp_points': esp_points,
                                                         'occ_energies': occ_energies,
                                                         'virt_energies': virt_energies,
                                                         'nucleophilicity': nucleophilicity
                                                         })

                coords_all_used.append(coords_all[conf_idx])
                elements_all_used.append(elements_all[conf_idx])
                boltzmann_data_used.append(boltzmann_data[conf_idx])
                conf_indeces_used.append(conf_idx)

                if not xtb_done_here or not lmocent_file.exists():
                    xtb_done = False
    else:
        logger.error('CREST appears to have not completed.')
        electronic_properties_conformers = []

    time3 = time.time()

    time_xtb_sterimol = time3-crest_end_time

    return (crest_done, xtb_done, coords_all_used, elements_all_used, boltzmann_data_used, conf_indeces_used, electronic_properties_conformers, [total_crest_time, time_xtb_sterimol])

def get_crest_results(file: Path):
    '''
    Extracts CREST results from CRESt output log
    and associated files
    '''
    done=False

    CREST_LOG = file.parent / 'crest.log'
    CREST_CONFORMERS = file.parent / 'crest_conformers.xyz'
    OPTIM = file.parent / 'OPTIM'
    METADYNAMICS_1 = file.parent / 'METADYN1'
    NORMMD_1 = file.parent / 'NORMMD1'

    if not CREST_LOG.exists():
        raise FileNotFoundError(f'{CREST_LOG.absolute()} does not exist.')

    with open(CREST_LOG, 'r') as infile:
        text = infile.read()

    if not 'CREST terminated normally.' in text:
        raise ValueError(f'CREST did not complete successfully at {CREST_LOG.absolute()}.')

    # Do some additionally testing if CREST completed
    for _crest_test_item in [METADYNAMICS_1, OPTIM, NORMMD_1]:
        logger.warning('%s exists and indicates an incomplete CREST job', _crest_test_item.absolute())

    # Read in all conformers
    coords_all, elements_all = readXYZs(CREST_CONFORMERS)

    # Get the Boltzmann data
    data = read_crest_log(CREST_LOG)

    return True, coords_all, elements_all, data

def read_crest_log(crest_log_file: Path):
    '''
    What is going on in this function?
    '''
    read=False
    data=[]

    CREST_212_SUMMARY_PATTERN = re.compile(r'(?<=Erel/kcal        Etot weight/tot  conformer     set   degen     origin)(.*?)(?=T /K)', re.DOTALL)

    with open(crest_log_file, 'r') as infile:
        text = infile.read()

    with open(crest_log_file, 'r') as infile:
        lines = infile.readlines()

    # Use this for parsing CREST version 2.12
    if 'Version 2.12' in text:
        # Get the table of conformer summary information
        matches = re.findall(CREST_212_SUMMARY_PATTERN, text)
        # Make sure there is only one table (sanity)
        assert len(matches) == 1
        table = re.split('\n', matches[0].strip())
        table = [re.sub('\s+', ' ', x).strip().split(' ') for x in table]

        # Make sure all the values are appropriate length (sanity)
        assert all([(len(x) == 8 or len(x) == 5) for x in table])

        # Get all the of the conformers that have listed degeneracy (i.e., get confs not rotamers)
        table = [x for x in table if len(x) == 8]

        # Add everything to the data list
        for i in table:
            # This energy is the RELATIVE energy in kcal/mol
            energy = float(i[1])

            # This is the total conformer weight
            weight = float(i[4])

            # This is the second integer  that indicates how many rotamers
            degen = int(i[6])

            # This is the last item that shows what subroutine generated the structure
            origin = str(i[7])
            data.append({'energy': energy, 'weight': weight, 'degen': degen, 'origin': origin})

    else:
        for line in lines:
            if "T /K" in line:
                read=False

            if read:
                if len(line.split())>=7:
                    energy=float(line.split()[1])
                    weight=float(line.split()[4])
                    degen=int(line.split()[6])
                    if len(line.split())==8:
                        origin=line.split()[7]
                    else:
                        origin=None
                    data.append({"energy":energy,"weight":weight,"degen":degen,"origin":origin})

            if "Erel/kcal     Etot      weight/tot conformer  set degen    origin" in line:
                read=True

        #for i in data:
        #    print(i)

    return data