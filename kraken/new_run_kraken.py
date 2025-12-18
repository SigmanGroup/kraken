#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Code for running Kraken conformer searches
'''

# stdlib
import time
import shutil
import logging

from pathlib import Path

import yaml
import numpy as np

# Custom
from kraken.geometry import get_Ni_CO_3, replace
from kraken.semiempirical import run_crest, _get_crest_version
from kraken.xtb import _get_xtb_version, xtb_opt
from kraken.Kraken_Conformer_Selection_Only import conformer_selection_main
from kraken.file_io import write_xyz
from kraken.structure_generation import get_coords_from_smiles
from kraken.structure_generation import get_nickel_co3_complex_with_replace_method
from kraken.structure_generation import perform_pdcl5_complexation_to_get_metal_complexation_geometry
from kraken.structure_generation import generate_nickel_carbonyl_complex
from kraken.utils import _str_is_smiles
from kraken.utils import get_P_bond_indeces_of_ligand
from kraken.utils import get_rotatable_bonds, reduce_data, combine_yaml
from kraken.utils import confirm_defined_stereochemistry
from kraken.morfeus_properties import run_morfeus

from morfeus import read_xyz

logger = logging.getLogger(__name__)

def convert_conversion_flag(flag: int) -> tuple[bool, str]:
    '''
    Converts the integer conversion flag which is either
    0, 1, 2, 3, 4 and translates it to a string that
    is understood by other functions.

    #TODO This functionality should be changed.
    '''

    if flag == 0:                #0	RDkit
        generate_xyz = True
        conversion_method = "rdkit"
    elif flag == 1:              #1	Chemaxon
        conversion_method = "molconvert"
        raise Exception(f'Molconvert is not supported as of 27 March 2024.')
    elif flag == 2:              #2	manual (.xyz must be provided)
        conversion_method = 'manual'
        generate_xyz = False
    elif flag == 3:              #3	obabel
        generate_xyz = True
        conversion_method = "obabel"
    elif flag == 4:              #4	obabel / everything
        generate_xyz = True
        conversion_method = "any"
    else:
        raise ValueError(f'Could not find valid conversion method for flag {flag}')

    return generate_xyz, conversion_method

def normalize_structure_input(structure_input) -> tuple:
    '''
    Normalize structure input to either a validated SMILES or an existing file path.

    Parameters
    ----------
    structure_input: str | Path
        Either a SMILES string, or a Path to an existing structure file.

    Returns
    -------
    result: tuple
        (smiles, xyz_file_path, generate_xyz) where smiles is a validated SMILES or None,
        xyz_file_path is a Path or None, and generate_xyz is True iff an coordinates should
        be generated.
    '''
    if isinstance(structure_input, Path):
        xyz_file_path = structure_input
        if not xyz_file_path.exists():
            raise FileNotFoundError(f'{xyz_file_path} does not exist.')
        return (None, xyz_file_path, False)

    if isinstance(structure_input, str):
        candidate_path = Path(structure_input)

        # Prefer "path" interpretation if it exists on disk
        if candidate_path.exists():
            return (None, candidate_path, False)

        # Otherwise interpret as SMILES
        smiles = confirm_defined_stereochemistry(structure_input)
        return (smiles, None, True)

    raise TypeError(f'structure_input must be a str or Path, got {type(structure_input)}')

def run_kraken_calculation(kraken_id: str,
                           structure_input: str | Path,
                           charge: int,
                           mol_dir: Path,
                           reduce_crest_output: bool,
                           dummy_distance: float,
                           settings: dict,
                           metal_char: str = 'Ni',
                           jobs: list[str] = ['noNi', 'Ni'],
                           nprocs: int = 4,
                           conversion_flag: int = 4,
                           add_Pd_Cl2: bool = False,
                           add_Pd_Cl2_PH3: bool = False,
                           add_Ni_CO_3: bool = False) -> None:
    '''
    This is the primary function that executes the Kraken conformer
    search and conformer selection code.

    Parameters
    ----------
    kraken_id: str
        The Kraken ID. Must be entirely numeric and 8 digits
        formatted as a string (NOT an integer)

    structure_input: str | Path
        The input that represents the structure to be computed.
        Can be either a SMILES string or a pathlib.Path object
        that points to a .xyz file contain the coordinates of
        the monophosphine.

    charge: int
        Charge of the molecule.

    mol_dir: Path
        Directory for this particular molecule/ligand that will
        contain the CREST and xTB calculations as well as the
        DFT directory for the second part of Kraken.

    reduce_crest_output: bool
        Removes files from the CREST and xTB calculations to save
        space.

    dummy_distance: float
        Dummy distance for MORFEUS (default should be 2.1 Å)

    settings: dict
        A dictionary of settings that are useful to this function.

            'max_E': 6.0, (UNUSED)
            'max_p': 0.1, (UNUSED)
            'OMP_NUM_THREADS': nprocs, (UNUSED - set before calling function)
            'MKL_NUM_THREADS': nprocs, (UNUSED - set before calling function)
            'dummy_distance': 2.1, # Additional record of dummy distance
            'remove_scratch': True, (UNUSED)
            'reduce_output': reduce_crest_output, # Additional record of reduce_output
            'add_Pd_Cl2': False, (UNUSED)
            'add_Pd_Cl2_PH3': False, (UNUSED)
            'add_Ni_CO_3': False (UNUSED)

    metal_char: str (default = 'Ni')
        Metal character to add to the structure when computing the
        Ni-bound conformations.

    jobs: list[str]
        List of jobs to run. Should include 'Ni' and 'noNi' by default

    nprocs:
        Number of processors to use in the CREST/xTB calculations

    conversion_flag: int (options = 1, 2, 3, 4)
        Method to use for converting SMILES to 3D coordinates

    add_Pd_Cl2: (default = False)
        Deprecated

    add_Pd_Cl2_PH3: (default = False)
        Deprecated

    add_Ni_CO_3: (default = False)
        Deprecated

    Returns
    -------
    None
    '''
    #TODO add options for noreftopo and nocross

    # Do some existence checking
    if not mol_dir.exists():
        raise FileNotFoundError(f'{mol_dir.absolute()} does not exist')

    if mol_dir.name != kraken_id:
        raise FileNotFoundError(f'directory name ->{mol_dir.name}<- does not match the Kraken ID ->{kraken_id}<-')

    # Check if the number/type of jobs requested
    # agrees with how Kraken was originally designed
    if len(jobs) not in [1, 2]:
        raise ValueError(f'Number of jobs was {len(jobs)} when expecting 2')

    if any([z not in ['noNi', 'Ni'] for z in jobs]):
        raise ValueError(f'Only "noNi" and "Ni" are acceptable job types for Kraken.')

    conversion_flag = int(conversion_flag)

    generate_xyz, conversion_method = convert_conversion_flag(int(conversion_flag))

    logger.info('Beginning conformer search procedure on %s', kraken_id)
    logger.info('The 3D conversion method is %d (%s)', conversion_flag, conversion_method)

    logger.debug('kraken_id: %s, %s', kraken_id, type(kraken_id))
    logger.debug('reduce_crest_output: %s, %s', reduce_crest_output, type(reduce_crest_output))
    logger.debug('nprocs: %s, %s', nprocs, type(nprocs))

    # Get "time1"
    start_time = time.time()

    # Make the parent directory that will contain the CREST calculations for
    # both the nickel complex and the free ligand
    crest_parent_dir = mol_dir / 'crest_calculations'
    crest_parent_dir.mkdir(exist_ok=True)

    logger.debug('generate_xyz:\t%s', generate_xyz)
    logger.debug('conversion_method:\t%s', conversion_method)
    logger.debug('Additional settings defined by "settings" dictionary:')

    logger.debug('add_Pd_Cl2: %s', add_Pd_Cl2)
    logger.debug('mol_dir: %s', mol_dir.absolute())

    # Check whether we're working with a smiles or path
    smiles, xyz_file_path, generate_xyz = normalize_structure_input(structure_input=structure_input)

    # Validate defined stereochemistry
    if smiles is not None:
        smiles = confirm_defined_stereochemistry(smiles=smiles)

    # Begin primary job loop here.
    for job in jobs:

        logger.info('Starting job %s for %s', job, kraken_id)
        logger.debug('smiles: %s', smiles)
        logger.debug('generate_xyz: %s', generate_xyz)

        # Make the directory for running this particular CREST calculations
        crest_calculation_dir = crest_parent_dir / f'{kraken_id}_{job}'
        crest_calculation_dir.mkdir(exist_ok=True)

        # Structure generation
        if generate_xyz and isinstance(smiles, str):

            # If we are requested to add Ni and Ni is not already in the smiles
            if job == 'Ni' and (metal_char not in smiles):

                logger.info('Job requested was %s but %s was not in %s. Adding %s.', job, metal_char, smiles, metal_char)

                # Make a directory for creating the initial structure
                # This is the new method that seems to have a higher success rate
                # The old method is get_nickel_co3_complex_with_replace_method
                structure_generation_directory = mol_dir / 'structure_generation'
                structure_generation_directory.mkdir(exist_ok=True)

                elements, coords = generate_nickel_carbonyl_complex(kraken_id=kraken_id,
                                                                    smiles=smiles,
                                                                    charge=charge,
                                                                    structure_gen_dir=structure_generation_directory)

                # Set the phosphorus index
                P_index = list(elements).index('P')
                settings['P_index'] = P_index

            else:
                logger.info('Generating coordinates from SMILES %s', smiles)

                coords, elements = get_coords_from_smiles(smiles=smiles, conversion_method=conversion_method)

            # Make the file for running crest
            xyz_file_path = crest_calculation_dir / f'{kraken_id}_{job}.xyz'

            write_xyz(destination=xyz_file_path, coords=coords, elements=elements, mask=[])

            # Keep a copy in the parent directory for comparison
            shutil.copy2(xyz_file_path, mol_dir / f'{kraken_id}_{job}_crest_input_structure_copy.xyz')

        # Else if handed an xyz file
        else:
            if xyz_file_path is None:
                raise FileNotFoundError(f'Something went wrong and xyz_file_path was set to None.')

            elements, coords = read_xyz(xyz_file_path)

            if 'Ni' in elements:
                raise ValueError(f'Using a .xyz file directly require no Ni to be present.')

            shutil.copy2(xyz_file_path, crest_calculation_dir / xyz_file_path.name)
            xyz_file_path = crest_calculation_dir / xyz_file_path.name

        # Run the CREST calculation
        logger.info('Running CREST calculation of Kraken ID %s at %s', kraken_id, xyz_file_path)

        crest_done, xtb_done, coords_all, elements_all, boltzmann_data_conformers, conf_indeces, electronic_properties_conformers, time_needed = run_crest(file=xyz_file_path,
                                                                                                                                                           nprocs=nprocs,
                                                                                                                                                           reduce_output=reduce_crest_output,
                                                                                                                                                           smiles=smiles,
                                                                                                                                                           charge=charge,
                                                                                                                                                           metal_char=metal_char,
                                                                                                                                                           add_Pd_Cl2=add_Pd_Cl2,
                                                                                                                                                           add_Pd_Cl2_PH3=add_Pd_Cl2_PH3,
                                                                                                                                                           add_Ni_CO_3=add_Ni_CO_3)

        if not crest_done:
            raise ValueError(f'CREST did not terminate normally for {xyz_file_path.name} on job {job}.')

        if not xtb_done:
            raise ValueError(f'xTB calculations did not complete for {xyz_file_path.name} on job {job}.')

        logger.info('Found %d conformers of %s', len(elements_all), xyz_file_path.name)

        # Holds data for the MORFEUS calculations
        morfeus_parameters_conformers = []

        # Enumerate through all coordinates
        for conf_idx, conformer_coordinates in enumerate(coords_all):

            # Get the dummy positions
            dummy_positions = electronic_properties_conformers[conf_idx]["dummy_positions"]

            # Get the elements for this conformer
            elements_conf = elements_all[conf_idx]

            # Get the index of the phosphorus atom for the MORFEUS calculation
            morfeus_phosphorus_index = list(elements).index('P')

            # Get the directory of this conformer
            #moldir_conf = "%s/conf_%i"%(moldir,conf_indeces[conf_idx])
            conf_dir = xyz_file_path.parent / f'conf_{conf_indeces[conf_idx]}'

            logger.info('Running MORFEUS on %s(conformer %d of %d)', xyz_file_path.name, conf_idx + 1, len(coords_all))

            morfeus_parameters = run_morfeus(coords=conformer_coordinates,
                                             elements=elements_conf,
                                             dummy_positions=dummy_positions,
                                             dummy_distance=dummy_distance,
                                             P_index=morfeus_phosphorus_index,
                                             metal_char=metal_char,
                                             conf_dir=conf_dir,
                                             suffix=job,
                                             smiles=smiles)

            morfeus_parameters_conformers.append(morfeus_parameters)

        # Define output files
        output_summary_file = mol_dir / f'{kraken_id}_{job}.yml'
        output_conformer_file = mol_dir / f'{kraken_id}_{job}_confs.yml'
        output_combined_file = mol_dir / f'{kraken_id}_{job}_combined.yml'
        logger.info('Saving the summary results of molecule %s to %s', kraken_id, output_summary_file)

        # If CREST and all xTB calculations completed for the "job"
        if crest_done and xtb_done:

            # Create a dictionary that contains the results
            results = {}

            results['coords_start'] = coords.tolist()
            results['elements_start'] = elements
            results['smiles'] = smiles

            # Calculate number of rotatable bonds
            if (smiles == "not available") or (smiles is None):
                rotatable_bonds = []
            else:
                try:
                    rotatable_bonds = get_rotatable_bonds(smiles)
                except Exception as _e:
                    logger.error('Rotatable bonds calculation failed because %s', _e)
                    rotatable_bonds = []

            results['rotatable_bonds'] = rotatable_bonds
            num_rotatable_bonds = len(rotatable_bonds)
            results['num_rotatable_bonds'] = num_rotatable_bonds

            # Add conformer data. Until this point,
            # all of the data is still available
            # Iterate through all the list of elements
            for conf_idx, elements_conf in enumerate(elements_all):

                # Add typical conformer data
                results[f'conf_{conf_idx}'] = {'coords': coords_all[conf_idx],
                                               'elements': elements_conf,
                                               'boltzmann_data': boltzmann_data_conformers[conf_idx],
                                               'electronic_properties': electronic_properties_conformers[conf_idx],
                                               'sterimol_parameters': morfeus_parameters_conformers[conf_idx]
                                               #'sterimol_parameters': sterimol_parameters_conf
                }

            # Sort the data in different output files and kill unnecessary data
            data_here, data_here_confs, data_here_esp_points = reduce_data(results)

            # Add CREST and xTB version
            data_here['crest_version'] = _get_crest_version()
            data_here['xtb_version'] = _get_xtb_version()

            # add the timings
            # TODO When computing the timings, they should be
            # TODO saved to a file and be reread in. Otherwise, the
            # TODO timings will be less about calculation time and
            # TODO more about how long it takes to read the files
            time2 = time.time()
            time_all = time2 - start_time

            results['settings'] = settings
            results['time_crest'] = time_needed[0]
            results['time_morfeus'] = time_needed[1]
            results['time_all'] = time_all

            # save the main output file (this will hopefully be the smallest file with the most important data
            with open(output_summary_file, 'w', encoding='utf-8') as outfile:
                outfile.write(yaml.dump(data_here, default_flow_style=False))
            logger.info('Saved summary file to %s', output_summary_file.absolute())

            # Conformer data goes to an extra output file
            with open(output_conformer_file, 'w', encoding='utf-8') as outfile:
                outfile.write(yaml.dump(data_here_confs, default_flow_style=False))
            logger.info('Saved conformer file to %s', output_conformer_file.absolute())

            # Combine things
            combined = combine_yaml(kraken_id,
                                    data_here,
                                    data_here_confs)
            with open(output_combined_file, 'w', encoding='utf-8') as outfile:
                outfile.write(yaml.dump(combined, default_flow_style=False))
            logger.info('Saved combined file to %s', output_combined_file.absolute())

        # Else if crest_done and xtb_done are false
        else:
            logger.error('Kraken ID %s FAILED job %s', kraken_id, job)
            failed_xtb_yaml_file = mol_dir / f'{kraken_id}_{job}.yml'
            with open(failed_xtb_yaml_file, 'w', encoding='utf-8') as outfile:
                outfile.write('FAILED\n')

        logger.info('Finished job %s for Kraken ID %s', job, kraken_id)

    # Define the two combined data files
    noNi_datafile = mol_dir / f'{kraken_id}_noNi_combined.yml'
    Ni_datafile = mol_dir / f'{kraken_id}_Ni_combined.yml'

    # Make the directory to save the DFT com files to
    dft_dir = mol_dir / 'dft'
    dft_dir.mkdir(exist_ok=True)

    logger.info('Selecting conformers for Kraken ID %s. Output .com files to %s', kraken_id, str(dft_dir.absolute()))

    conformer_selection_main(kraken_id,
                             save_dir=dft_dir,
                             noNi_datafile=noNi_datafile,
                             Ni_datafile=Ni_datafile,
                             nprocs=nprocs,
                             charge=charge)

    logger.info('Completed all semiempirical calculations and conformer selection for Kraken ID %s', kraken_id)
