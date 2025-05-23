#!/usr/bin/env python3
# coding: utf-8

'''
Code for running Kraken conformer searches

#TODO INCLUDE ORIGINAL AUTHORS OF CODE

Major revisions March 2024:
    James Howard, PhD
'''

# stdlib
import os
import sys
import time
import shutil
import logging

from pathlib import Path

# Dependencies
import yaml
import numpy as np

#conf_prune_idx = Path(__file__).parent / 'ConfPruneIdx.pyx'
#shutil.copy2(conf_prune_idx, Path.cwd() / conf_prune_idx.name)

# Custom
from kraken.geometry import get_Ni_CO_3, replace
from kraken.semiempirical import run_crest, _get_crest_version
from kraken.xtb import _get_xtb_version, xtb_opt
from kraken.Kraken_Conformer_Selection_Only import conformer_selection_main
from kraken.file_io import write_xyz
from kraken.structure_generation import get_coords_from_smiles
from kraken.utils import _str_is_smiles, get_num_bonds_P, add_Hs_to_P
from kraken.utils import add_to_smiles, remove_complex, get_P_bond_indeces_of_ligand
from kraken.utils import get_rotatable_bonds, reduce_data, combine_yaml
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

def run_kraken_calculation(kraken_id: str,
                           structure_input: str | Path,
                           mol_dir: Path,
                           dft_dir: Path,
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

    mol_dir: Path
        Directory for this particular molecule/ligand that will contain
        the CREST and xTB calculations as well as the dft/ directory for
        the second part of Kraken.
    '''
    #TODO add options for noreftopo and nocross

    # Do some existence checking
    if not mol_dir.exists():
        raise FileNotFoundError(f'{mol_dir.absolute()} does not exist')

    if not dft_dir.exists():
        raise FileNotFoundError(f'{dft_dir.absolute()} does not exist')

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
    if _str_is_smiles(structure_input):
        smiles = structure_input
        generate_xyz = True
        xyz_file_path = None
    else:
        smiles = None
        xyz_file_path = Path(structure_input)
        if not xyz_file_path.exists():
            raise FileNotFoundError(f'{xyz_file_path} does not exist.')
        generate_xyz = False

    # Begin primary job loop here.
    for job in jobs:

        logger.info('Starting job %s for %s', job, kraken_id)
        logger.debug('xyz_file_path: %s', str(xyz_file_path))
        logger.debug('smiles: %s', smiles)
        logger.debug('generate_xyz: %s', generate_xyz)

        # Make the directory for running this particular CREST calculations
        crest_calculation_dir = crest_parent_dir / f'{kraken_id}_{job}'
        crest_calculation_dir.mkdir(exist_ok=True)

        # Structure generation
        if generate_xyz and smiles is not None:

            # If we are requested to add Ni and Ni is not already in the smiles
            if job == 'Ni' and metal_char not in smiles:

                logger.info('Job requested was %s but %s was not in %s. Adding %s.', job, metal_char, smiles, metal_char)

                num_bonds_P = get_num_bonds_P(smiles)

                logger.debug('num_bonds_P of smiles %s: %d', smiles, num_bonds_P)

                # Add the Hs to smiles phosphorus atom
                smiles_Hs = add_Hs_to_P(smiles, num_bonds_P)

                logger.debug('New formatted smiles is %s', smiles)

                #print(smiles_Hs)
                #smiles = utils.add_to_smiles(smiles,"[Pd@SP1]([Cl])([PH3])[Cl]")
                #smiles = utils.add_to_smiles(smiles,"[Pd]([Cl])([PH3])[Cl]")
                #spacer_smiles="[Pd]([As+](F)(F)F)([As+](F)(F)F)[As+](F)(F)F"
                spacer_smiles="[Pd]([Cl])([Cl])([Cl])([Cl])[Cl]"

                smiles_incl_spacer = add_to_smiles(smiles_Hs, spacer_smiles)

                logger.debug('smiles_incl_space: %s', smiles_incl_spacer)

                coords_ligand_complex, elements_ligand_complex = get_coords_from_smiles(smiles=smiles_incl_spacer,
                                                                                        conversion_method=conversion_method)

                # Get the number of atoms in the fake Pd(Cl)5 complex
                num_atoms_with_fake_complex = len(coords_ligand_complex)

                logger.debug('Number of atoms after adding fake complex: %d', num_atoms_with_fake_complex)

                # Remove the complex and get the coordinates of just the ligand
                coords_ligand, elements_ligand, done = remove_complex(coords=coords_ligand_complex,
                                                                      elements=elements_ligand_complex,
                                                                      smiles=smiles)

                if not done:
                    _dest = Path(mol_dir / f'{kraken_id}_failed_complex_in_generate_xyz.xyz')

                    if (coords_ligand_complex is not None) and (elements_ligand_complex is not None):
                        write_xyz(destination=_dest,
                                    coords=coords_ligand_complex,
                                    elements=elements_ligand_complex)

                    raise ValueError(f'utils.remove_complex did not complete correctly. Saved file to {_dest.absolute()}')

                # Get the number of atoms without the fake complex
                num_atoms_without_fake_complex = len(coords_ligand)

                logger.debug('Number of atoms of %s after removing the fake complex: %d', kraken_id, num_atoms_without_fake_complex)

                # Compute the difference
                difference = num_atoms_with_fake_complex - num_atoms_without_fake_complex

                # Sanity check
                if difference != 6:
                    write_xyz(destination=Path(mol_dir / f'{kraken_id}_failed_complex_in_generate_xyz_atom_no_difference.xyz'),
                              coords=coords_ligand_complex,
                              elements=elements_ligand_complex,
                              comment='Failed complex generation for Pd(Cl)5 complex',
                              mask=[])

                    raise ValueError(f'number of removed atoms is {difference}, but should be 6 for Pd(Cl)5. Saved file to {Path(moldir / f"{kraken_id}_failed_complex_in_generate_xyz_atom_no_difference.xyz").absolute()}')

                P_index, bond_indeces = get_P_bond_indeces_of_ligand(coords_ligand, elements_ligand)
                if len(bond_indeces) != 3:
                    logger.warning('Number of P-bonds before adding complex was %d instead of 3 for SMILES %s', len(bond_indeces), smiles)

                direction = np.zeros((3))

                for bond_index in bond_indeces:
                    direction += (coords_ligand[bond_index] - coords_ligand[P_index])

                direction /= (-np.linalg.norm(direction))
                coords_ligand=np.array(coords_ligand.tolist() + [(coords_ligand[P_index] + 2.25 * direction).tolist()])
                elements_ligand.append(metal_char)
                match_pd_ind=len(elements_ligand) - 1
                match_p_idx = P_index

                # replace(c1_i, e1_i, c2_i, e2_i,  Au_index, P_index, match_Au_index, match_P_index, rotate_third_axis=True)
                #coords_pd, elements_pd, pd_idx, p_idx = geometry.get_Pd_NH3_Cl_Cl()
                #metal_char="Pd"

                coords_pd, elements_pd, pd_idx, p_idx = get_Ni_CO_3()
                success, coords, elements = replace(coords_pd, elements_pd, coords_ligand, elements_ligand, pd_idx, p_idx, match_pd_ind, match_p_idx, smiles, rotate_third_axis=True)
                if elements==None:
                    exit(f'[FATAL] Elements is None for {smiles}. Exiting gracefully.')
                if len(elements)==0:
                    exit(f'[FATAL] Elements is empty for {smiles}. Exiting gracefully.')

                #print(coords[0])
                xtb_scr_dir = mol_dir / 'xtb_scr_dir'
                xtb_scr_dir.mkdir(exist_ok=True)

                logger.info('Optimizing preliminary complex.')

                coords, elements = xtb_opt(coords=coords, elements=elements, smiles=smiles, scratch_dir=xtb_scr_dir, charge=0, nprocs=nprocs, freeze=[])

                P_index = list(elements).index('P')

                #As_index=elements.index("As")
                #elements[As_index]="N"

                settings['P_index'] = P_index
                if not success:
                    exit("[FATAL] Pd addition did not work. Exiting gracefully.")

            else:
                logger.info('No metal coordination requested. Generating coordinates from SMILES %s', smiles)

                coords, elements = get_coords_from_smiles(smiles=smiles,
                                                          conversion_method=conversion_method)

            # Make the file for running crest
            xyz_file_path = crest_calculation_dir / f'{kraken_id}_{job}.xyz'

            write_xyz(destination=xyz_file_path, coords=coords, elements=elements, mask=[])

            # Keep a copy in the parent directory for comparison
            shutil.copy2(xyz_file_path, crest_calculation_dir / f'{kraken_id}_{job}_crest_input_structure_copy.xyz')

        # Else if handed an xyz file
        else:
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
                                                                                                                                                           metal_char=metal_char,
                                                                                                                                                           add_Pd_Cl2=add_Pd_Cl2,
                                                                                                                                                           add_Pd_Cl2_PH3=add_Pd_Cl2_PH3,
                                                                                                                                                           add_Ni_CO_3=add_Ni_CO_3)

        # Electronic properties wils, alphas, fukui are making it here in testing
        # The missing keys must be a result of something else

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

    logger.info('Selecting conformers for Kraken ID %s', kraken_id)

    # Define the two combined data files
    noNi_datafile = mol_dir / f'{kraken_id}_noNi_combined.yml'
    Ni_datafile = mol_dir / f'{kraken_id}_Ni_combined.yml'

    conformer_selection_main(kraken_id,
                             save_dir=dft_dir,
                             noNi_datafile=noNi_datafile,
                             Ni_datafile=Ni_datafile,
                             nprocs=nprocs)

    logger.info('Completed all semiempirical calculations and conformer selection for Kraken ID %s', kraken_id)
