#!/usr/bin/env python3
# coding: utf-8

'''
Code for running properties with MORFEUS for the
conformer search part of Kraken
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import annotations

import time
import yaml
import logging

from pathlib import Path

import numpy as np

from numpy.typing import NDArray

import scipy.spatial as scsp

from morfeus import BuriedVolume
from morfeus import Pyramidalization
from morfeus import ConeAngle
from morfeus import Sterimol
from morfeus import SASA
from morfeus import Dispersion

from .file_io import write_xyz
from .utils import get_ligand_indices

logger = logging.getLogger(__name__)

def run_morfeus(coords: NDArray,
                elements: NDArray,
                dummy_positions: list[list[float]],
                dummy_distance: float,
                P_index: int,
                metal_char: str,
                conf_dir: Path,
                suffix: str,
                smiles: str) -> dict:
    '''
    Runs MORFEUS
    '''

    if suffix not in ['noNi', 'Ni']:
        raise ValueError(f'Only valid suffixes for run_morfeus are "Ni" and "noNi"')

    outfile = conf_dir / 'morfeus.yml'

    # Return the existing parameters if they exist
    if outfile.exists():
        logger.info('Found MORFEUS parameters at %s. Returning parameters.', outfile.absolute())
        with open(outfile, 'r', encoding='utf-8') as infile:
            return yaml.load(infile, Loader=yaml.FullLoader)

    times = {}
    time0 = time.time()
    do_pyramid = True

    if suffix == 'Ni':

        mask, done = get_ligand_indices(coords=np.array(coords),
                                        elements=elements,
                                        P_index=P_index,
                                        smiles=smiles,
                                        metal_char=metal_char)

        if (not done) or (mask is None):
            logger.critical('get_ligand_indices returned (mask, done) of (%s, %s)', mask, done)
            exit()

        # This raises a ValueError if the metal_char is not in elements
        pd_idx_full_ligand = list(elements).index(metal_char)

        # Extend the molecule
        coords_list = []
        elements_list = []
        coords_extended = []
        elements_extended = []

        # Iterate through the indices of the mask
        for atomidx in mask:
            atom = coords[atomidx]
            elements_extended.append(elements[atomidx])
            coords_extended.append([atom[0],atom[1],atom[2]])
            coords_list.append([atom[0],atom[1],atom[2]])
            elements_list.append(elements[atomidx])

        coords_extended.append(coords[pd_idx_full_ligand])

        elements_extended+=[metal_char]

        # Get another p_idx but this is not capitalized?
        p_idx = list(elements).index('P')

        # Get the pd_idx
        pd_idx = list(elements).index(metal_char)

        # Get the dummy_idx
        dummy_idx = len(elements_extended) - 1

        # Save the sterimol input to a file
        write_xyz(destination=conf_dir / 'sterimol_input.xyz',
                  coords=np.array(coords_extended),
                  elements=np.array(elements_extended),
                  comment=f'File used for MORFEUS sterimol (metal_containing). p_idx: {p_idx} pd_idx: {pd_idx}',
                  mask=[])

        atom_distances_to_p = scsp.distance.cdist([coords_extended[p_idx]],coords_extended)[0]

        neighbor_indeces = [i for i in np.argsort(atom_distances_to_p)[1:4] if i != pd_idx]

        if len(neighbor_indeces) != 3:
            logger.warning('Found %d neigbors instead of 3 on %s for %s', len(neighbor_indeces), metal_char, conf_dir.name)

            # Turn off pyramidalization
            do_pyramid = False

        selected_dummy_idx = -1

    # If we're doing a free ligand calculation
    else:

        # Get the p_idx
        p_idx = list(elements).index('P')

        # Get the correct dummy atom
        dummy_distances_to_p = scsp.distance.cdist([coords[p_idx]], dummy_positions)[0]

        #logger.debug('dummy_distances_to_p: %s', str(dummy_distances_to_p))

        nearest_dummy_indeces=np.argsort(dummy_distances_to_p)[:4]

        atom_distances_to_p=scsp.distance.cdist([coords[p_idx]],coords)[0]

        neighbor_indeces=np.argsort(atom_distances_to_p)[1:4]

        neighbor_dummy_distances=scsp.distance.cdist(np.array(dummy_positions)[nearest_dummy_indeces],np.array(coords)[neighbor_indeces])

        minimal_distances=np.min(neighbor_dummy_distances,axis=1)

        dummy_atom_with_largest_minimal_distance=np.argmax(minimal_distances)

        selected_dummy_idx=nearest_dummy_indeces[dummy_atom_with_largest_minimal_distance]

        # Get the direction from P to dummy
        dummy_direction=np.array(dummy_positions[selected_dummy_idx])-np.array(coords[p_idx])
        dummy_direction_norm=np.linalg.norm(dummy_direction)
        dummy_direction/=dummy_direction_norm

        # Go from p into dummy direction
        dummy_position=np.array(coords[p_idx]) + dummy_distance * dummy_direction

        # Extend the molecule
        coords_list=[]
        elements_list=[]
        coords_extended=[]

        for atomidx, atom in enumerate(coords):
            coords_extended.append([atom[0], atom[1], atom[2]])
            coords_list.append([atom[0], atom[1], atom[2]])
            elements_list.append(elements[atomidx])

        coords_extended.append(dummy_position.tolist())

        elements_extended=elements + ["H"]

        dummy_idx=len(elements_extended) - 1

        # Save the sterimol input to a file
        write_xyz(destination=conf_dir / 'sterimol_input.xyz',
                  coords=np.array(coords_extended),
                  elements=np.array(elements_extended),
                  comment=f'indeces: {dummy_idx} and {p_idx}',
                  mask=[])

    # Save the preparation time
    time1 = time.time()
    times["preparation"] = time1 - time0

    # start morfeus stuff
    if len(elements_extended) != len(coords_extended):
        logger.warning('MORFEUS coords (%d) and elements (%d) are different lengths', len(coords_extended), len(elements_extended))
    try:
        cone_angle = ConeAngle(elements_extended,
                               coords_extended,
                               atom_1=dummy_idx + 1,
                               method='internal')
        cone_angle_val = float(cone_angle.cone_angle)
    except Exception as _e:
        logger.error('MORFEUS cone angle failed because %s. Setting value to None.', _e)
        cone_angle_val = None

    # Save the ConeAngle time
    time2 = time.time()
    times["ConeAngle"] = time2 - time1

    if len(elements_list)!=len(coords_list):
        logger.warning('SASA calculation elements (%d) and coords (%d) are different lengths', len(elements_list), len(coords_list))

    try:
        sasa = SASA(elements_list, coords_list)
        sasa_val = float(sasa.area)
        sasa_val_P = float(sasa.atom_areas[p_idx+1])
        sasa_volume = float(sasa.volume)
        sasa_volume_P = float(sasa.atom_volumes[p_idx+1])
    except Exception as _e:
        logger.error('MORFEUS SASA failed because %s. Setting sasa_val, sasa_val_P, sasa_volume, sasa_volume_P to None.', _e)
        sasa_val = None
        sasa_val_P = None
        sasa_volume = None
        sasa_volume_P = None

    # Record the SASA time
    time3 = time.time()
    times["SASA"]=time3-time2

    if len(elements_extended) != len(coords_extended):
        logger.warning('MORFEUS coords (%d) and elements (%d) are different lengths', len(coords_extended), len(elements_extended))
    try:
        sterimol = Sterimol(elements_extended, coords_extended, dummy_idx+1, p_idx+1)
        lval = float(sterimol.L_value)
        B1 = float(sterimol.B_1_value)
        B5 = float(sterimol.B_5_value)
    except Exception as _e:
        logger.error('MORFEUS SASA failed because %s. Setting lval, B1, B5 to None.', _e)
        lval = None
        B1 = None
        B5 = None

    time4 = time.time()
    times["Sterimol"] = time4 - time3

    if len(elements_list)!=len(coords_list):
        logger.warning('Dispersion calculation elements (%d) and coords (%d) are different lengths', len(elements_list), len(coords_list))
    try:
        disp = Dispersion(elements_list, np.array(coords_list))
        p_int = float(disp.p_int)
        p_int_atoms = disp.atom_p_int
        p_int_atom = float(p_int_atoms[p_idx+1])
        p_int_area = float(disp.area)
        p_int_atom_areas = disp.atom_areas
        p_int_atom_area = float(p_int_atom_areas[p_idx+1])
        p_int_times_p_int_area = float(p_int*p_int_area)
        p_int_atom_times_p_int_atom_area = float(p_int_atom*p_int_atom_area)
    except Exception as _e:
        logger.error('MORFEUS Dispersion failed because %s. Setting 6 Pint values to None.', _e)
        p_int = None
        p_int_atom = None
        p_int_area = None
        p_int_atom_area = None
        p_int_times_p_int_area = None
        p_int_atom_times_p_int_atom_area = None

    time5 = time.time()
    times["Dispersion"] = time5 - time4

    # Pyramidalization - two equivalent measurments P and alpha
    if do_pyramid:
        try:
            pyr = Pyramidalization(elements = elements_extended, coordinates = coords_extended, atom_index = p_idx+1, excluded_atoms = [dummy_idx+1]) # remove Pd
            pyr_val = float(pyr.P)
            pyr_alpha = float(pyr.alpha)
        except Exception as e:
            logger.error('MORFEUS Dispersion failed because %s. Setting pyr_val, pyr_alpha to None.', _e)
            pyr_val = None
            pyr_alpha = None
    else:
        logger.warning('Pyramidalization was skipped!')
        pyr_val = None
        pyr_alpha = None

    time6 = time.time()
    times["Pyramidalization"] = time6 - time5

    #Buried volume - get quadrant volumes and distal volume
    # iterate through P-substituents, aligning the quadrants paralell to each once (= xz_plane definition)
    # Metal/point of reference should be 2.28 A away from P
    # z_axis_atoms: P
    # xz_plane_atoms: each of the substituents once
    # keep lowest and highest quadrant and octant volume across all three orientations of the coordinate system
    # keep highest difference of any neighboring quadrant volume
    # keep volume in each of the two hemispheres

    try:
        qvbur_all = np.array([])
        qvdist_all = np.array([])
        qvtot_all = np.array([])
        max_delta_qvbur_all = []
        max_delta_qvtot_all = []
        ovbur_all = np.array([])
        ovtot_all = np.array([])

        for i in neighbor_indeces:
            bv = BuriedVolume(elements_extended, coords_extended, dummy_idx+1, excluded_atoms=[dummy_idx+1], z_axis_atoms=[p_idx+1], xz_plane_atoms=[i+1], density=0.01) # dummy_idx+1 = 2
            bv.octant_analysis()
            bv.compute_distal_volume(method="buried_volume", octants=True)

            vbur = bv.buried_volume   # these are identical for each iteration
            #vbur = bv.percent_buried_volume   # these are identical for each iteration
            vdist = bv.distal_volume  #
            vtot = vbur + vdist       #

            qvbur = np.asarray(list(bv.quadrants["buried_volume"].values()))
            qvdist = np.asarray(list(bv.quadrants["distal_volume"].values()))
            qvtot = qvbur + qvdist

            qvbur_all = np.append(qvbur_all,qvbur)
            qvtot_all = np.append(qvtot_all,qvtot)

            max_delta_qvbur_all.append(max([abs(qvbur[i]-qvbur[i-1]) for i in range(4)]))
            max_delta_qvtot_all.append(max([abs(qvtot[i]-qvtot[i-1]) for i in range(4)]))

            ovbur = np.asarray(list(bv.octants["buried_volume"].values()))
            ovdist = np.asarray(list(bv.octants["distal_volume"].values()))
            ovtot = ovbur + ovdist

            ovbur_all = np.append(ovbur_all,ovbur)
            ovtot_all = np.append(ovtot_all,ovtot)

            near_vbur = ovbur[4:].sum()   # these are identical for each iteration
            far_vbur = ovbur[:4].sum()    #
            near_vtot = ovtot[4:].sum()   #
            far_vtot = ovtot[:4].sum()    #

        qvbur_min = float(min(qvbur_all))
        qvbur_max = float(max(qvbur_all))
        qvtot_min = float(min(qvtot_all))
        qvtot_max = float(max(qvtot_all))

        max_delta_qvbur = float(max(max_delta_qvbur_all))
        max_delta_qvtot = float(max(max_delta_qvtot_all))

        ovbur_min = float(min(ovbur_all))
        ovbur_max = float(max(ovbur_all))
        ovtot_min = float(min(ovtot_all))
        ovtot_max = float(max(ovtot_all))

        # this is just a reminder to keep these properties
        vbur = float(vbur)
        vtot = float(vtot)
        near_vbur = float(near_vbur)
        far_vbur = float(far_vbur)
        near_vtot = float(near_vtot)
        far_vtot = float(far_vtot)


    except Exception as _e:
        logger.error('MORFEUS BuriedVolume failed because %s. Setting 16 volumes to None.', _e)
        qvbur_min = None
        qvbur_max = None
        qvtot_min = None
        qvtot_max = None

        max_delta_qvbur = None
        max_delta_qvtot = None

        ovbur_min = None
        ovbur_max = None
        ovtot_min = None
        ovtot_max = None

        vbur = None
        vtot = None
        near_vbur = None
        far_vbur = None
        near_vtot = None
        far_vtot = None

    time7 = time.time()
    times["BuriedVolume"] = time7 - time6

    results={'lval': lval,
             'B1': B1,
             'B5': B5,
             'sasa': sasa_val,
             'sasa_P': sasa_val_P,
             'sasa_volume': sasa_volume,
             'sasa_volume_P': sasa_volume_P,
             'cone_angle': cone_angle_val,
             'p_int': p_int,
             'p_int_atom': p_int_atom,
             'p_int_area': p_int_area,
             'p_int_atom_area': p_int_atom_area,
             'p_int_times_p_int_area': p_int_times_p_int_area,
             'p_int_atom_times_p_int_atom_area': p_int_atom_times_p_int_atom_area,
             'pyr_val': pyr_val,
             'pyr_alpha': pyr_alpha,
             'qvbur_min': qvbur_min,
             'qvbur_max': qvbur_max,
             'qvtot_min': qvtot_min,
             'qvtot_max': qvtot_max,
             'max_delta_qvbur': max_delta_qvbur,
             'max_delta_qvtot': max_delta_qvtot,
             'ovbur_min': ovbur_min,
             'ovbur_max': ovbur_max,
             'ovtot_min': ovtot_min,
             'ovtot_max': ovtot_max,
             'vbur': vbur,
             'vtot': vtot,
             'near_vbur': near_vbur,
             'far_vbur': far_vbur,
             'near_vtot': near_vtot,
             'far_vtot': far_vtot,
             'selected_dummy_idx': int(selected_dummy_idx),
             'coords_extended': coords_extended,
             'elements_extended': elements_extended,
             'dummy_idx': int(dummy_idx),
             'p_idx': int(p_idx)
             }

    # Write to an output file
    logger.debug('Writing MORFEUS results to %s', outfile.name)
    with open(outfile, 'w', encoding='utf-8') as outfile:
        outfile.write(yaml.dump(results, default_flow_style=False))

    return results