#!/usr/bin/env python3
# coding: utf-8

'''
Primary script for running Kraken

#TODO INCLUDE ORIGINAL AUTHORS OF CODE

Major revisions March 2024 provided by:
    James Howard, PhD

FOR TESTING
run_kraken_conf_search.py -i "CP(C)C" --kraken-id 88888888  --calculation-dir ./data/ --nprocs 4 --conversion-flag 4 --noreftopo --nocross
'''

import os
import sys
import argparse
import logging

from pathlib import Path

import morfeus
import pandas as pd

from rdkit import Chem

from kraken.semiempirical import _get_crest_version
from kraken.xtb import _get_xtb_version
from kraken.utils import _str_is_smiles

from kraken.new_run_kraken import run_kraken_calculation

logger = logging.getLogger(__name__)

DESCRIPTION = r'''
╔══════════════════════════════════════╗
║   | |/ / _ \  /_\ | |/ / __| \| |    ║
║   | ' <|   / / _ \| ' <| _|| .` |    ║
║   |_|\_\_|_\/_/ \_\_|\_\___|_|\_|    ║
╚══════════════════════════════════════╝
Kolossal viRtual dAtabase for moleKular dEscriptors
of orgaNophosphorus ligands.


This is the first script required to run the Kraken
workflow. This script accepts either SMILES strings
or .xyz files directly. Alternatively, users may
specify a .csv file that contains the following columns.

'CONVERSION_FLAG', 'KRAKEN_ID', 'SMILES'

The script can be called directly in the terminal or
called in a shell script that is submitted to SLURM
or some other job scheduler.
'''

DESCRIPTION = '\n'.join(line.center(80) for line in DESCRIPTION.strip('\n').split('\n'))

def get_args() -> argparse.Namespace:
    '''Gets the arguments for running Kraken'''

    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, 2, 40),
        allow_abbrev=False,
        add_help=False)

    parser.add_argument('-h', '--help',
                        action='help',
                        default=argparse.SUPPRESS,
                        help='Show this help message and exit.\n\n')

    parser.add_argument('-i', '--input',
                        dest='input',
                        required=True,
                        help='SMILES string, XYZ file, or CSV file\n\n',
                        metavar='STR')

    parser.add_argument('--kraken-id',
                        dest='kraken_id',
                        required=True,
                        help='8-digit Kraken ID number specific to the input\n\n',
                        metavar='INT')

    parser.add_argument('--nprocs',
                        dest='nprocs',
                        default=4,
                        help='Number of processors to use for CREST and xTB\n\n',
                        metavar='INT')

    # Not in order: 0=RDKit, 1=molconvert, 2=??? (likely coords from XYZ file), 3=obabel, 4=try_all_methods #TODO
    parser.add_argument('--conversion-flag',
                        choices=['0', '1', '2', '3', '4'],
                        dest='conversion_flag',
                        help='Method for 3D structure generation (see README.md, default=4)\n\n')

    parser.add_argument('--calculation-dir',
                        dest='calc_dir',
                        default='./data/',
                        help='Path to the directory that will contain the results of Kraken\n\n',
                        metavar='DIR')

    parser.add_argument('--noreftopo',
                        action='store_true',
                        help='Employs the --noreftopo flag for CREST calculations\n\n')

    parser.add_argument('--nocross',
                        action='store_true',
                        help='Employs the --nocross flag for CREST calculations\n\n')

    parser.add_argument('--reduce_crest_output',
                        action='store_true',
                        help='Delete files from CREST to reduce disk usage (not recommended)\n\n')

    parser.add_argument('--disable-nickel-calculation',
                        action='store_true',
                        help='Disables the calculation of the Ni(CO)3 complex\n\n')

    parser.add_argument('--disable-ligand-calculation',
                        action='store_true',
                        help='Disables the calculation of the free ligand\n\n')

    parser.add_argument('--debug', action='store_true', help='Prints debug information\n\n')

    args = parser.parse_args()

    args.calc_dir = Path(args.calc_dir)
    if not args.calc_dir.exists():
        raise FileNotFoundError(f'The calculation directory {args.calc_dir.absolute()} does not exist.')

    return args

def _correct_kraken_id(kid: int | str) -> str:
    '''
    Formats a Kraken ID as an 8-digit zero-padded string.

    Parameters
    ----------
    kid : int or str
        The Kraken ID. Must be entirely numeric and 8 digits
        or less.

    Returns
    -------
    str
        A zero-padded 8-digit Kraken ID string.

    Raises
    ------
    ValueError
        If the ID is not numeric or exceeds 8 digits.
    '''
    kid = str(kid)

    if not kid.isdigit():
        raise ValueError(f'Kraken ID must be numeric: {kid}')

    if len(kid) > 8:
        raise ValueError(f'Kraken ID too long ({len(kid)} digits). Max allowed is 8.')

    return kid.zfill(8)

def _parse_csv(csv: Path) -> tuple[list, list, list]:
    '''
    Reads a csv file with headers KRAKEN_ID, SMILES,
    and CONVERSION_FLAG and returns them as lists.

    Parameters
    ----------
    csv: Path
        Path to the csv file.

    Returns
    ----------
    tuple[list, list, list]
        Lists of Kraken ids, inputs (smiles), and
        conversion flags.
    '''
    if isinstance(csv, str):
        csv = Path(csv)

    if csv.suffix != '.csv':
        raise ValueError(f'{csv.name} is not a .csv file.')
    df = pd.read_csv(csv, header=0)

    if sorted(list(set(df.columns))) != ['CONVERSION_FLAG', 'KRAKEN_ID', 'SMILES']:
        raise ValueError(f'{csv.name} is not properly formatted. Only use columns CONVERSION_FLAG, KRAKEN_ID, and SMILES.')

    return df['KRAKEN_ID'].to_list(), df['SMILES'].to_list(), df['CONVERSION_FLAG'].to_list()

def _parse_input(args: argparse.Namespace) -> tuple[list, list, list]:
    '''
    returns ids, inputs, conversion_flags
    '''

    # Make the list of calculations to return to main function
    ids = []
    inputs = []
    conversion_flags = []

    # Handling SMILES first
    if _str_is_smiles(args.input):

        # If it's not a Path, check if it's a SMILES
        mol = Chem.MolFromSmiles(args.input)

        if args.conversion_flag is None:
            raise ValueError(f'A conversion flag must be supplied if using SMILES as an initial input.')

        conversion_flags.append(args.conversion_flag)

        if args.kraken_id is None:
            raise ValueError(f'A kraken_id must be supplied if using SMILES as an initial input.')

        ids.append(args.kraken_id)
        inputs.append(args.input)

    elif Path(args.input).exists():
        # TODO This is passed to a read_xyz func later, convert to path
        file = Path(args.input)

        if file.suffix == '.csv':
            return _parse_csv(file)

        elif file.suffix == '.xyz':
            logger.info('Input contained .xyz file. Using conversion flag 2')
            conversion_flags.append(2)

            if args.kraken_id is None:
                raise ValueError(f'A kraken_id must be supplied if using a .xyz file as an initial input.')

            ids.append(args.kraken_id)
            inputs.append(args.input)

        else:
            raise NotImplementedError(f'{file.name} is not supported as an input.')
    else:
        raise ValueError(f'{args.input} is not a valid SMILES string or a Path that does not exist.')

    return ids, inputs, conversion_flags

def main():
    '''Main entrypoint'''
    #TODO figure out what the user is inputting in this section
    #TODO for now we're just doing smiles
    args = get_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    XTB_VERSION = _get_xtb_version()
    CREST_VERSION = _get_crest_version()

    logger.info('Using xTB version %s', str(XTB_VERSION))
    logger.info('Using CREST version %s', str(CREST_VERSION))

    if XTB_VERSION != '6.2.2':
        logger.warning('xTB version does not match the original kraken version 6.2.2')

    if CREST_VERSION != '2.8':
        logger.warning('CREST version does not match the original kraken version 2.8')

    if morfeus.__version__ != '0.5.0':
        logger.warning('MORFEUS version does not match the original kraken version 0.5.0')

    # Get all requested kraken_ids, inputs (smiles), and conversion_flags
    ids, inputs, conversion_flags = _parse_input(args)
    ids = [_correct_kraken_id(x) for x in ids]

    # Get the user supplied args
    reduce_crest_output = bool(args.reduce_crest_output)
    nprocs = int(args.nprocs)
    calc_dir = Path(args.calc_dir)
    metal_char = 'Ni'

    # Create a list of jobs to do for each ligand
    jobs = []
    if not args.disable_ligand_calculation:
        jobs.append('noNi')
    if not args.disable_nickel_calculation:
        jobs.append('Ni')

    # Set the environment variables for the rest of the run
    logger.info('Setting OMP_NUM_THREADS to %d', nprocs)
    os.environ["OMP_NUM_THREADS"] = str(nprocs)
    logger.info('Setting MKL_NUM_THREADS to %d', nprocs)
    os.environ["MKL_NUM_THREADS"] = str(nprocs)

    logger.debug('Inital input: %s', str(args.input))
    logger.info('Found %d Kraken IDs for the Kraken workflow', len(ids))

    for kraken_id, structure_input, conversion_flag in zip(ids, inputs, conversion_flags):
        logger.info('\t%s\tstructure_input=%s\tconversion_flag=%s', str(kraken_id), str(structure_input), str(conversion_flag))

    # Set default settings from the original workflow
    # These should not be removed
    # because they are saved in the results file
    settings = {
        'max_E': 6.0,
        'max_p': 0.1,
        'OMP_NUM_THREADS': nprocs,
        'MKL_NUM_THREADS': nprocs,
        'dummy_distance': 2.1,
        'remove_scratch': True,
        'reduce_output': reduce_crest_output,
        'add_Pd_Cl2': False,
        'add_Pd_Cl2_PH3': False,
        'add_Ni_CO_3': False
    }

    for k, v in settings.items():
        logger.debug('\t\t%s\t%s', k, v)

    logger.info('The metal_char is %s', metal_char)
    settings['char'] = metal_char

    # Adding this setting for posterity
    settings['use_scratch'] = False

    # Iterate over the different kraken entries
    for kraken_id, structure_input, conversion_flag in zip(ids, inputs, conversion_flags):

        conversion_flag = int(conversion_flag)

        # Define the directory for the molecule
        mol_dir = calc_dir / kraken_id
        mol_dir.mkdir(exist_ok=True)

        dft_dir = mol_dir / 'dft'
        dft_dir.mkdir(exist_ok=True)

        run_kraken_calculation(kraken_id=kraken_id,
                               structure_input=structure_input,
                               mol_dir=mol_dir,
                               dft_dir=dft_dir,
                               reduce_crest_output=settings['reduce_output'],
                               dummy_distance=settings['dummy_distance'],
                               settings=settings,
                               metal_char=metal_char,
                               jobs=jobs,
                               nprocs=nprocs,
                               conversion_flag=conversion_flag,
                               add_Pd_Cl2=settings['add_Pd_Cl2'],
                               add_Pd_Cl2_PH3=settings['add_Pd_Cl2_PH3'],
                               add_Ni_CO_3=settings['add_Ni_CO_3'])

    logger.info('KRAKEN TERMINATED NORMALLY')

if __name__ == "__main__":
    main()


