#!/usr/bin/env python3
# coding: utf-8

'''
For converting confdata yamls from Kraken into SDF files
'''

import re
import os
import sys
import math
import yaml
import shutil
import logging
import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

from rdkit import Chem


def main():
    '''
    Main function
    '''
    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)-5s - %(asctime)s] [%(module)s] %(message)s',
        datefmt='%m/%d/%Y:%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    list_of_series = []

    for kraken_id in ['90000031', '90000032', '90000033']:

        data_dir = Path(f'./{kraken_id}')

        if not data_dir.exists():
            raise FileNotFoundError(f'Could not locate {data_dir.absolute()}')

        # Define the two yamls
        confdata_yml = data_dir / f'{kraken_id}_confdata.yml'
        data_yml = data_dir / f'{kraken_id}_data.yml'

        # Read in the summary data file
        with open(data_yml, 'r', encoding='utf-8') as f:
            data = yaml.full_load(f)

        # Make a dictionary to convert the data
        main_dict = {'name': str(kraken_id),
                     'smiles': ''}

        # Boltzmann weighted properties
        boltz = list(data['boltzmann_averaged_data'])
        boltz_values = []
        for value in boltz:
            boltz_desc = data['boltzmann_averaged_data'][value]
            boltz_values.append(boltz_desc)
        boltz_dict = {boltz[i] + str('_boltz'): boltz_values[i] for i in range(len(boltz))}
        main_dict.update(boltz_dict)

        # Delta descriptors
        delta = list(data['delta_data'])
        delta_values = []
        for value in delta:
            delta_desc = data['delta_data'][value]
            delta_values.append(delta_desc)
        delta_dict = {delta[i] + str('_delta'): delta_values[i] for i in range(len(delta))}
        main_dict.update(delta_dict)

        # Max descriptors
        max_data = list(data['max_data'])
        max_data_values = []
        for value in max_data:
            max_desc = data['max_data'][value]
            max_data_values.append(max_desc)
        max_dict = {max_data[i] + str('_max'): max_data_values[i] for i in range(len(max_data))}
        main_dict.update(max_dict)

        # Min descriptors
        min_data = list(data['min_data'])
        min_data_values = []
        for value in min_data:
            min_desc = data['min_data'][value]
            min_data_values.append(min_desc)
        min_dict = {min_data[i] + str('_min'): min_data_values[i] for i in range(len(min_data))}
        main_dict.update(min_dict)

        # Vbur_min_conf descriptors
        vburminconf = list(data['vburminconf_data'])
        vburminconf_values = []
        for value in vburminconf:
            vburminconf_desc = data['vburminconf_data'][value]
            vburminconf_values.append(vburminconf_desc)
        vburminconf_dict = {vburminconf[i] + str('_vburminconf'): vburminconf_values[i] for i in range(len(vburminconf))}
        main_dict.update(vburminconf_dict)

        list_of_series.append(pd.Series(data=main_dict))

    df = pd.DataFrame(list_of_series)
    df.set_index('name', inplace=True, drop=True)
    print(df)

    df.to_csv('./new_phosphines.csv')
    exit()


if __name__ == "__main__":
    main()