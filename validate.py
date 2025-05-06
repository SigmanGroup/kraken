#!/usr/bin/env python3
# coding: utf-8

'''
For validating the new and old yaml files to demonstrate
that this new workflow behaves the same as the old one
'''

import yaml
import logging

from io import BytesIO
from pathlib import Path
from yaml import CLoader as Loader

import numpy as np
import pandas as pd

from typing import List
from reportlab.lib import colors
from reportlab.platypus import Table, Spacer, Paragraph, TableStyle
from reportlab.platypus import SimpleDocTemplate, Image, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import getSampleStyleSheet

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import matplotlib.pyplot as plt

# Global style
styles = getSampleStyleSheet()
base_style = styles['Normal']
base_style.fontName = 'Helvetica'
base_style.fontSize = 10

def smiles_to_image(smiles: str, size: tuple = (150, 150), render_size: tuple = (600, 600)) -> Image:
    '''
    Generate a high-resolution 2D structure image from a SMILES string, scaled for sharp display in a PDF.

    Parameters
    ----------
    smiles:str
        The SMILES string of the molecule.
    size:tuple
        Display size in points (width, height) for ReportLab.
    render_size:tuple
        Pixel size (width, height) to render the image at (controls resolution).

    Returns
    -------
    Image
        A ReportLab Image object of the molecule structure.
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    AllChem.Compute2DCoords(mol)

    img = Draw.MolToImage(mol, size=render_size)

    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    return Image(buf, width=size[0], height=size[1], hAlign='LEFT')

def mol_from_elements_coords_connectivity(elements: list[str],
                                          coords: list[tuple[float, float, float]] | np.ndarray,
                                          connectivity: np.ndarray) -> Chem.Mol:
    '''
    Construct a sanitized RDKit Mol object from atomic elements, 3D coordinates, and a 0/1 connectivity matrix.

    Parameters
    ----------
    elements:list of str
        Atomic symbols (e.g., ['C', 'O', 'H']).
    coords:list of tuple[float, float, float] or np.ndarray
        Atomic coordinates.
    connectivity:np.ndarray
        Symmetric 0/1 matrix indicating bonded atom pairs.

    Returns
    -------
    Mol
        A sanitized RDKit Mol object with inferred bond orders.
    '''
    if isinstance(coords, list):
        coords = np.array(coords)
    if coords.shape[0] != len(elements) or connectivity.shape != (len(elements), len(elements)):
        raise ValueError('Inconsistent dimensions among elements, coords, and connectivity matrix.')

    mol = Chem.RWMol()
    for symbol in elements:
        mol.AddAtom(Chem.Atom(symbol))

    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if connectivity[i, j] == 1:
                mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)

    conf = Chem.Conformer(len(elements))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(x, y, z))
    mol.AddConformer(conf, assignId=True)

    # Convert to Mol and sanitize (infers bond types, valences, aromaticity)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)

    return mol

def validate_keys(new: dict,
                  old: dict):
    '''
    Tests if the first layer of keys are
    in each dictionary and tests if the
    type is correct
    '''

    new_keys = list(new.keys())
    old_keys = list(old.keys())

    for k in new_keys:
        if k not in old_keys:
            print(f'{k} is not in old keys')
        else:
            # Test if the new value is the same type as the old value
            if type(new[k]) != type(old[k]):
                print(f'new[{k}] is not the same type as old[k]: {type(new[k])} != {type(old[k])}')

            # Call recursively if the type is dict
            #if isinstance(new[k], dict) and isinstance(old[k], dict):
            #    validate_keys(new=new[k], old=old[k])


    for k in old_keys:
        if k not in new_keys:
            print(f'{k} is not in new keys')
        else:
            # Test if the new value is the same type as the old value
            if type(old[k]) != type(new[k]):
                print(f'old[{k}] is not the same type as new[k]: {type(old[k])} != {type(new[k])}')

            # Call recursively if the type is dict
            #if isinstance(new[k], dict) and isinstance(old[k], dict):
            #    validate_keys(new=new[k], old=old[k])

def float_validation(new_float: float, old_float: float) -> pd.Series:
    '''
    Computes validation metrics for the new and old values

    Parameters
    ----------
    new_float : float
        New computed value
    old_float : float
        Reference value to compare against

    Returns
    -------
    pd.Series
        Series with validation metrics: old, new, abs_diff, rel_diff (%), squared_diff
    '''
    if not isinstance(new_float, float) or not isinstance(old_float, float):
        raise TypeError('Both old and new values must be floats')

    abs_diff = abs(new_float - old_float)
    rel_diff = abs_diff / abs(old_float) if old_float != 0 else float('inf')
    squared_diff = (new_float - old_float) ** 2

    return pd.Series({
        'old': old_float,
        'new': new_float,
        'abs_diff': abs_diff,
        'rel_diff (%)': rel_diff * 100,
        'squared_diff': squared_diff
    })

def create_pdf(filename: str, elements: List[Flowable]) -> None:
    '''
    Create a PDF document from a list of ReportLab flowables.

    Parameters
    ----------
    filename:str
        The output PDF file path.
    elements:list of Flowable
        A list of ReportLab flowables (e.g., tables, paragraphs) to include in the PDF.
    '''
    doc = SimpleDocTemplate(filename,
                            pagesize=letter,
                            leftMargin=72,
                            rightMargin=72,
                            topMargin=72,
                            bottomMargin=72,
    )

    doc.build(elements)


def add_table(df: pd.DataFrame, max_width: int = int(6.0*72)) -> Table:
    '''
    Convert a pandas DataFrame into a styled ReportLab Table object.

    Parameters
    ----------
    df:pd.DataFrame
        The DataFrame to convert.
    max_width:int
        Maximum total table width in points (default: 6 inches).

    Returns
    -------
    Table
        A ReportLab Table with basic styling applied.
    '''
    # Copy and round float values to 3 decimal points
    df = df.copy()
    df = df.round(3)
    df = df.astype(str)
    df.reset_index(inplace=True)

    # Calculate maximum width for each column based on content length
    col_widths = []
    for col in df.columns:
        max_len = max(df[col].apply(len).max(), len(col))  # Maximum length: data or header
        col_widths.append(max_len * 7)  # Adjust the multiplier to get suitable padding

    total_width = sum(col_widths)
    if total_width > max_width:  # If total width exceeds max width, scale columns
        scaling_factor = max_width / total_width
        col_widths = [width * scaling_factor for width in col_widths]

    # Adjust table with calculated column widths
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data, colWidths=col_widths)

    # Table styling
    style = TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ])
    table.setStyle(style)
    return table

def plot_xy_with_fit(x: list[float],
                     y: list[float],
                     xlabel: str,
                     ylabel: str,
                     title: str,
                     save: Path | None = None) -> Image | None:
    '''
    Plot x vs. y with a line of best fit, including the linear equation and R² on the plot.
    If save is None, return a ReportLab Image object for use in PDF generation.

    Parameters
    ----------
    x: list[float]
        X-axis data.
    y: list[float]
        Y-axis data.
    xlabel: str
        Label for the x-axis.
    ylabel: str
        Label for the y-axis.
    title: str
        Title of the plot.
    save: Path | None
        File path to save the figure. If None, the figure is returned as a ReportLab Image.

    Returns
    -------
    Image | None
        ReportLab Image object if save is None, otherwise None.
    '''
    x = np.array(x)
    y = np.array(y)

    coeffs = np.polyfit(x, y, 1)
    fit_line = np.poly1d(coeffs)
    y_pred = fit_line(x)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, label='Data', color='#084AB4')
    ax.plot(x, y_pred, label='Fit', color='#A12D4D')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    eq_str = f'y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}\nR² = {r2:.3f}'
    ax.text(0.05, 0.95, eq_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top')

    ax.legend(loc='center right')
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300)
        plt.close(fig)
        return None
    else:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300)
        plt.close(fig)
        buf.seek(0)
        return Image(buf, width=6*inch, height=4*inch)

def main():
    new_yaml_dir = Path('./validation/new_dft_yamls/')
    old_yaml_dir = Path('./validation/original_dft_yamls/')

    # Define the list of molecule Kraken IDs
    kraken_ids = ['00000015', '00000030', '00000062', '00000068', '00000069',
                  '00000079', '00000084', '00000104', '00000116', '00000130',
                  '00000139', '00000148', '00000217', '00000251', '00000280',
                  '00000327', '00000329', '00000338', '00000340', '00000351',
                  '00000401', '00000449', '00000458', '00000487', '00000640',
                  '00000644', '00000648', '00000650']

    doc_elements = []

    # Set the percentage for differences above which properties are flagged
    rel_dif_thresh = 1

    # List to hold the metric dfs for all properties for all kraken_ids
    metric_dfs = []

    for id in kraken_ids:

        # Get the overall data file
        new = new_yaml_dir / f'{id}_data.yml'
        old = old_yaml_dir / f'{id}_data.yml'

        # Get the old and new conf_data files
        new_confdata = new_yaml_dir / f'{id}_confdata.yml'
        old_confdata = old_yaml_dir / f'{id}_confdata.yml'

        with open(new, 'r', encoding='utf-8') as f:
            new_content = yaml.load(f, Loader=Loader)

        with open(old, 'r', encoding='utf-8') as f:
            old_content = yaml.load(f, Loader=Loader)

        with open(new_confdata, 'r', encoding='utf-8') as f:
            new_confdata = yaml.load(f, Loader=Loader)

        with open(old_confdata, 'r', encoding='utf-8') as f:
            old_confdata = yaml.load(f, Loader=Loader)

        for k, v in old_confdata.items():
            elements = np.array(old_confdata[k]['elements'])
            coords = np.array(old_confdata[k]['coords'])
            conmat = np.array(old_confdata[k]['conmat'])
            mol = mol_from_elements_coords_connectivity(elements, coords, conmat)
            rdDetermineBonds.DetermineBondOrders(mol)
            mol = AllChem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            break

        validate_keys(new=new_content, old=old_content)

        float_dfs = []

        # Validate floats in the first layer
        for k in new_content.keys():
            if isinstance(new_content[k], float):
                metrics = float_validation(new_float=new_content[k], old_float=old_content[k])

                float_dfs.append(pd.DataFrame([metrics], index=[k]))

        # Validate floats in the Boltzmann averaged data
        for k in new_content['boltzmann_averaged_data'].keys():

            if isinstance(new_content['boltzmann_averaged_data'][k], float):
                metrics = float_validation(new_float=new_content['boltzmann_averaged_data'][k],
                                           old_float=old_content['boltzmann_averaged_data'][k])

                float_dfs.append(pd.DataFrame([metrics], index=[f'{k}_boltz']))

                #if k.lower() == 'vbur_ovtot_min':
                #    print(new_content['boltzmann_averaged_data'][k])
                #    print(old_content['boltzmann_averaged_data'][k])
                #    print(k)
                #    print(pd.DataFrame([metrics], index=[f'{k}_boltz']))
                #    exit()

        # Validate floats in the delta data
        for k in new_content['delta_data'].keys():
            if isinstance(new_content['delta_data'][k], float):
                metrics = float_validation(new_float=new_content['delta_data'][k], old_float=old_content['delta_data'][k])

                float_dfs.append(pd.DataFrame([metrics], index=[f'{k}_delta']))

        # Validate floats in the max data
        for k in new_content['max_data'].keys():
            if isinstance(new_content['max_data'][k], float):
                metrics = float_validation(new_float=new_content['max_data'][k], old_float=old_content['max_data'][k])

                float_dfs.append(pd.DataFrame([metrics], index=[f'{k}_max']))

        # Validate floats in the min data
        for k in new_content['min_data'].keys():
            if isinstance(new_content['min_data'][k], float):
                metrics = float_validation(new_float=new_content['min_data'][k], old_float=old_content['min_data'][k])

                float_dfs.append(pd.DataFrame([metrics], index=[f'{k}_min']))

        # Validate floats in the Vbur_min_conf data
        for k in new_content['vburminconf_data'].keys():
            if isinstance(new_content['vburminconf_data'][k], float):
                metrics = float_validation(new_float=new_content['vburminconf_data'][k], old_float=old_content['vburminconf_data'][k])

                float_dfs.append(pd.DataFrame([metrics], index=[f'{k}_vbur_min_conf']))

        df = pd.concat(float_dfs)
        df.index.name = 'property'

        # Drop the time category
        df.drop(labels=['time_all'], inplace=True)

        # Get the rows that are equal to infinity
        infinity_df = df[df['rel_diff (%)'] == np.inf].copy()
        infinity_rows = []
        to_drop = []
        for prop, row in infinity_df.iterrows():
            if row['old'] == row['new']:
                infinity_rows.append(prop)
        #df.drop(labels=infinity_rows, inplace=True)

        # Append the dataframes to the metric dfs
        metric_dfs.append(df.copy(deep=True))

        # Get properties with greater than 1% difference
        df = df[df['rel_diff (%)'] >= rel_dif_thresh].sort_values('rel_diff (%)', ascending=False)

        doc_elements.extend([Paragraph(f'KRAKEN ID {id}', styles['Heading1']),
                             smiles_to_image(smiles, size=[100, 100]),
                             #Paragraph(f'These properties had identical values and relative differences of infinity: {infinity_rows}\n', styles['Normal']),
                             Spacer(1, 2),
                             Paragraph(f'Table 1. Properties that exceed {rel_dif_thresh}% relative difference', styles['Normal']),
                             add_table(df),
                             PageBreak(),
                             ])


        #if id == '00000068':
        #    break

    # Make a list of properties for which we want regressions
    flagged_properties = []

    # Go through all metric dfs
    for df in metric_dfs:

        # Get the subset that exceed the threshold
        df = df[df['rel_diff (%)'] >= rel_dif_thresh].sort_values('rel_diff (%)', ascending=False)
        flagged_properties.extend(list(df.index))

    # Add properties to the report
    doc_elements.append(Paragraph(f'Linear regressions for all properties with at least 1% difference in any tested monophosphine', styles['Heading1']),)
    for _prop in list(set(flagged_properties)):
        print(_prop)

        # Get the new and old values
        _new_vals = [_df.loc[[_prop], 'new'].values[0] for _df in metric_dfs]
        _old_vals = [_df.loc[[_prop], 'old'].values[0] for _df in metric_dfs]

        if all([x == _new_vals[0] for x in _new_vals]):
            print(f'{_prop} had all 0s for _new_vals. Skipping. {_new_vals}')
            continue

        if all([x == _old_vals[0] for x in _old_vals]):
            print(f'{_prop} had all 0s for _old_vals. Skipping. {_old_vals}')
            continue

        reg_plot = plot_xy_with_fit(x=_old_vals,
                                    y=_new_vals,
                                    xlabel=f'{_prop} (old)',
                                    ylabel=f'{_prop} (new)',
                                    title=_prop,
                                    save=None) # Path(f'./validation/{_prop}.png')

        doc_elements.append(reg_plot)


    create_pdf('./validation/validation_report.pdf', doc_elements)




if __name__ == "__main__":
    main()



