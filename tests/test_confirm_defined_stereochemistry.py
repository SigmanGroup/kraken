#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Tests undefined stereochemistry handling
'''

import pytest

from kraken.utils import confirm_defined_stereochemistry

# Test phosphine
TWO_UNDEFINED_POINT_STEREOCENTERS = 'O=C(O)C1CCCC1P(c1ccccc1)c1ccccc1'
ONE_UNDEFINED_POINT_STEREOCENTERS = 'O=C(O)[C@@H]1CCCC1P(c1ccccc1)c1ccccc1'
ZERO_UNDEFINED_POINT_STEREOCENTERS = 'O=C(O)[C@@H]1CCC[C@@H]1P(c1ccccc1)c1ccccc1'

# UNDF = undefined, DF = defined
UNDF_DOUBLE_UNDF_POINT = 'CC(Br)CCC=CP(c1ccccc1)c1ccccc1'
UNDF_DOUBLE_DF_POINT = 'C[C@H](Br)CCC=CP(c1ccccc1)c1ccccc1'
DF_DOUBLE_UNDF_POINT_E = 'CC(Br)CC/C=C/P(c1ccccc1)c1ccccc1'   # Defined E
DF_DOUBLE_UNDF_POINT_Z = r'CC(Br)CC/C=C\P(c1ccccc1)c1ccccc1'   # Defined Z
UNDF_DOUBLE_BOND = 'CC(C)C=CP(c1ccccc1)c1ccccc1'

# One stereogenic center in total
UNDF_POINT_STEREOCENTER = 'CCC(C)P(c1ccccc1)c1ccccc1'

# RDKit detected stereochemistry at heteroatoms
HETEROATOM_STEREOCHEM = 'NCP(CCCCS(=O)(=O)O)c1ccccc1'

@pytest.mark.parametrize(
    "smiles",
    [
        TWO_UNDEFINED_POINT_STEREOCENTERS,
        ONE_UNDEFINED_POINT_STEREOCENTERS,
        UNDF_DOUBLE_UNDF_POINT,
        DF_DOUBLE_UNDF_POINT_E,
        DF_DOUBLE_UNDF_POINT_Z,
        UNDF_DOUBLE_BOND,
    ],
)

def test_confirm_defined_stereochemistry_rejects_expected(smiles: str) -> None:
    '''
    Verifies that SMILES with ambiguous stereochemistry are rejected.

    Parameters
    ----------
    smiles: str
        Input SMILES string expected to raise ValueError.

    Returns
    -------
    None
    '''
    with pytest.raises(ValueError):
        confirm_defined_stereochemistry(smiles)

@pytest.mark.parametrize(
    "smiles",
    [
        ZERO_UNDEFINED_POINT_STEREOCENTERS,
        UNDF_POINT_STEREOCENTER,
        HETEROATOM_STEREOCHEM
    ],
)

def test_confirm_defined_stereochemistry_accepts_expected(smiles: str) -> None:
    '''
    Verifies that SMILES with either fully specified stereochemistry or a single
    undefined point stereocenter are accepted. It asserts that the function
    returns a non-empty str.

    Parameters
    ----------
    smiles: str
        Input SMILES string expected to be accepted.

    Returns
    -------
    None
    '''
    out = confirm_defined_stereochemistry(smiles)

    # Confirm it returns a string SMILES
    assert isinstance(out, str)

    # Confirm it is not empty
    assert out.strip() != ""

def test_confirm_defined_stereochemistry_rejection_messages_include_context() -> None:
    '''
    Ensures that rejection messages are informative for double-bond ambiguity.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    with pytest.raises(ValueError) as excinfo:
        confirm_defined_stereochemistry(UNDF_DOUBLE_BOND)

    msg = str(excinfo.value)

    # The function's double-bond path uses this phrase
    assert "double bond" in msg.lower()
    assert "ambiguous" in msg.lower()
