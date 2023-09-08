import numpy as np
import pandas as pd
from ionmob.utilities.chemistry import calculate_mz, ccs_to_one_over_reduced_mobility, reduced_mobility_to_ccs


def test_mz_calculation():
    """
    Test that the mz calculation works as expected
    """
    df = pd.read_parquet('ionmob/example_data/Tenzer_unimod.parquet')

    # calculate the mz from the sequence and charge
    df['mz_calc'] = df.apply(lambda x: calculate_mz(x['sequence-tokenized'], x['charge']), axis=1)
    mz_calc = df.mz_calc.values
    mz = df.mz.values

    # calculate the difference between the two, rounded to 4 decimal places to account for floating point errors
    differences = [np.round(x - y, 4) for x, y in zip(mz_calc, mz)]

    assert all([x == 0.0 for x in differences])


def test_ccs_one_over_k0_translation():
    """
    Test that the ccs to 1/k0 and back translation works as expected
    """
    df = pd.read_parquet('ionmob/example_data/Tenzer_unimod.parquet')
    
    # read data from given parquet file
    mz, charge, ccs = df.mz.values, df.charge.values, df.ccs.values
    # calculate the inverse of the reduced mobility
    inv_mob = ccs_to_one_over_reduced_mobility(ccs, mz, charge)
    # calculate the ccs from the inverse of the reduced mobility
    ccs_calc = reduced_mobility_to_ccs(inv_mob, mz, charge)
    # calculate the difference between the two, rounded to 4 decimal places to account for floating point errors
    differences = [np.round(x - y, 4) for x, y in zip(ccs_calc, ccs)]
    assert all([x == 0.0 for x in differences])
