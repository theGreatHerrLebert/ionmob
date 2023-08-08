import numpy as np
import pandas as pd
from ionmob.utilities.chemistry import calculate_mz


def test_mz_calculation():
    """
    Test that the mz calculation works as expected
    """
    df = pd.read_parquet('example_data/Tenzer_unimod.parquet')

    # calculate the mz from the sequence and charge
    df['mz_calc'] = df.apply(lambda x: calculate_mz(x['sequence-tokenized'], x['charge']), axis=1)
    mz_calc = df.mz_calc.values
    mz = df.mz.values

    # calculate the difference between the two, rounded to 4 decimal places to account for floating point errors
    differences = [np.round(x - y, 4) for x, y in zip(mz_calc, mz)]

    assert all([x == 0.0 for x in differences])
