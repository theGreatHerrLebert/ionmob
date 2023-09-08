import pkg_resources

import pandas as pd
from ionmob.models.low_parametric import get_sqrt_slopes_and_intercepts, get_slopes_and_intercepts


def test_linear_fit_four_charges():
    """
    Test that the linear fit works as expected
    """
    df = pd.read_parquet('ionmob/example_data/Tenzer_unimod.parquet')
    slopes, intercepts = get_slopes_and_intercepts(df.mz, df.charge, df.ccs)
    assert len(slopes) == len(intercepts) == 4


def test_sqrt_fit_four_charges():
    """
    Test that the sqrt fit works as expected
    """
    df = pd.read_parquet('ionmob/example_data/Tenzer_unimod.parquet')
    slopes, intercepts = get_sqrt_slopes_and_intercepts(df.mz, df.charge, df.ccs, fit_charge_state_one=False)
    assert len(slopes) == len(intercepts) == 4
    assert slopes[0] == 0.0


def test_sqrt_fit_five_charges():
    """
    Test that the sqrt fit works as expected
    """
    df = pd.read_parquet('ionmob/example_data/Zepeda_thunder_unique_unimod.parquet')
    slopes, intercepts = get_sqrt_slopes_and_intercepts(df.mz, df.charge, df.ccs, fit_charge_state_one=True)
    assert len(slopes) == len(intercepts) == 4
    assert slopes[0] != 0.0
