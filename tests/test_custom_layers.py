import numpy as np
import pandas as pd
import tensorflow as tf

from ionmob.models.low_parametric import get_sqrt_slopes_and_intercepts, get_slopes_and_intercepts
from ionmob.models.deep_models import ProjectToInitialSqrtCCS, ProjectToInitialCCS


def test_linear_projection_layer():
    data = pd.read_parquet('ionmob/example_data/Tenzer_unimod.parquet')
    # mz values need to be expanded to 2D for the layer to work
    mz = np.expand_dims(data.mz.values, 1)
    # charges need to be one-hot encoded and values are 0 indexed, so subtract 1
    charge = tf.one_hot(data.charge.values - 1, depth=4)

    slopes, intercepts = get_slopes_and_intercepts(data.mz, data.charge, data.ccs)
    linear_layer = ProjectToInitialCCS(slopes, intercepts)

    init_ccs = linear_layer([mz, charge])

    assert init_ccs.shape == (len(data), 1)


def test_sqrt_projection_layer():
    data = pd.read_parquet('ionmob/example_data/Zepeda_thunder_unique_unimod.parquet')
    # mz values need to be expanded to 2D for the layer to work
    mz = np.expand_dims(data.mz.values, 1)
    # charges need to be one-hot encoded and values are 0 indexed, so subtract 1
    charge = tf.one_hot(data.charge.values - 1, depth=4)

    slopes, intercepts = get_sqrt_slopes_and_intercepts(data.mz, data.charge, data.ccs, fit_charge_state_one=True)
    sqrt_layer = ProjectToInitialSqrtCCS(slopes, intercepts)

    init_ccs = sqrt_layer([mz, charge])

    assert init_ccs.shape == (len(data), 1)
