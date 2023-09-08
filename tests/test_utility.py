import numpy as np
import pandas as pd
from ionmob.utilities.chemistry import reduced_mobility_to_ccs
from ionmob.utilities.utility import get_ccs_shift, preprocess_peaks_sequence


def test_apply_ccs_shift():
    data = pd.read_csv('ionmob/example_data/DB_search_psm.zip')
    data = data.rename(columns={'Z': 'charge'})

    # create tokenized sequence
    data['sequence-tokenized'] = data.apply(lambda s: preprocess_peaks_sequence(s['Peptide']), axis=1)
    data['sequence-join'] = data.apply(lambda s: ''.join(s['sequence-tokenized']), axis=1)

    # calculate ccs values from reduced mobility
    data['k0_start'] = data.apply(lambda k: float(k['1/k0 Range'].split('-')[0]), axis=1)
    data['k0_end'] = data.apply(lambda k: float(k['1/k0 Range'].split('-')[1]), axis=1)
    data['k0_mean'] = data.apply(lambda k: (k['k0_start'] + k['k0_end']) / 2, axis=1)
    data['ccs'] = data.apply(lambda k: reduced_mobility_to_ccs(k['k0_mean'], k['m/z'], k['charge']), axis=1)

    ref_data = pd.read_parquet('ionmob/example_data/Tenzer_unimod.parquet')
    ref_data['sequence-join'] = ref_data.apply(lambda s: ''.join(s['sequence-tokenized']), axis=1)

    # join on sequence to pairwise ccs values
    table_joined = pd.merge(data, ref_data, left_on=['sequence-join', 'charge'], right_on=['sequence-join', 'charge'])

    # calculate the ccs shift
    diffs = np.abs(np.mean(table_joined.ccs_x.values - table_joined.ccs_y.values))
    # check that the mean difference is greater 0
    assert diffs > 1

    ccs_diff = get_ccs_shift(data, ref_data)
    data['ccs_shifted'] = data.apply(lambda x: x['ccs'] + ccs_diff, axis=1)

    data_both = pd.merge(data, ref_data, left_on=['sequence-join', 'charge'], right_on=['sequence-join', 'charge'])

    # calculate the ccs shift
    diffs = np.mean(data_both.ccs_y.values - data_both.ccs_shifted.values)

    # check that the mean difference is less than 0.1
    assert -1 <= diffs <= 1

