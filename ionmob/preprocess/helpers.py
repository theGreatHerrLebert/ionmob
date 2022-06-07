import io
import json
import re

from itertools import combinations
import numpy as np
import pandas as pd
import tensorflow as tf
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from numpy import ndarray

from scipy.optimize import curve_fit


def apply_shift_per_charge(table, reference):
    """
    shift a given dataset by a constant offset based on sequence and charge pairs of reference
    :param reference: a reference dataset to align CCS values to
    :param table: a table with CCS values to be shifted
    :return: a table with an appended shifted CCS value column
    """

    shift_list = []

    table['sequence'] = table.apply(lambda r: ''.join(list(r['sequence-tokenized'])), axis=1)
    reference['sequence'] = reference.apply(lambda r: ''.join(list(r['sequence-tokenized'])), axis=1)

    for c in range(2, 5):
        reference_tmp = reference[reference.charge == c]
        table_tmp = table[table.charge == c]

        both = pd.merge(left=reference_tmp, right=table_tmp, right_on=['sequence', 'charge'],
                        left_on=['sequence', 'charge'])

        factor = np.mean(both.ccs_x - both.ccs_y)
        table_tmp['ccs_shifted'] = table_tmp['ccs'] + factor
        shift_list.append(table_tmp)

    shifted_data = pd.concat(shift_list).drop(columns=['sequence'])

    return shifted_data


def get_ccs_shift(table: pd.DataFrame, reference: pd.DataFrame, use_charge_state: int = 2) -> ndarray:
    """
    shift a given dataset by a constant offset based on sequence and charge pairs of reference
    :param reference: a reference dataset to align CCS values to
    :param table: a table with CCS values to be shifted
    :param use_charge_state:
    :return: a global shift factor
    """

    shift_list = []

    tmp_table = table.copy(deep=True)
    tmp_reference = reference.copy(deep=True)

    tmp_table['sequence'] = table.apply(lambda r: ''.join(list(r['sequence-tokenized'])), axis=1)
    tmp_reference['sequence'] = reference.apply(lambda r: ''.join(list(r['sequence-tokenized'])), axis=1)

    reference_tmp = tmp_reference[tmp_reference.charge == use_charge_state]
    table_tmp = tmp_table[tmp_table.charge == use_charge_state]

    both = pd.merge(left=reference_tmp, right=table_tmp, right_on=['sequence', 'charge'],
                    left_on=['sequence', 'charge'])

    return np.mean(both.ccs_x - both.ccs_y)


def get_sqrt_slopes_and_intercepts(mz: ndarray, charge: ndarray, ccs: ndarray, fit_charge_state_one: bool = False):
    """

    Args:
        mz:
        charge:
        ccs:
        fit_charge_state_one:

    Returns:

    """

    if fit_charge_state_one:
        slopes, intercepts = [], []
    else:
        slopes, intercepts = [0.0], [0.0]

    if fit_charge_state_one:
        c_begin = 1
    else:
        c_begin = 2

    for c in range(c_begin, 5):
        def fit_func(x, a, b):
            return a * np.sqrt(x) + b

        triples = list(filter(lambda x: x[1] == c, zip(mz, charge, ccs)))
        
        mz_tmp, charge_tmp = np.array([x[0] for x in triples]), np.array([x[1] for x in triples])
        ccs_tmp = np.array([x[2] for x in triples])

        popt, _ = curve_fit(fit_func, mz_tmp, ccs_tmp)

        slopes.append(popt[0])
        intercepts.append(popt[1])

    return np.array(slopes, np.float32), np.array(intercepts, np.float32)


def get_non_overlapping_pairs(ds_ref, ds_test):
    """
    reduce a dataframe to only contain seq, charge pairs not present in ref dataset
    :ds_ref: dataframe containing all charge, seq pairs that should be excluded from other frame
    :ds_test: dataframe that should be reduced to only contain seq, charge pairs not in reference
    :return: dataframe only containing seq, charge pairs not in ref data
    """
    ref_pairs = set(zip(ds_ref.sequence, ds_ref.charge))
    test_pairs = set(zip(ds_test.sequence, ds_test.charge))
    candidates = test_pairs - ref_pairs

    row_list = []
    for index, row in ds_test.iterrows():
        if (row['sequence'], row['charge']) in candidates:
            row_list.append(row)

    return pd.DataFrame(row_list)


def align_annotation(sequence: str, from_str: str = '(Oxidation (M))', to_str: str = '(ox)'):
    """
    replace parts of a string with other string, can be used for annotation alignment for e.g. modifications
    :param from_str: string part to be changed
    :param to_str: string that changed part should be changed into
    :param sequence: sequence to change
    :return: changed sequence
    """
    length = len(from_str)

    while sequence.find(from_str) != -1:
        start_index = sequence.find(from_str)
        left = sequence[:start_index]
        right = sequence[start_index + length:]
        sequence = left + to_str + right

    return sequence


def sequence_to_tokens(sequence: str, drop_ends: bool = False):
    """
    transform a sequence to a set of tokens
    :param sequence: a sequence to be tokenized
    :param drop_ends: if true, start and stop AAs will not be treated as separate tokens
    :return: a tokenized sequence
    """
    seq = sequence.replace('_', '#').replace('(ox)', '§').replace('(ac)', '&')

    # turn string into list of symbols
    seq_list = list(seq)

    matches_ox = re.finditer('§', seq)
    matches_ac = re.finditer('&', seq)

    matches_ox = [match.start() for match in matches_ox]
    matches_ac = [match.start() for match in matches_ac]

    if len(matches_ox) > 0:
        # the given symbol needs to be replaced by its modified version
        for match in matches_ox:
            mod_symbol = seq_list[match - 1]
            seq_list[match - 1] = mod_symbol + '-OX'

    while '§' in seq_list:
        seq_list.remove('§')

    if len(matches_ac) > 0:
        # the given symbol needs to be replaced by its modified version
        for match in matches_ac:
            mod_symbol = seq_list[match + 1]
            seq_list[match + 1] = mod_symbol + '-AC'

    while '&' in seq_list:
        seq_list.remove('&')

    if not drop_ends:
        s_start, s_end = seq_list[1], seq_list[len(seq_list) - 2]
        seq_list[1] = '#-' + s_start
        seq_list[len(seq_list) - 2] = s_end + '-#'
        seq_list = seq_list[1:-1]

    if drop_ends:
        return seq_list[1:-1]

    return seq_list


def fit_tokenizer(sequence_tokens: list):
    """
    will create a tensorflow tokenizer and fit it on a given set of tokens
    CAUTION, tokens should be single AAs, so a sequence should be provided as a list of tokens
    :param sequence_tokens: a list of lists of sequences as tokens
    :return: a tokenizer ready to be used for tokens -> ids and ids -> tokens conversion
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=False, lower=False)
    tokenizer.fit_on_texts(sequence_tokens)
    return tokenizer


def tokenizer_to_json(tokenizer: tf.keras.preprocessing.text.Tokenizer, path: str):
    """
    save a fit keras tokenizer to json for later use
    :param tokenizer: fit keras tokenizer to save
    :param path: path to save json to
    """
    tokenizer_json = tokenizer.to_json()
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def tokenizer_from_json(path: str):
    """
    load a pre-fit tokenizer from a json file
    :param path: path to tokenizer as json file
    :return: a keras tokenizer loaded from json
    """
    with open(path) as f:
        data = json.load(f)
    return tf.keras.preprocessing.text.tokenizer_from_json(data)


def get_gravy_score(seq: str, drop_ends: bool = True, normalize: bool = True):
    """
    calculate normalized gravy scores for a given sequence
    :param seq: peptide sequence
    :param drop_ends: if true, first and last character will be stripped from sequence
    :param normalize:
    :return: gravy score normalized by sequence length
    """
    seq = seq.replace('(ac)', '')
    seq = seq.replace('(ox)', '')
    if drop_ends:
        seq = seq[1:-1]
        
    if normalize:
        return ProteinAnalysis(seq).gravy() / len(seq)
    
    return ProteinAnalysis(seq).gravy()


def get_helix_score(seq: str, drop_ends: bool = True):
    """
    calculate portion of helix peptides
    :param seq: sequence to calculate helix portion of
    :param drop_ends: if true, first and last character will be stripped from sequence
    :return: helix portion
    """
    seq = seq.replace('(ac)', '')
    seq = seq.replace('(ox)', '')

    if drop_ends:
        seq = seq[1:-1]

    return ProteinAnalysis(seq).secondary_structure_fraction()[0]


def get_tokens_order(seq_p: str):
    """
    tokens need to be sorted, otherwise the vector of counts might be random
    :param seq_p: a list of lists, each sublist is a list of string containing all found symbols in the alphabet
    :return: a sorted list as a set of the alphabet, containing all found symbols in the correct order
    """
    return sorted(list(set([item for sublist in seq_p for item in sublist])))


def create_two_mers_in_order(tokenizer: tf.keras.preprocessing.text.Tokenizer):
    """
    uses a tokenizer to create all possible two-mer pairs for regression
    :tokenizer: a tensorflow tokenizer containing a word index for sequence symbols
    :return: a list of possible two-mer combinations
    """
    w_index = list(tokenizer.word_index.keys())

    self_tuple = [(x, x) for x in w_index]

    # get all start symbols
    start_symbols = list(filter(lambda x: x[0] == '#', w_index))

    # get all end symbold
    end_symbols = list(filter(lambda x: x[-1] == '#', w_index))

    combs = list(combinations(w_index, 2))
    back_comb = [(y, x) for x, y in combs]

    combs = combs + back_comb + self_tuple

    combs = list(filter(lambda x: x[0] not in end_symbols, combs))
    combs = list(filter(lambda x: x[1] not in start_symbols, combs))
    two_mers = [x[0] + x[1] for x in combs]

    fina_list = []
    for two_mer in two_mers:
        if two_mer.count('#') < 2:
            fina_list.append(two_mer)

    fina_list.sort()
    return fina_list


def get_counter_dict(ordered_tokens: list):
    """
    creates a dictionary for k-mer counts
    :param ordered_tokens: a list of tokens in order that can appear in a sequence
    :return: a dict of the form (token, 0) that can be used to create counts for a given sequence
    """
    return dict([(x, 0) for x in ordered_tokens])


def get_counts_in_order(seq_as_tokens: list, tokens_in_order: list):
    """
    get a dense, sorted vector of token counts ready for prediction
    :param seq_as_tokens: a sequence as a list of symbols
    :param tokens_in_order: a sorted list of all possible tokens
    :return: a dense vector with the appearance count of symbols
    """
    # create empty counter dict
    tmp_dict = get_counter_dict(tokens_in_order)
    # go over symbols in current sequence
    for s in seq_as_tokens:
        # increment the counter for given symbold
        tmp_dict[s] += 1
    # create a sorted count vector
    counts = np.array([tmp_dict[x] for x in tokens_in_order])
    # append to result

    return np.array(counts).astype(np.float32)


def get_two_mer_counts_in_order(seq_as_tokens: list, tokens_in_order: list):
    """
    get a dense, sorted vector of token counts
    :param seq_as_tokens: a sequence as a list of symbols
    :param tokens_in_order: a sorted list of all possible tokens
    :return: a dense vector with the appearance count of symbols
    """
    tmp_dict = get_counter_dict(tokens_in_order)
    # go over symbols in current sequence
    for t in seq_as_tokens:
        # increment the counter for given symbold
        tmp_dict[t] += 1
    # create a sorted count vector
    counts = np.array([tmp_dict[x] for x in tokens_in_order])

    return np.array(counts).astype(np.float32)


def reduced_mobility_to_ccs(one_over_k0, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    convert reduced ion mobility (1/k0) to CCS
    :param one_over_k0: reduced ion mobility
    :param charge: charge state of the ion
    :param mz: mass-over-charge of the ion
    :param mass_gas: mass of drift gas
    :param temp: temperature of the drift gas in C°
    :param t_diff: factor to translate from C° to K
    """
    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return (SUMMARY_CONSTANT * charge) / (np.sqrt(reduced_mass * (temp + t_diff)) * 1/one_over_k0)


def preprocess_peaks_sequence(s):
    """
    :param s:
    """

    seq = s

    is_acc = False

    if seq.find('(+42.01)') != -1:
        is_acc = True

    # C-<CM>
    seq = seq.replace('(+57.02)', '')
    # Phosphorylation
    seq = seq.replace('(+79.97)', '$')
    # Oxidation
    seq = seq.replace('(+15.99)', '&')
    # Acetylation
    seq = seq.replace('(+42.01)', '')
    # c-cy
    seq = seq.replace('C(+119.00)', '!')

    # form list from string
    slist = list(seq)

    slist = [s if s != '$' else '<PH>' for s in slist]
    slist = [s if s != '&' else '<OX>' for s in slist]
    slist = [s if s != '!' else 'C-<CY>' for s in slist]

    r_list = []

    for i, char in enumerate(slist):

        if char == '<PH>':
            C = slist[i - 1]
            C = C + '-<PH>'
            r_list = r_list[:-1]
            r_list.append(C)

        elif char == '<OX>':
            M = slist[i - 1]
            M = M + '-<OX>'
            r_list = r_list[:-1]
            r_list.append(M)

        elif char == 'C':
            r_list.append('C-<CM>')

        else:
            r_list.append(char)

    if is_acc:
        return ['<START>-<AC>'] + r_list + ['<END>']

    return ['<START>'] + r_list + ['<END>']


def preprocess_diann_sequence(s):
    """
    :param s:
    """

    seq = s
    seq = seq.replace('(UniMod:4)', '')
    seq = seq.replace('(UniMod:21)', '$')

    # form list from string
    slist = list(seq)

    slist = [s if s != '$' else '<PH>' for s in slist]

    r_list = []

    for i, char in enumerate(slist):

        if char == '<PH>':
            C = slist[i - 1]
            C = C + '-<PH>'
            r_list = r_list[:-1]
            r_list.append(C)

        elif char == 'C':
            r_list.append('C-<CM>')

        else:
            r_list.append(char)

    return ['<START>'] + r_list + ['<END>']


def preprocess_max_quant_sequence(s, old_annotation=False):
    """
    :param s:
    :param old_annotation:
    """

    seq = s[1:-1]

    is_acc = False

    if old_annotation:
        seq = seq.replace('(ox)', '$')

        if seq.find('(ac)') != -1:
            is_acc = True
            seq = seq.replace('(ac)', '')

    else:
        seq = seq.replace('(Oxidation (M))', '$')

    # form list from string
    slist = list(seq)

    slist = [s if s != '$' else '<OX>' for s in slist]

    r_list = []

    for i, char in enumerate(slist):

        if char == '<OX>':
            C = slist[i - 1]
            C = C + '-<OX>'
            r_list = r_list[:-1]
            r_list.append(C)

        elif char == 'C':
            r_list.append('C-<CM>')

        else:
            r_list.append(char)


    if is_acc:
        return ['<START>-<AC>'] + r_list + ['<END>']

    return ['<START>'] + r_list + ['<END>']


def sequence_with_charge(seqs_tokenized, charges):
    """
    Args:
        seqs_tokenized:
        charges:
    Returns:
    """
    s_w_c = []
    for (s, c) in list(zip(seqs_tokenized, charges)):
        s_w_c.append([str(c)] + s)

    return s_w_c
