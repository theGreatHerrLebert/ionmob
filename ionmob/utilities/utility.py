import re
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from numpy import ndarray


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

    tmp_table = table.copy(deep=True)
    tmp_reference = reference.copy(deep=True)

    tmp_table['sequence'] = table.apply(lambda r: ''.join(list(r['sequence-tokenized'])), axis=1)
    tmp_reference['sequence'] = reference.apply(lambda r: ''.join(list(r['sequence-tokenized'])), axis=1)

    reference_tmp = tmp_reference[tmp_reference.charge == use_charge_state]
    table_tmp = tmp_table[tmp_table.charge == use_charge_state]

    both = pd.merge(left=reference_tmp, right=table_tmp, right_on=['sequence', 'charge'],
                    left_on=['sequence', 'charge'])

    return np.mean(both.ccs_x - both.ccs_y)


def get_non_overlapping_pairs(ds_ref, ds_test):
    """
    reduce a dataframe to only contain seq, charge pairs not present in ref dataset
    :ds_ref: dataframe containing all charge, seq pairs that should be excluded from other frame
    :ds_test: dataframe that should be reduced to only contain seq, charge pairs not in reference
    :return: dataframe only containing seq, charge pairs not in ref example_data
    """
    ref_pairs = set(zip(ds_ref.sequence, ds_ref.charge))
    test_pairs = set(zip(ds_test.sequence, ds_test.charge))
    candidates = test_pairs - ref_pairs

    row_list = []
    for _, row in ds_test.iterrows():
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


def get_gravy_score(seq: List[str], drop_ends: bool = True, normalize: bool = True):
    """
    calculate normalized gravy scores for a given sequence
    :param seq: peptide sequence
    :param drop_ends: if true, first and last character will be stripped from sequence
    :param normalize:
    :return: gravy score normalized by sequence length
    """
    if drop_ends:
        seq = seq[1:-1]

    sanitized_sequence = []

    for amino_acid in seq:
        if len(amino_acid) > 1:
            sanitized_sequence.append(amino_acid[0])
        else:
            sanitized_sequence.append(amino_acid)

    seq = ''.join(sanitized_sequence)

    if normalize:
        return ProteinAnalysis(seq).gravy() / len(seq)

    return ProteinAnalysis(seq).gravy()


def get_helix_score(seq: List[str], drop_ends: bool = True):
    """
    calculate portion of helix peptides
    :param seq: sequence to calculate helix portion of
    :param drop_ends: if true, first and last character will be stripped from sequence
    :return: helix portion
    """
    if drop_ends:
        seq = seq[1:-1]

    sanitized_sequence = []

    for aa in seq:
        if len(aa) > 1:
            sanitized_sequence.append(aa[0])
        else:
            sanitized_sequence.append(aa)

    seq = ''.join(sanitized_sequence)

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
        # increment the counter for given symbols
        tmp_dict[t] += 1
    # create a sorted count vector
    counts = np.array([tmp_dict[x] for x in tokens_in_order])

    return np.array(counts).astype(np.float32)


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

    slist = [s if s != '$' else '[UNIMOD:21]' for s in slist]
    slist = [s if s != '&' else '[UNIMOD:35]' for s in slist]
    slist = [s if s != '!' else 'C[UNIMOD:312]' for s in slist]

    r_list = []

    for i, char in enumerate(slist):

        if char == '[UNIMOD:21]':
            C = slist[i - 1]
            C = C + '[UNIMOD:21]'
            r_list = r_list[:-1]
            r_list.append(C)

        elif char == '[UNIMOD:35]':
            M = slist[i - 1]
            M = M + '[UNIMOD:35]'
            r_list = r_list[:-1]
            r_list.append(M)

        elif char == 'C':
            r_list.append('C[UNIMOD:4]')

        else:
            r_list.append(char)

    if is_acc:
        return ['<START>[UNIMOD:1]'] + r_list + ['<END>']

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

    slist = [s if s != '$' else '[UNIMOD:21]' for s in slist]

    r_list = []

    for i, char in enumerate(slist):

        if char == '[UNIMOD:21]':
            C = slist[i - 1]
            C = C + '[UNIMOD:21]'
            r_list = r_list[:-1]
            r_list.append(C)

        elif char == 'C':
            r_list.append('C[UNIMOD:4]')

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
        seq = seq.replace('(Phospho (STY))', '&')

        if seq.find('(Acetyl (Protein N-term))') != -1:
            is_acc = True
            seq = seq.replace('(Acetyl (Protein N-term))', '')

    # form list from string
    slist = list(seq)

    tmp_list = []

    for item in slist:
        if item == '$':
            tmp_list.append('[UNIMOD:35]')

        elif item == '&':
            tmp_list.append('[UNIMOD:21]')

        else:
            tmp_list.append(item)

    slist = tmp_list

    r_list = []

    for i, char in enumerate(slist):

        if char == '[UNIMOD:35]':
            C = slist[i - 1]
            C = C + '[UNIMOD:35]'
            r_list = r_list[:-1]
            r_list.append(C)

        elif char == 'C':
            r_list.append('C[UNIMOD:4]')

        elif char == '[UNIMOD:21]':
            C = slist[i - 1]
            C = C + '[UNIMOD:21]'
            r_list = r_list[:-1]
            r_list.append(C)

        else:
            r_list.append(char)

    if is_acc:
        return ['<START>[UNIMOD:1]'] + r_list + ['<END>']

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


def split_dataset(data: pd.DataFrame, train_frac=80, valid_frac=90):
    num_rows = data.shape[0]
    train_index = int((num_rows / 100) * train_frac)
    valid_index = int((num_rows / 100) * valid_frac)

    d_train = data.iloc[:train_index]
    d_valid = data.iloc[train_index:valid_index]
    d_test = data.iloc[valid_index:]

    return d_train, d_valid, d_test


def preprocess_max_quant_evidence(exp: pd.DataFrame) -> pd.DataFrame:
    """
    select columns from evidence txt, rename to ionmob naming convention and transform to raw example_data rt in seconds
    Args:
        exp: a MaxQuant evidence dataframe from evidence.txt table

    Returns: cleaned evidence dataframe, columns renamed to ionmob naming convention

    """

    # select columns
    exp = exp[['m/z', 'Mass', 'Charge', 'Modified sequence', 'Retention time',
               'Retention length', 'Ion mobility index', 'Ion mobility length', '1/K0', '1/K0 length',
               'Number of isotopic peaks', 'Max intensity m/z 0', 'Intensity', 'Raw file', 'CCS']].rename(
        # rename columns to ionmob naming convention
        columns={'m/z': 'mz', 'Mass': 'mass',
                 'Charge': 'charge', 'Modified sequence': 'sequence', 'Retention time': 'rt',
                 'Retention length': 'rt_length', 'Ion mobility index': 'im', 'Ion mobility length': 'im_length',
                 '1/K0': 'inv_ion_mob', '1/K0 length': 'inv_ion_mob_length', 'CCS': 'ccs',
                 'Number of isotopic peaks': 'num_peaks', 'Max intensity m/z 0': 'mz_max_intensity',
                 'Intensity': 'intensity', 'Raw file': 'raw'}).dropna()

    # transform retention time from minutes to seconds as stored in tdf raw example_data
    exp['rt'] = exp.apply(lambda r: r['rt'] * 60, axis=1)
    exp['rt_length'] = exp.apply(lambda r: r['rt_length'] * 60, axis=1)
    exp['rt_start'] = exp.apply(lambda r: r['rt'] - r['rt_length'] / 2, axis=1)
    exp['rt_stop'] = exp.apply(lambda r: r['rt'] + r['rt_length'] / 2, axis=1)

    exp['im_start'] = exp.apply(lambda r: int(np.round(r['im'] - r['im_length'] / 2)), axis=1)
    exp['im_stop'] = exp.apply(lambda r: int(np.round(r['im'] + r['im_length'] / 2)), axis=1)

    exp['inv_ion_mob_start'] = exp.apply(lambda r: r['inv_ion_mob'] - r['inv_ion_mob_length'] / 2, axis=1)
    exp['inv_ion_mob_stop'] = exp.apply(lambda r: r['inv_ion_mob'] + r['inv_ion_mob_length'] / 2, axis=1)

    # remove duplicates
    exp = exp.drop_duplicates(['sequence', 'charge', 'rt', 'ccs'])

    return exp


def percent_difference(ccs_x, ccs_y):
    """
    calculate percent difference between two ccs values
    Args:
        ccs_x: first ccs value
        ccs_y: second ccs value

    Returns: percent difference between two ccs values

    """
    return np.round((np.abs(ccs_x - ccs_y) / ccs_x) * 100, 2)


def old_sequence_to_pro_forma(sequence: List[str]) -> List[str]:
    """Translates a peptide sequence given as a list of tokens into a string ProForma representation.

    Args:
        sequence (list[str]): Sequence as list of tokens.
        padd_ends (bool): if True, will add '_' to start and end of sequence indicating a peptide

    Returns:
        str: Sequence now formatted according to ProForma convention

    .. ProForma Format:
        https://github.com/HUPO-PSI/ProForma
    .. Unimod Homepage:
        https://www.unimod.org/

    """
    # Acetylation=(UniMod:1)
    # Carbomethylation=(UniMod:4)
    # Phosphorylation=(UniMod:21)
    # Oxidation=(UniMod:35)
    # Cysteinylation=(UniMod:312)

    TRANSLATION_DICT = {
        '<START>': '<START>', '<END>': '<END>', '<START>-<AC>': '<START>[UNIMOD:1]', 'C-<CM>': 'C[UNIMOD:4]',
        'S-<PH>': 'S[UNIMOD:21]', 'T-<PH>': 'T[UNIMOD:21]', 'Y-<PH>': 'Y[UNIMOD:21]',
        'M-<OX>': 'M[UNIMOD:35]', 'C-<CY>': 'C[UNIMOD:312]', 'K-<AC>': 'K[UNIMOD:1]',
    }

    TRANSLATION_DICT.update({c: c for c in 'ACDEFGHIKLMNPQRSTVWY'})

    return [TRANSLATION_DICT[char] for char in sequence]
