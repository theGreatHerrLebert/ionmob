import io
import json
import re

from itertools import combinations
import numpy as np
import tensorflow as tf
from Bio.SeqUtils.ProtParam import ProteinAnalysis


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


def get_gravy_score(seq: str, drop_ends: bool = True):
    """
    calculate normalized gravy scores for a given sequence
    :param seq: peptide sequence
    :param drop_ends: if true, first and last character will be stripped from sequence
    :return: gravy score normalized by sequence length
    """
    seq = seq.replace('(ac)', '')
    seq = seq.replace('(ox)', '')
    if drop_ends:
        seq = seq[1:-1]
    return ProteinAnalysis(seq).gravy() / len(seq)


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
