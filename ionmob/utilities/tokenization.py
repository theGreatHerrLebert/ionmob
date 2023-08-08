import tensorflow as tf
import io
import json
from collections import Counter
import re
import string
from itertools import islice, product
from typing import Iterator, Iterable
import functools
import numpy as np
import pandas as pd

from functools import reduce


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


def get_occurring_kmers(data):
    """
    create a set of all 2-mers in a given dataset
    :param data: dataset containing a tokenized sequence list
    :return: a set off all occurring 2-mers
    """
    pairs = list(data.apply(lambda r: list(zip(list(r['sequence-tokenized']),
                                               list(r['sequence-tokenized'][1:]))), axis=1))

    pairs_unique = list(map(lambda l: set(l), pairs))
    total_set = reduce(lambda l, r: l.union(r), pairs_unique)

    return total_set


def get_index_dict(total_set):
    """
    create a sorted, indexed dictionary tuple -> list_index
    :param total_set: a set containing all tuples to be indexed
    :return: a dictionary: tuple -> list_index, where tuples are lexicographically sorted
    """
    sorted_list = sorted(list(total_set))
    return dict(zip(sorted_list, np.arange(len(sorted_list))))


def create_count_vector(pairs, num_columns, index_dict):
    """
    helper function to count occurrences of tuples in a given tuple sequence
    :param pairs: a list of all aa tuples
    :param num_columns: length of zero vector
    :param index_dict: sorted dict: tuple -> list_index
    :return: a vector of length 1 x num_tokens, holding counts of tuples in a given sequence
    """
    count_vec = np.zeros(num_columns)

    for p in pairs:
        count_vec[index_dict[p]] += 1

    return count_vec


def create_count_vectors(data, index_dict):
    """
    creates a column of tuple counts in a given dataset
    :param data: a dataset to create tuple count vectors for
    :param index_dict: sorted dict: tuple -> list_index
    :return: a column with all count vectors for a given dataset
    """
    pairs = pd.DataFrame({'pairs': data.apply(lambda r: list(zip(list(r['sequence-tokenized']),
                                                                 list(r['sequence-tokenized'][1:]))), axis=1)})
    # num_rows = len(pairs)
    num_cols = len(index_dict)

    pairs['pair_count'] = pairs.apply(lambda r: create_count_vector(r['pairs'], num_cols, index_dict), axis=1)

    return pairs[['pair_count']]


def get_token_pattern_str(
    modification_pattern: str,
    letters: str
) -> str:
    """Get a token pattern.

    The token pattern describes what are the basic units in a string containing the sequence of (potentially) modified amino-acids.
    It extends the simple letter notation, eg AAACCCPPPAACP, to include additional descriptors of amino-acids, eg AA(ox)ACCCPP(silly)PAACP.
    Note that the modifications pattern must precede the letters, as letters might be included in the modifications.

    Arguments:
        modification_pattern (str): A string containing the pattern of modifications.
        letters (str): A string with all the one-symbol letters of the alphabet.

    Returns:
        str: the pattern string describing the rule for regex to find individual tokens in a sequence.
    """
    return "(" + modification_pattern + "|" + "|".join(letters) + ")"


token_pattern_MaxQuant_v2 = re.compile(get_token_pattern_str(
    modification_pattern="[A-Z][(][a-zA-Z]+ [(][A-Z][)][)]",
    letters=string.ascii_uppercase
))

token_pattern_MaxQuant_v1 = re.compile(get_token_pattern_str(
    modification_pattern="[A-Z][(][a-zA-Z]+[)]",
    letters=string.ascii_uppercase
))

# TODO: add more patterns for other programs, like DiaNN/Spectronaut...


def tokenize(token_pattern: re.Pattern, sequence: str) -> Iterable[str]:
    """Iterate over tokens.

    Arguments:
        token_pattern: a compiled token_pattern.
        sequence (str): a sequence to tokenize.

    Yields:
        str: A valid token.
    """
    for x in re.finditer(token_pattern, sequence):
        yield x.group()


def tag_first_and_last(
    first_prefix: str = "@",
    first_suffix: str = "",
    last_prefix: str = "#",
    last_suffix: str = ""
) -> Iterator[str]:
    def decorator(tokenize):
        @functools.wraps(tokenize)
        def wrapper(*args, **kwargs):
            iter_ = tokenize(*args, **kwargs)
            prev_ = f"{first_prefix}{next(iter_)}{first_suffix}"
            for next_ in iter_:
                yield prev_
                prev_ = next_
            yield f"{last_prefix}{prev_}{last_suffix}"
        return wrapper
    return decorator


def merize(degree: int = 2, separator: str = "") -> Iterator[str]:
    assert degree >= 2, f"merizing makes sense for degree >= 2, not {degree}."

    def decorator(tokenize):
        @functools.wraps(tokenize)
        def wrapper(*args, **kwargs):
            iter_ = tokenize(*args, **kwargs)
            prev_lst = list(islice(iter_, degree))
            yield separator.join(prev_lst)
            for next_token in iter_:
                prev_lst.pop(0)
                prev_lst.append(next_token)
                yield separator.join(prev_lst)
        return wrapper
    return decorator


def create_vocab_set(counter_list):
    """
    create a dictionary that maps from tokens to indices
    :param counter_list: a list of counters that counted unique tokens in a set of sequences
    :return: a dictonary: token -> index
    """
    ret_set = set()
    for counter in counter_list:
        ret_set = ret_set.union(set(counter))

    rs = sorted(list(ret_set))
    return dict(zip(rs, range(0, len(rs))))


def create_counter_vector(counter, index_dict):
    """

    :param counter:
    :param index_dict:
    :return: a vector of len(all_tokens) that counted occurences of each token in a given sequence
    """
    counter_vector = np.zeros(len(index_dict))

    for (k, v) in counter.items():
        counter_vector[index_dict[k]] = v

    return counter_vector


def create_indexed_vocab(token_pattern, sequences, degree=2):
    """
    ::
    ::
    """
    tokenize_tag_first_and_last = tag_first_and_last(first_prefix="!", last_prefix="", last_suffix="#")(tokenize)

    nmer_tokenize = merize(degree)(tokenize_tag_first_and_last)

    seq_counter = [Counter(nmer_tokenize(token_pattern, sequence)) for sequence in sequences]

    # get an empty dictionary with all tokens in vocabulary
    index_dict = create_vocab_set(seq_counter)

    return index_dict, nmer_tokenize


def create_nmer_counts(indexed_vocab, token_pattern, sequences, nmer_tokenize_function):
    count_vec = np.array([create_counter_vector(counter, indexed_vocab) for counter in
                          [Counter(nmer_tokenize_function(token_pattern, sequence)) for sequence in sequences]])
    return count_vec



if __name__ == "__main__":
    sequence = '_AADM(Oxidation (M))Z(Oxidation (Z))VIEAVFEDLSLK_'
    token_pattern = token_pattern_MaxQuant_v1

    tokenize_tag_first_and_last = tag_first_and_last(first_prefix="!")(tokenize)

    tokenize_tag_first_and_last = tag_first_and_last(first_prefix="!")(tokenize)

    list(tokenize_tag_first_and_last(token_pattern, sequence))
    print(Counter(tokenize_tag_first_and_last(token_pattern, sequence)))

    tokenize_tag_first_and_last = tag_first_and_last(first_prefix="!")(tokenize)
    list(tokenize_tag_first_and_last(token_pattern, sequence))
    Counter(tokenize_tag_first_and_last(token_pattern, sequence))

    tokenize_2_mers = merize(degree=2)(tokenize)
    list(tokenize_2_mers(token_pattern, sequence))
    print(Counter(tokenize_2_mers(token_pattern, sequence)))

    tokenize_2_mers = merize(degree=2)(tag_first_and_last()(tokenize))
    list(tokenize_2_mers(token_pattern, sequence))
    Counter(tokenize_2_mers(token_pattern, sequence))

    @merize(degree=2)
    @tag_first_and_last(first_prefix="!", last_suffix="$")
    def tokenize(token_pattern: re.Pattern, sequence: str) -> Iterable[str]:
        """Iterate over tokens.

        Arguments:
            token_pattern: a compiled token_pattern.
            sequence (str): a sequence to tokenize.

        Yields:
            str: A valid token.
        """
        for x in re.finditer(token_pattern, sequence):
            yield x.group()

    print(list(tokenize(token_pattern, sequence)))
