from collections import Counter
import re
import string
from itertools import islice, product


sequence = '_AADM(Oxidation (M))Z(Oxidation (Z))VIEAVFEDLSLK_'

def mer_pattern_str_maker(
        degree=1,
        tokens=list(string.ascii_uppercase) + ["[A-Z][(][a-zA-Z]+ [(][A-Z][)][)]"]
    ):
    "|".join("".join(letter_tuple) for letter_tuple in product(tokens , repeat=2))


def make_token_pattern(
        LETTERS = string.ascii_uppercase,
        modification_pattern_str = "[A-Z][(][a-zA-Z]+ [(][A-Z][)][)]"
    ):
    token_pattern_str = "(" + modification_pattern_str + "|" + "|".join(LETTERS)  + ")" 
    return re.compile(token_pattern_str)


token_pattern = make_token_pattern()



def iter_tokens(token_pattern, sequence):
    for x in re.finditer(token_pattern, sequence):
        yield x.group()


def get_token(token_pattern, sequence):
    return list(iter_tokens(token_pattern, sequence))


def get_token_begins_ends(token_pattern, sequence):
    res = list(iter_tokens(token_pattern, sequence))
    res[0] = "@" + res[0]
    res[-1] = "#" + res[-1]
    return res 


def get_token_counts(token_pattern, sequence):
    return Counter(iter_tokens(token_pattern, sequence))


def Counter_begins_ends(tokenator):
    first = next(tokenator)
    cnt = Counter()
    cnt[f"@{first}"] += 1
    for token in tokenator:
        cnt[token] += 1
    last = token
    if cnt[last] == 1:
        cnt.pop(last)
    else:
        cnt[last] -= 1
    cnt[f"#{last}"] += 1
    return cnt


def iter_complicated_mers(tokenator, separator="", degree=1):
    prev_lst = list(islice(tokenator, degree))
    yield separator.join(prev_lst)
    for next_token in tokenator:
        prev_lst.pop(0)
        prev_lst.append(next_token) 
        yield separator.join(prev_lst)


Counter(iter_complicated_mers(iter_tokens(token_pattern, sequence), degree=1))
Counter(iter_complicated_mers(iter_tokens(token_pattern, sequence), degree=2))
Counter_begins_ends(iter_complicated_mers(iter_tokens(token_pattern, sequence), degree=2))
Counter_begins_ends(iter_complicated_mers(iter_tokens(token_pattern, sequence), degree=3))


def get_mers(token_patterns, sequence, degree=1):
    if degree == 1:
        return get_token_counts_begins_ends(token_pattern, sequence)
    else:
        return get_complicated_mers(degree)

def get_token_counts(
        sequence,
        token_pattern,
        # compiled_modification_pattern,
        # compiled_dropable_pattern,
    ):
    pass