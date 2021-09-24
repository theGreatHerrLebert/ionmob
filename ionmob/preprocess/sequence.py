from collections import Counter
import re
import string
from itertools import islice, product
from typing import Iterator
import functools


sequence = '_AADM(Oxidation (M))Z(Oxidation (Z))VIEAVFEDLSLK_'


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
    return "(" + modification_pattern + "|" + "|".join(letters)  + ")" 


token_pattern_MaxQuant_v1dot8 = re.compile(get_token_pattern_str(
    # modification_pattern="[A|C|D|E|F|G|H|I|K|L|M|N|P|Q|R|S|T|W|Y][(][a-zA-Z]+ [(][A|C|D|E|F|G|H|I|K|L|M|N|P|Q|R|S|T|W|Y][)][)]",
    modification_pattern="[A-Z][(][a-zA-Z]+ [(][A-Z][)][)]",
    letters=string.ascii_uppercase
))

token_pattern_MaxQuant_v1dot7 = re.compile(get_token_pattern_str(
    # modification_pattern="[A|C|D|E|F|G|H|I|K|L|M|N|P|Q|R|S|T|W|Y][(][a-zA-Z]+[)]",
    modification_pattern="[A-Z][(][a-zA-Z]+ [(][A-Z][)][)]",
    letters=string.ascii_uppercase  
))

token_pattern = token_pattern_MaxQuant_v1dot8


def tokenize(token_pattern: re.Pattern, sequence:str) -> Iterable[str]:
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
        first_prefix: str="@",
        first_suffix: str="",
        last_prefix: str="#",
        last_suffix: str=""
    )-> Iterator[str]:
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


def merize(degree: int=2, separator: str="") -> Iterator[str]:
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


tokenize_tag_first_and_last = tag_first_and_last()(tokenize)
list(tokenize_tag_first_and_last(token_pattern, sequence))
Counter(tokenize_tag_first_and_last(token_pattern, sequence))

tokenize_tag_first_and_last = tag_first_and_last(first_prefix="!")(tokenize)
list(tokenize_tag_first_and_last(token_pattern, sequence))
Counter(tokenize_tag_first_and_last(token_pattern, sequence))

tokenize_1_mers = merize(degree=1)(tokenize)
list(tokenize_1_mers(token_pattern, sequence))
Counter(tokenize_1_mers(token_pattern, sequence))

tokenize_2_mers = merize(degree=2)(tokenize)
list(tokenize_2_mers(token_pattern, sequence))
Counter(tokenize_2_mers(token_pattern, sequence))

tokenize_tag_first_and_last = tag_first_and_last(first_prefix="!")(tokenize)
list(tokenize_tag_first_and_last(token_pattern, sequence))

tokenize_2_mers = merize(degree=2)(tag_first_and_last()(tokenize))
list(tokenize_2_mers(token_pattern, sequence))
Counter(tokenize_2_mers(token_pattern, sequence))
