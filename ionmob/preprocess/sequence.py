from collections import Counter
import re
import string
from itertools import islice, product

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


def iter_tokens(token_pattern, sequence):
    for x in re.finditer(token_pattern, sequence):
        yield x.group()

def iter_strings_modifying_first_and_last(iterator, first_prefix="@", first_suffix="", last_prefix="#", last_suffix=""):
    prev_ = f"{first_prefix}{next(iterator)}{first_suffix}"
    for next_ in iterator:
        yield prev_
        prev_ = next_
    yield f"{last_prefix}{prev_}{last_suffix}" 

def iter_complicated_mers(iterator, separator="", degree=1):
    prev_lst = list(islice(iterator, degree))
    yield separator.join(prev_lst)
    for next_token in iterator:
        prev_lst.pop(0)
        prev_lst.append(next_token) 
        yield separator.join(prev_lst)



list(iter_tokens(token_pattern, sequence))
list(iter_strings_modifying_first_and_last(iter_tokens(token_pattern, sequence)))
Counter(iter_tokens(token_pattern, sequence))
Counter(iter_strings_modifying_first_and_last(iter_tokens(token_pattern, sequence)))

list(iter_complicated_mers(iter_tokens(token_pattern, sequence), degree=2))
list(iter_complicated_mers(iter_strings_modifying_first_and_last(iter_tokens(token_pattern, sequence)), degree=2))
list(iter_complicated_mers(iter_strings_modifying_first_and_last(iter_tokens(token_pattern, sequence)), degree=3))

Counter(iter_complicated_mers(iter_tokens(token_pattern, sequence), degree=2))
Counter(iter_complicated_mers(iter_strings_modifying_first_and_last(iter_tokens(token_pattern, sequence)), degree=2))
Counter(iter_complicated_mers(iter_strings_modifying_first_and_last(iter_tokens(token_pattern, sequence)), degree=3))
