import pandas as pd
from ionmob.utilities.tokenization import tokenizer_from_json


def test_load_tokenizer():
    tokenizer = tokenizer_from_json('ionmob/pretrained_models/tokenizers/tokenizer.json')
    # assert that tokenizer is not None
    assert tokenizer is not None


def test_tokenization():
    tokenizer = tokenizer_from_json('ionmob/pretrained_models/tokenizers/tokenizer.json')
    df = pd.read_parquet('ionmob/example_data/Tenzer_unimod.parquet')
    # tokenize the sequences
    df['t'] = df.apply(lambda x: tokenizer.texts_to_sequences(x['sequence-tokenized']), axis=1)
    # assert that the tokenized sequences are not empty
    assert all([len(x) > 0 for x in df.t.values])