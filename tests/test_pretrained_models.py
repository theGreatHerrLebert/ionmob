import pkg_resources

import pandas as pd
import tensorflow as tf

from ionmob.preprocess.data import to_tf_dataset_inference
from ionmob.utilities.tokenization import tokenizer_from_json


def test_model_load():
    model = tf.keras.models.load_model('ionmob/pretrained_models/GRUPredictor/')
    assert model is not None


def test_model_prediction():
    data = pd.read_parquet('ionmob/example_data/Zepeda_thunder_unique_unimod.parquet').sample(frac=.1)
    tokenizer = tokenizer_from_json('ionmob/pretrained_models/tokenizers/tokenizer.json')
    tf_ds = to_tf_dataset_inference(data.mz, data.charge, [list(s) for s in data['sequence-tokenized']], tokenizer)
    model = tf.keras.models.load_model('ionmob/pretrained_models/GRUPredictor/')

    ccs, _ = model.predict(tf_ds, verbose=False)

    assert len(ccs) == len(data)
