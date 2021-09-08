import numpy as np
import tensorflow as tf
from ionmob.preprocess.helpers import tokenizer_from_json, sequence_to_tokens, get_helix_score, get_gravy_score


def get_tf_dataset(mz, charge, sequence, ccs, tokenizer_path=''):
    """
    takes data and puts them into a tensorflow dataset for easy tf interop
    :param mz: arrays of mz values
    :param charge: arrays of one hot encoded charge state values between 1 and 4 (one hot 0 to 3)
    :param sequence: array of sequences as strings
    :param ccs: array of ccs values
    :param tokenizer_path: path to a pre-fit tokenizer
    :return: a tensorflow dataset for prediction
    """
    if ccs is not None:

        masses, charges_one_hot, seq_padded, helix_score, gravy_score, ccs = get_training_data(mz, charge, sequence,
                                                                                               ccs, tokenizer_path)
        return tf.data.Dataset.from_tensor_slices(((masses, charges_one_hot, seq_padded,
                                                    helix_score, gravy_score), ccs))

    else:
        masses, charges_one_hot, seq_padded, helix_score, gravy_score = get_prediction_data(mz, charge, sequence,
                                                                                            tokenizer_path)
        dummy_ccs = np.expand_dims(np.zeros(masses.shape[0]), 1)
        return tf.data.Dataset.from_tensor_slices(((masses, charges_one_hot, seq_padded,
                                                    helix_score, gravy_score), dummy_ccs))


def get_prediction_data(mz, charge, sequence, tokenizer_path):
    """
    takes data for prediction and preprocesses it
    :param mz: arrays of mz values
    :param charge: arrays of one hot encoded charge state values between 1 and 4 (one hot 0 to 3)
    :param sequence: array of sequences as strings
    :param tokenizer_path: path to a prefittet tokenizer
    :return: a tensorflow dataset for prediction
    """
    # prepare masses
    masses = np.expand_dims(mz, 1)
    # prepare charges
    charges_one_hot = tf.one_hot(charge - 1, 4)

    # prepare sequences
    json_string = tokenizer_from_json(tokenizer_path)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
    seq_tokens = [sequence_to_tokens(s) for s in sequence]
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(seq_tokens), 40,
                                                               padding='post')

    # calculate meta features
    gravy_score = np.expand_dims(np.array([get_gravy_score(s) for s in sequence]), 1)
    helix_score = np.expand_dims(np.array([get_helix_score(s) for s in sequence]), 1)

    # generate dataset
    return masses, charges_one_hot, seq_padded, helix_score, gravy_score


def get_training_data(mz, charge, sequence, ccs, tokenizer_path):
    """
    takes data for training and preprocesses it
    :param mz: arrays of mz values
    :param charge: arrays of one hot encoded charge state values between 1 and 4 (one hot 0 to 3)
    :param sequence: array of sequences as strings
    :param ccs: array of ccs values
    :param tokenizer_path: path to a prefittet tokenizer
    :return: a tensorflow dataset for prediction
    """
    # ccs values
    ccs = np.expand_dims(ccs, 1)

    masses, charges_one_hot, seq_padded, helix_score, gravy_score = get_prediction_data(mz, charge, sequence,
                                                                                        tokenizer_path)

    # generate dataset
    return masses, charges_one_hot, seq_padded, helix_score, gravy_score, ccs
