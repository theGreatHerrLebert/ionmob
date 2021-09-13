import numpy as np
import tensorflow as tf
from ionmob.preprocess.helpers import sequence_to_tokens, get_helix_score, get_gravy_score


def get_tf_dataset(mz, charge, sequence, ccs, tokenizer):
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
                                                                                               ccs, tokenizer)
        return tf.data.Dataset.from_tensor_slices(((masses, charges_one_hot, seq_padded,
                                                    helix_score, gravy_score), ccs))

    else:
        masses, charges_one_hot, seq_padded, helix_score, gravy_score = get_prediction_data(mz, charge, sequence,
                                                                                            tokenizer)
        dummy_ccs = np.expand_dims(np.zeros(masses.shape[0]), 1)
        return tf.data.Dataset.from_tensor_slices(((masses, charges_one_hot, seq_padded,
                                                    helix_score, gravy_score), dummy_ccs))


def get_prediction_data(mz, charge, sequence, tokenizer):
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


def partition_tf_dataset(ds: tf.data.Dataset, ds_size: int, train_frac: float = 0.8, val_frac: float = 0.1,
                         test_frac: float = 0.1, shuffle: bool = True, shuffle_size: int = int(1e7)):
    """
    partitions a tensorflow dataset into fractions for training, validation and test
    :param ds: a unbatched tensorflow dataset
    :param ds_size: number of samples inside the data set
    :param train_frac: fraction of data that should be used for training
    :param val_frac: --""-- validation
    :param test_frac: --""-- testing
    :param shuffle: if true, dataset will be shuffled before splitting
    :param shuffle_size: buffer size of shuffle, should be greater then number of examples
    :return: split of dataset into three non-overlapping subsets for training, validation and test
    """
    assert (train_frac + val_frac + test_frac) == 1
    assert (ds_size <= shuffle_size)

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=41)

    train_size = int(train_frac * ds_size)
    val_size = int(val_frac * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds
