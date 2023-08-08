import numpy as np
import tensorflow as tf
from ionmob.utilities.utility import sequence_to_tokens, get_helix_score, get_gravy_score, sequence_with_charge
from ionmob.utilities.tokenization import token_pattern_MaxQuant_v1, create_nmer_counts


def twomer_model_dataset(vocab, tok_func, mz, charge, sequence, ccs=None, pattern=token_pattern_MaxQuant_v1):
    """
    :param vocab: a dictionary enumerating all possible tokens: index -> token
    :param tok_func: a function that turns a sequence into a token count
    :param mz: array of mz values
    :param charge: array of charge state values
    :param sequence: array of sequences as strings
    :param ccs: if not none, returned dataset will also contain target ccs values
    :param pattern: a modification pattern to also identify PTMs
    :return: a tensorflow dataset containing ((mz, charge_one_hot, token_count_vector), ccs?) 
    """
    
    counts = create_nmer_counts(vocab, pattern, sequence, tok_func)
    c_oh = tf.one_hot(charge - 1, 4)
    m = np.array(np.expand_dims(mz, 1), dtype=np.float32)
    
    if ccs is not None:
        return tf.data.Dataset.from_tensor_slices(((m, c_oh, counts), ccs))

    ccs = np.zeros(m.shape[0])
    return tf.data.Dataset.from_tensor_slices(((m, c_oh, counts), ccs))


def sqrt_model_dataset(mz, charge, ccs=None):
    """
    :param mz: array of mz values
    :param charge: array of charges
    :param ccs: if not none, will also add ccs values to dataset
    :param bs: batch size of returned tensorflow dataset
    :return: a tensorflow dataset ready to be predicted with a SqrtModel
    """
    
    c_oh = tf.one_hot(charge - 1, 4)
    m = np.array(np.expand_dims(mz, 1), dtype=np.float32)
    
    if ccs is not None:
        return tf.data.Dataset.from_tensor_slices(((m, c_oh), ccs))

    ccs = np.zeros(m.shape[0])
    return tf.data.Dataset.from_tensor_slices(((m, c_oh), ccs))
  

def get_tf_dataset(mz: np.ndarray, charge: np.ndarray, sequence: np.ndarray, ccs: np.ndarray,
                   tokenizer: tf.keras.preprocessing.text.Tokenizer,
                   drop_sequence_ends: bool = False, add_charge=False) -> tf.data.Dataset:
    """
    takes example_data and puts them into a tensorflow dataset for easy tf interop
    :param mz: arrays of mz values
    :param charge: arrays of one hot encoded charge state values between 1 and 4 (one hot 0 to 3)
    :param sequence: array of sequences as strings
    :param ccs: array of ccs values
    :param tokenizer: pre-fit tokenizer
    :param drop_sequence_ends: if true, start end end AAs will not be treated as separate tokens
    :return: a tensorflow dataset for prediction
    """
    if ccs is not None:

        masses, charges_one_hot, seq_padded, helix_score, gravy_score, ccs = get_training_data(mz, charge, sequence,
                                                                                               ccs, tokenizer,
                                                                                               drop_sequence_ends,
                                                                                               add_charge)
        return tf.data.Dataset.from_tensor_slices(((masses, charges_one_hot, seq_padded,
                                                    helix_score, gravy_score), ccs))

    else:
        masses, charges_one_hot, seq_padded, helix_score, gravy_score = get_prediction_data(mz, charge, sequence,
                                                                                            tokenizer,
                                                                                            drop_sequence_ends,
                                                                                            add_charge)
        dummy_ccs = np.expand_dims(np.zeros(masses.shape[0]), 1)
        return tf.data.Dataset.from_tensor_slices(((masses, charges_one_hot, seq_padded,
                                                    helix_score, gravy_score), dummy_ccs))


def get_prediction_data(mz: np.ndarray, charge: np.ndarray, sequence: np.ndarray,
                        tokenizer: tf.keras.preprocessing.text.Tokenizer, drop_sequence_ends: bool, add_charge=False):
    """
    takes example_data for prediction and preprocesses it
    :param mz: arrays of mz values
    :param charge: arrays of one hot encoded charge state values between 1 and 4 (one hot 0 to 3)
    :param sequence: array of sequences as strings
    :param tokenizer: pre-fit tokenizer
    :param drop_sequence_ends: if true, start end end AAs will not be treated as separate tokens
    :param add_charge:
    :return: a tensorflow dataset for prediction
    """
    # prepare masses
    masses = np.expand_dims(mz, 1)
    # prepare charges
    charges_one_hot = tf.one_hot(charge - 1, 4)

    seq_tokens = [sequence_to_tokens(s, drop_sequence_ends) for s in sequence]
    if add_charge:
        seq_w_c = sequence_with_charge(tokenizer.texts_to_sequences(seq_tokens), charge)
        seq_padded = tf.keras.preprocessing.sequence.pad_sequences(seq_w_c, 50, padding='post')
    else:
        seq_padded = tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(seq_tokens), 50,
                                                                   padding='post')

    # calculate meta features
    gravy_score = np.expand_dims(np.array([get_gravy_score(s) for s in sequence]), 1)
    helix_score = np.expand_dims(np.array([get_helix_score(s) for s in sequence]), 1)

    # generate dataset
    return masses, charges_one_hot, seq_padded, helix_score, gravy_score


def get_training_data(mz, charge, sequence, ccs, tokenizer, drop_sequence_ends, add_charge):
    """
    takes example_data for training and preprocesses it
    :param mz: arrays of mz values
    :param charge: arrays of one hot encoded charge state values between 1 and 4 (one hot 0 to 3)
    :param sequence: array of sequences as strings
    :param ccs: array of ccs values
    :param tokenizer: pre-fit tokenizer
    :param drop_sequence_ends: if true, start end end AAs will not be treated as separate tokens
    :param add_charge:
    :return: a tensorflow dataset for prediction
    """
    # ccs values
    ccs = np.expand_dims(ccs, 1)

    masses, charges_one_hot, seq_padded, helix_score, gravy_score = get_prediction_data(mz, charge, sequence,
                                                                                        tokenizer, drop_sequence_ends,
                                                                                        add_charge)

    # generate dataset
    return masses, charges_one_hot, seq_padded, helix_score, gravy_score, ccs


def partition_tf_dataset(ds: tf.data.Dataset, ds_size: int, train_frac: float = 0.8, val_frac: float = 0.1,
                         test_frac: float = 0.1, shuffle: bool = True,
                         shuffle_size: int = int(1e7)) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    """
    partitions a tensorflow dataset into fractions for training, validation and test
    :param ds: a unbatched tensorflow dataset
    :param ds_size: number of samples inside the example_data set
    :param train_frac: fraction of example_data that should be used for training
    :param val_frac: --""-- validation
    :param test_frac: --""-- testing
    :param shuffle: if true, dataset will be shuffled before splitting
    :param shuffle_size: buffer size of shuffle, should be greater then number of examples
    :return: split of dataset into three non-overlapping subsets for training, validation and test
    """
    assert (train_frac + val_frac + test_frac) == 1
    assert (ds_size <= shuffle_size)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=41)

    train_size = int(train_frac * ds_size)
    val_size = int(val_frac * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def split_dataset(data, train_frac=80, valid_frac=90):
    num_rows = data.shape[0]
    train_index = int((num_rows / 100) * 80)
    valid_index = int((num_rows / 100) * 90)

    d_train = data.iloc[:train_index]
    d_valid = data.iloc[train_index:valid_index]
    d_test = data.iloc[valid_index:]

    return d_train, d_valid, d_test


def to_tf_dataset(mz: np.ndarray, charge: np.ndarray, sequences: np.ndarray, ccs: np.ndarray,
                  tokenizer: tf.keras.preprocessing.text.Tokenizer, batch=True, batch_size=2048):
    """
    Args:
        mz:
        charge:
        sequences:
        ccs:
        tokenizer:
        batch:
        batch_size:
    Returns:
    """
    # prepare masses, charges, sequences
    masses = np.expand_dims(mz, 1)
    charges_one_hot = tf.one_hot(charge - 1, 4)
    sequences = tokenizer.texts_to_sequences(sequences)
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, 50, padding='post')

    # prepare ccs
    ccs = np.expand_dims(ccs, 1)

    # generate dataset
    ds = tf.data.Dataset.from_tensor_slices(((masses, charges_one_hot, seq_padded), ccs))

    if batch:
        return ds.batch(batch_size)
    return ds


def to_tf_dataset_inference(mz: np.ndarray, charge: np.ndarray, sequences: np.ndarray,
                            tokenizer: tf.keras.preprocessing.text.Tokenizer, batch=True, batch_size=2048):
    """
    Args:
        mz:
        charge:
        sequences:
        tokenizer:
        batch:
        batch_size:

    Returns:

    """
    # prepare masses, charges, sequences
    masses = np.expand_dims(mz, 1)
    charges_one_hot = tf.one_hot(charge - 1, 4)
    sequences = tokenizer.texts_to_sequences(sequences)
    seq_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, 50, padding='post')

    # generate dataset
    ds = tf.data.Dataset.from_tensor_slices(((masses, charges_one_hot, seq_padded), np.zeros_like(masses)))

    if batch:
        return ds.batch(batch_size)
    return ds


def to_tf_dataset_kmer(mz: np.ndarray, charge: np.ndarray, token_counts: np.ndarray, ccs: np.ndarray,
                       batch=True, batch_size=2048):
    """

    Args:
        mz:
        charge:
        token_counts:
        ccs:
        batch:
        batch_size:

    Returns:

    """
    # prepare masses, charges, sequences
    masses = np.expand_dims(mz, 1)
    charges_one_hot = tf.one_hot(charge - 1, 4)

    # prepare ccs
    ccs = np.expand_dims(ccs, 1)

    token_counts = np.array(list(x for x in token_counts))

    # generate dataset
    ds = tf.data.Dataset.from_tensor_slices(((masses, charges_one_hot, token_counts), ccs))

    if batch:
        return ds.batch(batch_size)

    return ds
