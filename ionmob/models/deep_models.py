import tensorflow as tf


class ProjectToInitialCCS(tf.keras.layers.Layer):
    """
    Simple linear regression model, calculates ccs value as linear mapping from mz, charge -> ccs
    """

    def __init__(self, slopes, intercepts):
        super(ProjectToInitialCCS, self).__init__()
        self.slopes = tf.constant([slopes])
        self.intercepts = tf.constant([intercepts])

    def call(self, inputs):
        mz, charge = inputs[0], inputs[1]
        # since charge is one-hot encoded, can use it to gate linear prediction by charge state
        return tf.expand_dims(tf.reduce_sum((self.slopes * mz + self.intercepts) * tf.squeeze(charge), axis=1), 1)


class DeepRecurrentModel(tf.keras.models.Model):
    """
    Deep Learning model combining initial linear fit with sequence based features, both scalar and complex
    Model architecture is (partly) inspired by Meier et al.: https://doi.org/10.1038/s41467-021-21352-8
    """
    def __init__(self, slopes, intercepts):
        super(DeepRecurrentModel, self).__init__()

        self.linear = ProjectToInitialCCS(slopes, intercepts)

        self.emb = tf.keras.layers.Embedding(84, 128)
        self.gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))
        self.gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=False, 
                                                                      recurrent_dropout=0.2))

        self.dense1 = tf.keras.layers.Dense(128, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')

        self.dropout = tf.keras.layers.Dropout(0.3)

        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        """
        :param inputs: should contain: (mz, charge_one_hot, seq_as_token_indices, helix_score, gravy_score)
        """
        # get inputs
        mz, charge, seq, helix, gravy = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        # sequence learning
        x_recurrent = self.gru2(self.gru1(self.emb(seq)))
        # concat to feed to dense layers
        concat = tf.keras.layers.Concatenate()([charge, x_recurrent, helix, gravy])
        # regularize
        d1 = self.dropout(self.dense1(concat))
        d2 = self.dense2(d1)
        # combine simple linear hypotheses with deep part
        return self.linear([mz, charge]) + self.out(d2), self.out(d2)
