import numpy as np
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
    Model architecture is inspired by Meier et al.: https://doi.org/10.1038/s41467-021-21352-8
    """
    def __init__(self, slopes, intercepts, number_tokens):
        super(DeepRecurrentModel, self).__init__()

        self.linear = ProjectToInitialCCS(slopes, intercepts)

        self.emb = tf.keras.layers.Embedding(number_tokens + 1, 128)
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
        x_recurrent = self.gru2(self.gru1(self.emb(tf.keras.layers.Concatenate()([seq, tf.cast(charge, tf.int32)]))))
        # concat to feed to dense layers
        concat = tf.keras.layers.Concatenate()([charge, x_recurrent, helix, gravy])
        # regularize
        d1 = self.dropout(self.dense1(concat))
        d2 = self.dense2(d1)
        # combine simple linear hypotheses with deep part
        return self.linear([mz, charge]) + self.out(d2), self.out(d2)


if __name__ == '__main__':

    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_output_1_loss',
        patience=10
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='training/rnn/checkpoint',
        monitor='val_output_1_loss',
        save_best_only=True,
        mode='min'
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        filename='training/rnn/training.csv',
        separator=',',
        append=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_output_1_loss',
        factor=1e-1,
        patience=5,
        monde='auto',
        min_delta=1e-5,
        cooldown=0,
        min_lr=1e-7
    )

    cbs = [early_stopper, checkpoint, csv_logger, reduce_lr]

    model = DeepRecurrentModel(np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                               np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32), 83)

    model.build([(None, 1), (None, 4), (None, 83,), (None, 1), (None, 1)])

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), loss_weights=[1., 0.0],
                  optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['mae'])

    print(model.summary())