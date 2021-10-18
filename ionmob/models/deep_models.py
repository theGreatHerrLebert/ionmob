import numpy as np
import tensorflow as tf


class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d):
        super(SelfAttentionLayer, self).__init__()
        self.d = d

    def build(self, input_shape):
        self.Wq = self.add_weight(
            shape=(input_shape[-1], self.d), initializer='glorot_uniform',
            trainable=True, dtype='float32')

        self.Wk = self.add_weight(
            shape=(input_shape[-1], self.d), initializer='glorot_uniform',
            trainable=True, dtype='float32')

        self.Wv = self.add_weight(
            shape=(input_shape[-1], self.d), initializer='glorot_uniform',
            trainable=True, dtype='float32')

    def call(self, q_x, k_x, v_x, mask=None):
        q = tf.matmul(q_x, self.Wq)
        k = tf.matmul(k_x, self.Wk)
        v = tf.matmul(v_x, self.Wv)
        p = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(np.float32(self.d))

        if mask is None:
            p = tf.nn.softmax(p)
        else:
            p += mask * -1e9
            p = tf.nn.softmax(p)

        h = tf.matmul(p, v)
        return h, p


class FCLayer(tf.keras.layers.Layer):
    def __init__(self, d1, d2):
        super(FCLayer, self).__init__()
        self.d1 = d1
        self.d2 = d2

    def build(self, input_shape):
        self.W1 = self.add_weight(
            shape=(input_shape[-1], self.d1), initializer='glorot_uniform',
            trainable=True, dtype='float32'
        )
        self.b1 = self.add_weight(
            shape=(self.d1,), initializer='glorot_uniform',
            trainable=True, dtype='float32'
        )
        self.W2 = self.add_weight(
            shape=(input_shape[-1], self.d2), initializer='glorot_uniform',
            trainable=True, dtype='float32'
        )
        self.b2 = self.add_weight(
            shape=(self.d2,), initializer='glorot_uniform',
            trainable=True, dtype='float32'
        )

    def call(self, x):
        ff1 = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        ff2 = tf.matmul(x, self.W2) + self.b2
        return ff2


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d, n_heads):
        super(EncoderLayer, self).__init__()
        self.d = d
        self.d_head = int(d / n_heads)
        self.n_heads = n_heads

    def build(self, input_shape):
        self.attn_heads = [SelfAttentionLayer(self.d_head) for i in range(self.n_heads)]
        self.fc_layer = FCLayer(2048, self.d)

    def compute_multihead_output(self, x):
        outputs = [head(x, x, x)[0] for head in self.attn_heads]
        outputs = tf.concat(outputs, axis=-1)
        return outputs

    def call(self, x):
        h1 = self.compute_multihead_output(x)
        y = self.fc_layer(h1)
        return h1


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d, n_heads):
        super(DecoderLayer, self).__init__()
        self.d = d
        self.d_head = int(d / n_heads)
        self.dec_attn_heads = [SelfAttentionLayer(self.d_head) for i in range(n_heads)]
        self.attn_heads = [SelfAttentionLayer(self.d_head) for i in range(n_heads)]
        self.fc_layer = FCLayer(2048, self.d)

    def call(self, de_x, en_x, mask=None):
        def compute_multihead_output(de_x, en_x, mask=None):
            outputs = [head(en_x, en_x, de_x, mask)[0] for head in self.attn_heads]
            outputs = tf.concat(outputs, axis=-1)
            return outputs

        h1 = compute_multihead_output(de_x, de_x, mask)
        h2 = compute_multihead_output(h1, en_x)
        y = self.fc_layer(h2)
        return y


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
    def __init__(self, slopes, intercepts, num_tokens, seq_len=50, gru_1=128, gru_2=64):
        super(DeepRecurrentModel, self).__init__()
        self.__seq_len = seq_len

        self.linear = ProjectToInitialCCS(slopes, intercepts)

        self.emb = tf.keras.layers.Embedding(input_dim=num_tokens + 1, output_dim=128, input_length=seq_len)
        self.gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_1, return_sequences=True))
        self.gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_2, return_sequences=False,
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


class DeepAttentionModel(tf.keras.models.Model):
    """
    Deep Learning model combining initial linear fit with sequence based features, both scalar and complex
    Model architecture is inspired by Meier et al.: https://doi.org/10.1038/s41467-021-21352-8
    """
    def __init__(self, slopes, intercepts, num_tokens, seq_len=50, attn_dim=128, gru_dim=64, n_heads=4):
        super(DeepAttentionModel, self).__init__()
        self.__seq_len = seq_len

        self.linear = ProjectToInitialCCS(slopes, intercepts)
        self.emb = tf.keras.layers.Embedding(input_dim=num_tokens + 1, output_dim=128, input_length=seq_len)

        self.attn_1 = EncoderLayer(attn_dim, n_heads)
        self.attn_2 = EncoderLayer(attn_dim, n_heads)
        self.gru = tf.keras.layers.GRU(gru_dim, return_sequences=False)

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
        x_recurrent = self.gru(self.attn_2(self.attn_1(self.emb(seq))))
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

    model = DeepAttentionModel(np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                               np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32), 83)

    model.build([(None, 1), (None, 4), (None, 50), (None, 1), (None, 1)])

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), loss_weights=[1., 0.0],
                  optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['mae'])

    print(model.summary())
