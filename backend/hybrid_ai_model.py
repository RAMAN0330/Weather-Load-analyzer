import tensorflow as tf
from tensorflow.keras import Model, layers as L, optimizers as O
import numpy as np

# Constants from helper.py
SEQ_LEN = 96
HORIZON = 96
DERIV_LAMBDA = 0.3

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.W1 = L.Dense(units, use_bias=False)
        self.W2 = L.Dense(units, use_bias=False)
        self.V  = L.Dense(1,    use_bias=False)

    def call(self, query, values):
        q = tf.expand_dims(query, 1)
        score = tf.nn.tanh(self.W1(values) + self.W2(q))
        weights = tf.nn.softmax(self.V(score), axis=1)
        context = tf.reduce_sum(weights * values, axis=1)
        return context, tf.squeeze(weights, -1)

class ARDecoder(tf.keras.layers.Layer):
    def __init__(self, dec_units, exg_embed_dim, attn_units, dropout=0.0,
                 exog_activation='relu', dec_activation='tanh',
                 dec_recurrent_activation='sigmoid', **kwargs):
        super().__init__(**kwargs)
        self.dec_units = int(dec_units)
        self.exg_embed_dim = int(exg_embed_dim)
        self.attn_units = int(attn_units)
        self.dropout_rate = float(dropout)

        self.attn = BahdanauAttention(attn_units)
        self.ex_emb = tf.keras.layers.Dense(exg_embed_dim, activation=exog_activation)

        self.cell = tf.keras.layers.LSTMCell(
            dec_units,
            dropout=0.0,
            recurrent_dropout=0.0,
            activation=dec_activation,
            recurrent_activation=dec_recurrent_activation,
        )
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        fut, y_teacher, enc_outs, s_h, s_c, hist_in, training_flag = inputs
        horizon = tf.shape(fut)[1]

        last_y = hist_in[:, -1, 0:1]
        last_y = tf.ensure_shape(last_y, [None, 1])

        if isinstance(training_flag, bool):
            training_flag = tf.constant(training_flag, dtype=tf.bool)
        else:
            training_flag = tf.cast(training_flag, tf.bool)
        training_gate = tf.reduce_all(training_flag)

        outputs = tf.TensorArray(tf.float32, size=horizon)
        t = tf.constant(0)

        def cond(t, *_):
            return tf.less(t, horizon)

        def body(t, last_y, s_h, s_c, outputs):
            ex_t = self.ex_emb(fut[:, t, :])
            ctx_t, _ = self.attn(s_h, enc_outs)
            step_in = tf.concat([last_y, ex_t, ctx_t], axis=-1)

            if self.dropout_rate > 0.0:
                step_in = tf.cond(
                    training_gate,
                    lambda: tf.nn.dropout(step_in, rate=self.dropout_rate),
                    lambda: step_in,
                )

            out_t, [s_h, s_c] = self.cell(step_in, states=[s_h, s_c])
            y_t = self.out(out_t)
            y_t = tf.ensure_shape(y_t, [None, 1])

            y_teacher_t = y_teacher[:, t, :]
            y_teacher_t = tf.ensure_shape(y_teacher_t, [None, 1])
            next_last_y = tf.where(training_gate, y_teacher_t, y_t)
            next_last_y = tf.ensure_shape(next_last_y, [None, 1])

            outputs = outputs.write(t, y_t)
            return t + 1, next_last_y, s_h, s_c, outputs

        _, _, _, _, outputs = tf.while_loop(
            cond,
            body,
            loop_vars=[t, last_y, s_h, s_c, outputs],
            parallel_iterations=1,
        )

        Y = tf.transpose(outputs.stack(), [1, 0, 2])
        Y = tf.ensure_shape(Y, [None, None, 1])
        return Y

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        horizon = input_shape[0][1]
        return (batch, horizon, 1)

def build_model(hp: dict, seq_len=SEQ_LEN, Fh=13, Ff=11):
    enc_units  = int(hp.get('enc_units', 256))
    enc_layers = int(hp.get('enc_layers', 2))
    dec_units  = int(hp.get('dec_units', 256))
    attn_units = int(hp.get('attn_units', enc_units))
    exg_embed  = int(hp.get('exog_embed', 64))
    dropout    = float(hp.get('dropout', 0.1))
    lr         = float(hp.get('lr', 3e-4))

    enc_activation           = hp.get('enc_activation', 'tanh')
    enc_recurrent_activation = hp.get('enc_recurrent_activation', 'sigmoid')
    dec_activation           = hp.get('dec_activation', 'tanh')
    dec_recurrent_activation = hp.get('dec_recurrent_activation', 'sigmoid')
    exog_activation          = hp.get('exog_activation', 'relu')

    hist     = L.Input(shape=(seq_len, Fh), name='hist')
    fut_exog = L.Input(shape=(HORIZON, Ff), name='future_exog')
    y_in     = L.Input(shape=(HORIZON, 1),  name='y_in')
    is_training = L.Input(shape=(), dtype=tf.bool, name='is_training')

    x = hist
    for _ in range(enc_layers):
        x = L.LSTM(enc_units, return_sequences=True,
                   activation=enc_activation,
                   recurrent_activation=enc_recurrent_activation)(x)
        x = L.Dropout(dropout)(x)
    enc_outputs = x
    _, enc_h, enc_c = L.LSTM(enc_units, return_sequences=False, return_state=True,
                             activation=enc_activation,
                             recurrent_activation=enc_recurrent_activation)(enc_outputs)
    init_h = L.Dense(dec_units, activation='tanh')(enc_h)
    init_c = L.Dense(dec_units, activation='tanh')(enc_c)

    y_hat_resid = ARDecoder(dec_units=dec_units,
                            exg_embed_dim=exg_embed,
                            attn_units=attn_units,
                            dropout=dropout,
                            exog_activation=exog_activation,
                            dec_activation=dec_activation,
                            dec_recurrent_activation=dec_recurrent_activation)(
        ([fut_exog, y_in, enc_outputs, init_h, init_c, hist, is_training])
    )

    model = Model(inputs=[hist, fut_exog, y_in, is_training], outputs=y_hat_resid)

    huber = tf.keras.losses.Huber(delta=0.02, reduction=tf.keras.losses.Reduction.NONE)

    def deriv_loss_per_timestep(y_true, y_pred):
        dt_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        dt_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        d = tf.reduce_mean(tf.abs(dt_true - dt_pred), axis=-1)
        d = tf.pad(d, paddings=[[0, 0], [0, 1]], mode='CONSTANT', constant_values=0.0)
        return d

    def total_loss(y_true, y_pred):
        base = huber(y_true, y_pred)
        deriv = deriv_loss_per_timestep(y_true, y_pred)
        return base + DERIV_LAMBDA * deriv

    model.compile(
        optimizer=O.Adam(learning_rate=lr),
        loss=total_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                 tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model
