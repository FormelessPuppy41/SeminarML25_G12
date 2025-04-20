print('Latest best version')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import math
import time
import random

import argparse

# Argument parser for scalar to control mean centering
parser = argparse.ArgumentParser()
parser.add_argument('--with_mean', type=lambda x: x.lower() == 'true', default=False, help='Use mean centering in StandardScaler')
args = parser.parse_args()

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# feature groups
MAIN_COLS = ['HR', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']
TIME_COLS = ['sin_time', 'cos_time']
ALL_COLS  = MAIN_COLS + TIME_COLS

# column‑wise scalers
scaler_main = StandardScaler(with_mean=args.with_mean)
scaler_time = StandardScaler()


# Standard Sinusoidal Positional Encoding Layer
class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(self.position, self.d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model,
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model,
        )
        # Apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Interleave sines and cosines
        pos_encoding = tf.stack([sines, cosines], axis=-1)
        pos_encoding = tf.reshape(pos_encoding, [position, d_model])

        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def get_angles(self, position, i, d_model):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angle_rates

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        # Ensure positional encoding is not longer than input sequence
        # Add encoding to the input tensor
        return inputs + self.pos_encoding[:, :seq_len, :]

# LogSparse + Local Attention Layer
class LogSparseLocalMultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, local_context, dropout_rate=0.1, **kwargs):
        super(LogSparseLocalMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.local_context = local_context
        self.dropout_rate = dropout_rate

        if d_model % self.num_heads != 0:
            raise ValueError(
                f"embedding dimension = {d_model} should be divisible by number of heads = {num_heads}"
            )

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dropout = layers.Dropout(dropout_rate)
        self.dense = layers.Dense(d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "local_context": self.local_context,
            "dropout_rate": self.dropout_rate,
        })
        return config

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def create_log_sparse_local_mask(self, seq_len):
        # Causal Mask (standard)
        causal_mask = 1 - tf.linalg.band_part(tf.ones([seq_len, seq_len], dtype=tf.float32), -1, 0)

        causal_mask = tf.cast(causal_mask, tf.bool) # True where attention is NOT allowed

        # Local Mask
        local_indices = tf.range(seq_len)
        local_dist = tf.abs(local_indices[:, tf.newaxis] - local_indices[tf.newaxis, :])
        local_mask = local_dist > self.local_context # True where distance > local_context

        # LogSparse Mask
        # Indices j where attention is allowed from i: j <= i and roughly i-j = 2^k or 0
        row_indices = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        col_indices = tf.range(seq_len, dtype=tf.float32)[tf.newaxis, :]
        diff = row_indices - col_indices

        # Allow attention if diff is 0 or a power of 2 (approximately)
        # Using log2 for approximation: check if log2(diff) is close to an integer
        log_diff = tf.math.log(tf.maximum(diff, 1.0)) / tf.math.log(2.0)

        # Check if fractional part is small (close to power of 2) or diff is 0
        is_power_of_2_approx = tf.abs(log_diff - tf.round(log_diff)) < 0.1
        is_zero = tf.equal(diff, 0.0)

        # LogSparse attention allowed where it's zero OR approx power of 2
        log_sparse_allowed = tf.logical_or(is_zero, is_power_of_2_approx)

        # We want the mask where attention is NOT allowed
        log_sparse_mask = tf.logical_not(log_sparse_allowed)

        # Combine Masks: Attention is NOT allowed if it's causal OR outside local OR not logsparse
        # However, LogSparse should *override* the local context restriction for its specific indices
        # So, allow if (NOT causal) AND ( (inside local context) OR (is logsparse allowed) )
        not_causal_mask = tf.logical_not(causal_mask)
        inside_local_mask = tf.logical_not(local_mask)

        combined_allowed = tf.logical_and(not_causal_mask,
                                          tf.logical_or(inside_local_mask, log_sparse_allowed))

        final_mask = tf.logical_not(combined_allowed)

        return final_mask


    def call(self, v, k, q, training=False):
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]

        # Linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Calculate attention scores
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Create and apply the combined mask
        mask = self.create_log_sparse_local_mask(seq_len)

        # Add mask to the scaled tensor.
        scaled_attention_logits += (tf.cast(mask, dtype=scaled_attention_logits.dtype) * -1e9)

        # Softmax normalization to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply dropout to attention weights
        if training:
             attention_weights = self.dropout(attention_weights, training=training)

        # Weighted sum of values
        output = tf.matmul(attention_weights, v)

        # Transpose back to original shape
        output = tf.transpose(output, perm=[0, 2, 1, 3])

        # Concatenate heads
        original_size_output = tf.reshape(output, (batch_size, seq_len, self.d_model))

        # Final linear layer
        output = self.dense(original_size_output)

        return output, attention_weights


# Encoder Block
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, local_context, dropout_rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.local_context = local_context
        self.dropout_rate = dropout_rate

        self.mha = LogSparseLocalMultiHeadAttention(d_model, num_heads, local_context, dropout_rate)
        self.ffn = keras.Sequential(
            [layers.Dense(dff, activation="relu"), layers.Dense(d_model)]
        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "local_context": self.local_context,
            "dropout_rate": self.dropout_rate,
        })
        return config

    # Using Pre-LN structure (Norm -> Sublayer -> Dropout -> Add)
    def call(self, x, training=False):
        # Multi-Head Attention Block
        # Apply LayerNorm, then MHA, then dropout, then residual connection
        x_norm = self.layernorm1(x)
        attn_output, _ = self.mha(v=x_norm, k=x_norm, q=x_norm, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output

        # Feed Forward Block
        # Apply LayerNorm, then FFN, then dropout, then residual connection
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output

        return out2


# Decoder Block
class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, local_context, dropout_rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        # Masked self-attention
        self.mha1 = LogSparseLocalMultiHeadAttention(d_model, num_heads, local_context, dropout_rate)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)

        # Cross-attention
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(dropout_rate)

        # FFN
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation="relu"),
            layers.Dense(d_model)
        ])
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout3 = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError("DecoderLayer expects two inputs, got: {}".format(input_shape))

        dec_input_shape, enc_output_shape = input_shape

        self.mha1.build(dec_input_shape)
        self.mha2.build(
            query_shape=dec_input_shape,
            key_shape=enc_output_shape,
            value_shape=enc_output_shape
        )
        self.ffn.build(dec_input_shape)
        super(DecoderLayer, self).build(input_shape)



    def call(self, inputs, training=False):
        x, enc_output = inputs

        # Masked Self-Attention on decoder input
        attn1_output, _ = self.mha1(v=x, k=x, q=x, training=training)
        attn1_output = self.dropout1(attn1_output, training=training)
        out1 = self.layernorm1(x + attn1_output)

        # Cross-Attention: use the encoder output as keys/values
        attn2_output = self.mha2(query=out1, key=enc_output, value=enc_output, training=training)
        attn2_output = self.dropout2(attn2_output, training=training)
        out2 = self.layernorm2(out1 + attn2_output)

        # Feed Forward Network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3


# Full transformer model with encoder and decoder
def build_transformer_model_with_decoder(
    encoder_input_shape,
    decoder_input_shape,
    d_model,
    num_heads,
    dff,
    num_encoder_layers,
    num_decoder_layers,
    output_seq_len,
    conv_filters,
    conv_kernel_size,
    conv_stride,
    local_context,
    dropout_rate=0.1,
    use_positional_encoding=True
    ):

    # Encoder
    encoder_inputs = keras.Input(shape=encoder_input_shape, name="encoder_inputs")
    input_seq_len = encoder_input_shape[0]
    # Calculate sequence length after Conv1D downsampling
    conv_output_seq_len = math.ceil(input_seq_len / conv_stride)

    x = layers.Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_stride,
                      padding='causal', activation='relu')(encoder_inputs)
    if conv_filters != d_model:
         x = layers.Dense(d_model)(x)


    # Add positional encoding
    if use_positional_encoding:
         x = PositionalEncoding(position=conv_output_seq_len, d_model=d_model)(x)

    x = layers.Dropout(dropout_rate)(x)

    for _ in range(num_encoder_layers):
         x = EncoderLayer(d_model, num_heads, dff, local_context, dropout_rate)(x)
    
    # Take the output of the last time step from the final encoder layer
    encoder_output = x

    # Decoder
    decoder_inputs = keras.Input(shape=decoder_input_shape, name="decoder_inputs")
    # Project the forecast input to d_model
    y = layers.Dense(d_model)(decoder_inputs)

    # Add positional encoding for decoder input
    if use_positional_encoding:
         y = PositionalEncoding(position=decoder_input_shape[0], d_model=d_model)(y)

    y = layers.Dropout(dropout_rate)(y)

    # Stacked Decoder Layers
    for _ in range(num_decoder_layers):
        y = DecoderLayer(d_model, num_heads, dff, local_context=decoder_input_shape[0], dropout_rate=dropout_rate)([y, encoder_output])


    # Final output head: for each time step in the decoder output, predict the HR forecast.
    # Use a Dense layer to project the output to the desired output dimension (1 in this case)
    outputs = layers.Dense(1, name="output")(y)

    model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    return model


def scale_window(df_window: pd.DataFrame, fit: bool = False):
    """Return a (N, 9) numpy array with HR–A6 scaled by `scaler_main`
       and sin/cos scaled by `scaler_time`."""
    main = df_window[MAIN_COLS].values
    time = df_window[TIME_COLS].values

    if fit:
        scaler_main.fit(main)
        scaler_time.fit(time)

    main_scaled = scaler_main.transform(main)
    time_scaled = scaler_time.transform(time)

    return np.hstack([main_scaled, time_scaled])



# Main
if __name__ == '__main__':
    use_gpu = True

    if use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("Using GPU:", gpus[0])
            except RuntimeError as e:
                print("GPU setup error:", e)
        else:
            print("No GPU detected. Running on CPU.")
    else:
        # Force TensorFlow to use only CPU
        tf.config.set_visible_devices([], 'GPU')
        print("Forcing TensorFlow to run on CPU only.")



    # Hyperparameters & Configuration
    INPUT_DIM = 9               # HR + A1 to A6 + sin/cos time
    D_MODEL = 128               # Embedding dimension
    NUM_HEADS = 8               # Number of attention heads (must be divisor of D_MODEL)
    NUM_ENCODER_LAYERS = 3      # Number of encoder layers
    DFF = 512                   # Hidden dimension in FFN
    OUTPUT_SEQ_LEN = 96         # Forecast horizon (96 steps = 24 hours)
    DROPOUT_RATE = 0.1          # Dropout rate for regularization

    # Convolution Layer parameters
    CONV_FILTERS = D_MODEL      # Number of filters in Conv1D layer
    CONV_KERNEL_SIZE = 5        # Kernel size for Conv1D layer
    CONV_STRIDE = 4             # Stride for Conv1D layer (downsampling factor)

    # Attention parameters
    # Local context after downsampling. If stride=4, 24 steps = 96 quarter-hours original.
    LOCAL_CONTEXT = 720

    # Rolling Window & Training parameters
    INPUT_WINDOW_DAYS = 164
    INPUT_WINDOW_LEN = INPUT_WINDOW_DAYS * 96 

    BATCH_SIZE = 32             # Batch size for training
    EPOCHS_PER_WINDOW = 20      # Number of epochs for each rolling window
    LEARNING_RATE = 0.0005      # Learning rate for optimizer
    CLIP_VALUE = 1.0            # Gradient clipping value for optimizer

    # Forecast Period Setup
    # Set the forecast period as desired (ensure forecast_start_date is late enough to have 165 days history)
    forecast_start_date = pd.Timestamp("2012-06-15")
    forecast_end_date   = pd.Timestamp("2018-12-31")
    forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')

    # Load and Prepare Data
    print("Loading data...")
    try:
        # Load your full multi-year dataset here
        df_full = pd.read_csv('error_model_combined_forecasts2.csv', parse_dates=['datetime'], index_col='datetime')
        df_full = df_full.dropna()
        print(f"Full data loaded with shape: {df_full.shape}")
        if len(df_full) < INPUT_WINDOW_LEN + OUTPUT_SEQ_LEN:
             raise ValueError("Not enough data in the file for even one training window.")
    except FileNotFoundError:
        print("Error: Data file not found. Aborting.")
        exit()
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    # Ensure your data is sorted by time
    df_full = df_full.sort_index()

    # Create a complete datetime index from the first to the last date with 15-minute frequency
    full_index = pd.date_range(start=df_full.index.min(), end=df_full.index.max(), freq='15min')

    # Reindex the DataFrame to this complete index
    df_full = df_full.reindex(full_index)

    # Fill missing values; you can choose your preferred method (ffill, bfill, interpolation, etc.)
    df_full = df_full.ffill()


    df_full['hour'] = df_full.index.hour
    df_full['sin_time'] = np.sin(2 * np.pi * df_full['hour'] / 24)
    df_full['cos_time'] = np.cos(2 * np.pi * df_full['hour'] / 24)


    print(f"Full data loaded with shape: {df_full.shape}")

    # Select Features
    input_cols = ['HR', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'sin_time', 'cos_time']
    target_col = 'HR'
    all_cols = input_cols

    # Ensure columns exist
    missing_cols = [col for col in all_cols if col not in df_full.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Model Building
    input_shape = (INPUT_WINDOW_LEN, INPUT_DIM)
    DECODER_INPUT_DIM = 9
    decoder_input_shape = (OUTPUT_SEQ_LEN, DECODER_INPUT_DIM)

    # Use the new encoder-decoder model builder:
    model = build_transformer_model_with_decoder(
        encoder_input_shape=input_shape,
        decoder_input_shape=decoder_input_shape,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_ENCODER_LAYERS,
        output_seq_len=OUTPUT_SEQ_LEN,
        conv_filters=CONV_FILTERS,
        conv_kernel_size=CONV_KERNEL_SIZE,
        conv_stride=CONV_STRIDE,
        local_context=LOCAL_CONTEXT,
        dropout_rate=DROPOUT_RATE,
        use_positional_encoding=True
    )

    # Use Adam optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_VALUE)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

   # Rolling Window Training, Prediction, and Results Accumulation
    results_list = []
    current_year = None

    start_time_total = time.time()


    for current_day in forecast_dates:
        print(f"\n Forecasting for {current_day.date()}")
        start_time_day = time.time()

        
        # Window layout for this forecast day D
        # D‑1  = slack / “yesterday” (not used for training)
        # D‑2  = finetune
        # D‑3 - D‑8  = 6‑day validation slice
        # D‑9 - D‑165 = 157‑day training slice

        slack_day = current_day - pd.Timedelta(days=1)
        finetune_day = slack_day - pd.Timedelta(days=1)

        val_end_day = finetune_day - pd.Timedelta(minutes=15)
        val_start_day = val_end_day - pd.Timedelta(days=6) + pd.Timedelta(minutes=15)

        train_end_day = val_start_day - pd.Timedelta(minutes=15)
        train_start_day = train_end_day - pd.Timedelta(days=158) + pd.Timedelta(minutes=15)


        # pull dataframes
        df_train = df_full.loc[train_start_day:train_end_day]
        df_val = df_full.loc[val_start_day:val_end_day]
        df_slack = df_full.loc[slack_day:slack_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)]
        df_ft = df_full.loc[finetune_day:finetune_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)]
        df_target= df_full.loc[current_day:current_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)]

        # basic sanity
        if len(df_train) < 157*96 or len(df_val) < 6*96 or len(df_target) < 96:
            print(f"Skipping {current_day.date()} (insufficient history)")
            continue

        enc_train = scale_window(df_train, fit=True)
        enc_val = scale_window(df_val)

        # encoder input shape is (W, 9) where W = 165*96
        x_train_enc = np.expand_dims(np.vstack([enc_train, enc_val]), axis=0)
        x_val_enc = np.expand_dims(enc_val, axis=0)

        # Decoder teacher forcing 
        # HR_true plus exogenous, shifted right by 1 (first HR is 0 as BOS token)
        def make_teacher(df_slice):
            slc = scale_window(df_slice)[-96:]
            dec = np.zeros_like(slc)
            dec[1:, 0] = slc[:-1, 0]
            dec[:, 1:] = slc[:, 1:]
            return dec

        dec_train_teacher = np.expand_dims(make_teacher(df_train.iloc[-96:]), 0)
        y_train = scale_window(df_train.iloc[-96:])[:,0].reshape(1,96,1)

        train_ds = tf.data.Dataset.from_tensor_slices(
                    ( (x_train_enc, dec_train_teacher), y_train )
                ).batch(BATCH_SIZE)

        
        # build validation teacher/target
        dec_val_teacher = np.expand_dims(make_teacher(df_val.iloc[-96:]), 0)
        y_val = scale_window(df_val.iloc[-96:])[:,0].reshape(1,96,1)

        # validation dataset
        val_ds = tf.data.Dataset.from_tensor_slices(
                ((x_train_enc, dec_val_teacher), y_val)
            ).batch(BATCH_SIZE)
        
                                                                      
        history = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs = 30,
            callbacks = [EarlyStopping(patience=3, restore_best_weights=True)],
            verbose = 0,
        )

        # Fine‑tune with encoder that includes slack
        dec_ft_teacher = np.expand_dims(make_teacher(df_ft), axis=0)
        y_ft = scale_window(df_ft)[:96, 0].reshape(1, 96, 1)
        enc_ft = scale_window(df_ft)
        x_ft_enc_full = np.vstack([enc_train, enc_val, enc_ft])
        x_ft_enc = np.expand_dims(x_ft_enc_full[-INPUT_WINDOW_LEN:], axis=0)

        model.fit((x_ft_enc, dec_ft_teacher),
          y_ft,
          epochs=1, batch_size=1, verbose=0)
        
        # Build an initial decoder input that concatenates the known exogenous values
        # with a placeholder for HR (which you will fill in iteratively).
        target_scaled = scale_window(df_target)
        exogenous_inputs = target_scaled[:, 1:]

        initial_hr = np.zeros((OUTPUT_SEQ_LEN, 1), dtype=np.float32)
        decoder_input = np.concatenate([initial_hr, exogenous_inputs], axis=-1)
        decoder_input = np.expand_dims(decoder_input, axis=0)

        scaled_prediction = np.zeros((1, OUTPUT_SEQ_LEN, 1), dtype=np.float32)

        for t in range(OUTPUT_SEQ_LEN):
            current_input_padded = np.copy(decoder_input)
            # For time steps t+1 onward, zero out the first column (HR) only,
            # leaving the exogenous columns intact.
            current_input_padded[:, t+1:, 0] = 0

            pred = model.predict([x_train_enc, current_input_padded], verbose=0)
            next_value = pred[0, t, 0]

            decoder_input[0, t, 0] = next_value
            scaled_prediction[0, t, 0] = next_value

        # inverse only the HR column
        hr_scaled = scaled_prediction.reshape(-1, 1)

        # Create dummy input with the right shape (96, 7)
        hr_scaled_full = np.zeros((96, 7))
        hr_scaled_full[:, 0] = hr_scaled[:, 0]

        hr_original_scale = scaler_main.inverse_transform(hr_scaled_full)[:, 0]
        prediction_original_scale = hr_original_scale

        # Clip negative values to 0
        prediction_original_scale = np.clip(prediction_original_scale, 0, None)

        # Create a DataFrame for this forecast day with timestamps, actual HR, forecast, and squared error
        day_results = pd.DataFrame({
            "timestamp": df_target.index,
            "HR": df_target[target_col].values,
            "forecast": prediction_original_scale,
        })
        day_results["squared_error"] = (day_results["forecast"] - day_results["HR"])**2
        day_results["forecast_date"] = current_day.date()

        # Append results to our list
        results_list.append(day_results)

        # Check if we have finished a year (or this is the last forecast day)
        forecast_year = current_day.year
        if current_year is None:
            current_year = forecast_year
        if forecast_year > current_year or current_day == forecast_dates[-1]:
            # Combine all results so far
            master_df = pd.concat(results_list, ignore_index=True)
            master_df["year"] = master_df["timestamp"].dt.year
            # Compute MSE per year
            year_mse_map = master_df.groupby("year")["squared_error"].mean().to_dict()
            master_df["year_MSE"] = master_df["year"].map(year_mse_map)
            overall_mse = master_df["squared_error"].mean()
            master_df["overall_MSE"] = overall_mse

            # Write CSV (overwriting previous file) with selected columns
            master_df[["timestamp", "HR", "forecast", "squared_error", "year_MSE", "overall_MSE"]].to_csv(f"transformer_forecast_results_with_mean_{args.with_mean}.csv", index=False)
            print(f"  CSV file updated for year {current_year} (overall MSE so far: {overall_mse:.4f}).")
            current_year = forecast_year

        end_time_day = time.time()
        print(f"  Forecast for {current_day.date()} processed in {end_time_day - start_time_day:.2f} seconds.")

    end_time_total = time.time()
    print(f"\n Forecast Loop Finished")
    print(f"Total time: {end_time_total - start_time_total:.2f} seconds.")

    # Combine and print overall MSE
    if results_list:
        master_df = pd.concat(results_list, ignore_index=True)
        overall_mse = master_df["squared_error"].mean()
        print(f"\nOverall MSE for the forecast period: {overall_mse:.4f}")
    else:
        print("No forecasts were made.")