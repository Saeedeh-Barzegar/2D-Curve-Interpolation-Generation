import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import keras.backend as K

def inception_block(x, filters, conv_transpose=False):
    Conv = layers.Conv1DTranspose if conv_transpose else layers.Conv1D

    path1 = Conv(filters, kernel_size=1, padding='same', activation='relu')(x)
    path2 = Conv(filters, kernel_size=1, padding='same', activation='relu')(x)
    path2 = Conv(filters, kernel_size=3, padding='same', activation='relu')(path2)
    path3 = Conv(filters, kernel_size=1, padding='same', activation='relu')(x)
    path3 = Conv(filters, kernel_size=5, padding='same', activation='relu')(path3)
    
    path4 = layers.MaxPool1D(pool_size=3, strides=1, padding='same')(x)
    path4 = Conv(filters, kernel_size=1, padding='same', activation='relu')(path4)

    return layers.concatenate([path1, path2, path3, path4])

def calculate_laplacian(curve):
    lap1 = curve[:, 2:3] - 2 * curve[:, 1:2] + curve[:, 0:1]
    lapc = curve[:, 2:] - 2 * curve[:, 1:-1] + curve[:, :-2]
    lapn = curve[:, -1:] - 2 * curve[:, -2:-1] + curve[:, -3:-2]
    return tf.concat([lap1, lapc, lapn], axis=1)

def build_vae(input_size=161, latent_dim=32, kl_weight=1e-3):
    input_sdf = Input(shape=(input_size, 2))

    # Encoder
    x = layers.Conv1D(64, kernel_size=5, activation='relu')(input_sdf)
    x = inception_block(x, 64)
    x = layers.Conv1D(128, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=3)(x)

    x = inception_block(x, 128)
    x = layers.Conv1D(256, kernel_size=3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(pool_size=3)(x)

    flat = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(flat)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(flat)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
    encoder = Model(input_sdf, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = layers.Dense(K.int_shape(x)[1] * K.int_shape(x)[2], activation='relu')(latent_inputs)
    x = layers.Reshape((K.int_shape(x)[1], K.int_shape(x)[2]))(x)

    x = layers.Conv1DTranspose(256, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D(3)(x)

    x = inception_block(x, 128, conv_transpose=True)
    x = layers.Conv1DTranspose(128, 3, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D(3)(x)

    x = inception_block(x, 64, conv_transpose=True)
    x = layers.Conv1DTranspose(64, 5, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling1D(3)(x)

    x = layers.Conv1DTranspose(2, 3, activation='relu')(x)
    x = x[:, :input_size, :]  # Crop to match input length

    decoder = Model(latent_inputs, x, name="decoder")

    # VAE
    vae_output = decoder(z)
    vae = Model(input_sdf, vae_output, name="vae")

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())

    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    laplacian_loss = tf.reduce_mean(tf.abs(calculate_laplacian(input_sdf) - calculate_laplacian(vae_output)))

    vae.add_loss(kl_weight * kl_loss + laplacian_loss)
    

    return vae, encoder, decoder
