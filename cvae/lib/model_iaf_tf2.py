import tensorflow as tf
import numpy as np


def kl_divergence(sigma, epsilon, z_K, param, batch_mean=True):
    """KL divergence between posterior with autoregressive flow and prior."""
    # logprob of posterior
    log_q_z0 = -0.5 * tf.square(epsilon)

    # logprob of prior
    log_p_zK = 0.5 * tf.square(z_K)

    # Terms from each flow layer
    flow_loss = 0
    for l in range(param['iaf_flow_length'] + 1):
        # Make sure it can't take log(0) or log(neg)
        flow_loss -= tf.math.log(sigma[l] + 1e-10)

    kl_divs = tf.identity(log_q_z0 + flow_loss + log_p_zK)
    kl_divs_reduced = tf.reduce_sum(kl_divs, axis=1)

    if batch_mean:
        return tf.reduce_mean(kl_divs, axis=0), tf.reduce_mean(kl_divs_reduced)
    else:
        return kl_divs, kl_divs_reduced


class EncoderBlock(tf.keras.layers.Layer):
    """Encoder block with configurable activation and dropout."""
    def __init__(self, units, activation=tf.nn.relu, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            units, 
            activation=activation,
            kernel_initializer='orthogonal'
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        x = self.dense(inputs)
        return self.dropout(x, training=training)


class DecoderBlock(tf.keras.layers.Layer):
    """Decoder block with configurable activation and dropout."""
    def __init__(self, units, activation=tf.nn.relu, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            units, 
            activation=activation,
            kernel_initializer='orthogonal'
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        x = self.dense(inputs)
        return self.dropout(x, training=training)


class AutoregressiveFlow(tf.keras.layers.Layer):
    """Inverse Autoregressive Flow layer."""
    def __init__(self, hidden_units, latent_dim, activation=tf.nn.relu, **kwargs):
        super().__init__(**kwargs)
        self.hidden = tf.keras.layers.Dense(
            hidden_units, 
            activation=activation,
            kernel_initializer='orthogonal'
        )
        self.mu = tf.keras.layers.Dense(latent_dim, kernel_initializer='orthogonal')
        self.sigma = tf.keras.layers.Dense(
            latent_dim, 
            activation=tf.nn.softplus,
            kernel_initializer='orthogonal'
        )
        
    def call(self, z, h):
        hidden = self.hidden(h)
        mu = self.mu(hidden)
        sigma = self.sigma(hidden) + 1.0
        return mu + sigma * z, sigma


class VAEModel(tf.keras.Model):
    """Variational Autoencoder with Inverse Autoregressive Flow."""
    def __init__(self, param, batch_size, input_dim, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.param = param
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        
        # Architecture parameters
        self.cells_enc = param['cells_encoder']
        self.cells_dec = self.cells_enc[::-1]
        self.cells_hidden = param['cells_hidden']
        self.dim_latent = param['dim_latent']
        self.iaf_flow_length = param['iaf_flow_length']
        
        # Build encoder
        self.encoder_layers = []
        for units in self.cells_enc:
            self.encoder_layers.append(
                EncoderBlock(units, dropout_rate=dropout_rate)
            )
            
        # Latent space projections
        self.z_mu = tf.keras.layers.Dense(
            self.dim_latent,
            kernel_initializer='orthogonal',
            name='z_mu'
        )
        self.z_sigma = tf.keras.layers.Dense(
            self.dim_latent,
            activation=tf.nn.softplus,
            kernel_initializer='orthogonal',
            name='z_sigma'
        )
        
        # IAF layers
        self.context_h = tf.keras.layers.Dense(
            self.cells_hidden,
            activation=tf.nn.relu,
            kernel_initializer='orthogonal'
        )
        self.flow_layers = []
        for _ in range(self.iaf_flow_length):
            self.flow_layers.append(
                AutoregressiveFlow(self.cells_hidden, self.dim_latent)
            )
            
        # Build decoder
        self.decoder_layers = []
        for units in self.cells_dec:
            self.decoder_layers.append(
                DecoderBlock(units, dropout_rate=dropout_rate)
            )
        self.decoder_mu = tf.keras.layers.Dense(
            self.input_dim,
            kernel_initializer='orthogonal',
            name='decoder_mu'
        )
        
    def encode(self, x, training=False):
        # Encoder forward pass
        h = x
        for layer in self.encoder_layers:
            h = layer(h, training=training)
            
        # Get latent parameters
        z_mu = self.z_mu(h)
        z_sigma = self.z_sigma(h) + 1.0
        
        # Sample z_0
        epsilon = tf.random.normal(shape=tf.shape(z_mu))
        z_0 = z_mu + z_sigma * epsilon
        
        # Get context for IAF
        h = self.context_h(h)
        
        # Apply flow layers
        z_k = z_0
        sigmas = [z_sigma]
        for flow in self.flow_layers:
            z_k, sigma = flow(z_k, h)
            sigmas.append(sigma)
            
        return z_k, sigmas, epsilon
        
    def decode(self, z, training=False):
        h = z
        for layer in self.decoder_layers:
            h = layer(h, training=training)
        return self.decoder_mu(h)
    
    def call(self, inputs, training=False):
        # Encode
        z_k, sigmas, epsilon = self.encode(inputs, training=training)
        
        # Decode
        x_mu = self.decode(z_k, training=training)
        
        # Compute losses
        reconstruction_loss = 0.5 * tf.reduce_sum(
            tf.square(x_mu - inputs), axis=1
        )
        kl_loss = kl_divergence(
            sigmas, epsilon, z_k, self.param, batch_mean=False
        )[1]
        
        self.add_loss(tf.reduce_mean(reconstruction_loss + kl_loss))
        
        return x_mu
