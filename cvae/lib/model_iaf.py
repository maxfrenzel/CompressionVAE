import tensorflow as tf


def create_variable(name, shape, initializer_type=None):
    """Create weight variable with the specified name and shape,
    and initialize it specified initializer."""
    if initializer_type == 'truncated_normal':
        initializer = tf.initializers.truncated_normal()
    elif initializer_type == 'lecun_normal':
        initializer = tf.initializers.lecun_normal()
    elif initializer_type == 'orthogonal':
        initializer = tf.initializers.orthogonal()
    else:
        print('No initializer type provided or provided type unknown. Defaulting to orthogonal.')
        initializer = tf.initializers.orthogonal()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape):
    """Create a bias variable with the specified name and shape and initialize it."""
    initializer = tf.constant_initializer(value=0.001, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


# KL divergence between posterior with autoregressive flow and prior
def kl_divergence(sigma, epsilon, z_K, param, batch_mean=True):
    # logprob of posterior
    log_q_z0 = -0.5 * tf.square(epsilon)

    # logprob of prior
    log_p_zK = 0.5 * tf.square(z_K)

    # Terms from each flow layer
    flow_loss = 0
    for l in range(param['iaf_flow_length'] + 1):
        # Make sure it can't take log(0) or log(neg)
        flow_loss -= tf.log(sigma[l] + 1e-10)

    kl_divs = tf.identity(log_q_z0 + flow_loss + log_p_zK)
    kl_divs_reduced = tf.reduce_sum(kl_divs, axis=1)

    if batch_mean:
        return tf.reduce_mean(kl_divs, axis=0), tf.reduce_mean(kl_divs_reduced)
    else:
        return kl_divs, kl_divs_reduced


class VAEModel(object):

    def __init__(self,
                 param,
                 batch_size,
                 input_dim,
                 activation=tf.nn.relu,
                 activation_nf=tf.nn.relu,
                 keep_prob=1.0,
                 encode=False,
                 initializer='orthogonal'):

        self.input_dim = input_dim
        self.param = param
        self.batch_size = batch_size
        self.activation = activation
        self.activation_nf = activation_nf
        self.encode = encode
        self.cells_enc = self.param['cells_encoder']
        self.layers_enc = len(param['cells_encoder'])
        self.cells_dec = self.cells_enc[::-1]
        self.layers_dec = self.layers_enc
        self.cells_hidden = self.param['cells_hidden']
        self.dim_latent = param['dim_latent']
        self.keep_prob = keep_prob
        self.initializer = initializer
        self.variables = self._create_variables()

    def _create_variables(self):
        """This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function."""

        var = dict()

        with tf.variable_scope('VAE'):

            with tf.variable_scope("Encoder"):

                var['encoder_stack'] = list()
                with tf.variable_scope('encoder_stack'):

                    for l, num_units in enumerate(self.cells_enc):

                        with tf.variable_scope('layer{}'.format(l)):

                            layer = dict()

                            if l == 0:
                                units_in = self.input_dim
                            else:
                                units_in = self.cells_enc[l - 1]

                            units_out = num_units

                            layer['W'] = create_variable("W",
                                                         shape=[units_in, units_out],
                                                         initializer_type=self.initializer)
                            layer['b'] = create_bias_variable("b",
                                                              shape=[1, units_out])

                            var['encoder_stack'].append(layer)

                with tf.variable_scope('fully_connected'):

                    layer = dict()

                    num_cells_hidden = self.cells_hidden

                    layer['W_z0'] = create_variable("W_z0",
                                                    shape=[self.cells_enc[-1], 2 * num_cells_hidden],
                                                    initializer_type=self.initializer)
                    layer['b_z0'] = create_bias_variable("b_z0",
                                                         shape=[1, 2 * num_cells_hidden])

                    layer['W_mu'] = create_variable("W_mu",
                                                    shape=[self.cells_hidden, self.param['dim_latent']],
                                                    initializer_type=self.initializer)
                    layer['W_logvar'] = create_variable("W_logvar",
                                                        shape=[self.cells_hidden, self.param['dim_latent']],
                                                        initializer_type=self.initializer)
                    layer['b_mu'] = create_bias_variable("b_mu",
                                                         shape=[1, self.param['dim_latent']])
                    layer['b_logvar'] = create_bias_variable("b_logvar",
                                                             shape=[1, self.param['dim_latent']])

                    var['encoder_fc'] = layer

            with tf.variable_scope("IAF"):

                var['iaf_flows'] = list()
                for l in range(self.param['iaf_flow_length']):

                    with tf.variable_scope('layer{}'.format(l)):

                        layer = dict()

                        # Hidden state
                        layer['W_flow'] = create_variable("W_flow",
                                                          shape=[self.cells_enc[-1], self.dim_latent],
                                                          initializer_type=self.initializer)
                        layer['b_flow'] = create_bias_variable("b_flow",
                                                             shape=[1, self.dim_latent])

                        flow_variables = list()
                        # Flow parameters from hidden state (m and s parameters for IAF)
                        for j in range(self.dim_latent):
                            with tf.variable_scope('flow_layer{}'.format(j)):

                                flow_layer = dict()

                                # Set correct dimensionality
                                units_to_hidden_iaf = self.param['dim_autoregressive_nl']

                                flow_layer['W_flow_params_nl'] = create_variable("W_flow_params_nl",
                                                                                 shape=[self.dim_latent + j,
                                                                                        units_to_hidden_iaf],
                                                                                 initializer_type=self.initializer)
                                flow_layer['b_flow_params_nl'] = create_bias_variable("b_flow_params_nl",
                                                                                      shape=[1, units_to_hidden_iaf])

                                flow_layer['W_flow_params'] = create_variable("W_flow_params",
                                                                              shape=[units_to_hidden_iaf,
                                                                                     2],
                                                                              initializer_type=self.initializer)
                                flow_layer['b_flow_params'] = create_bias_variable("b_flow_params",
                                                                                   shape=[1, 2])

                                flow_variables.append(flow_layer)

                        layer['flow_vars'] = flow_variables

                        var['iaf_flows'].append(layer)

            with tf.variable_scope("Decoder"):

                var['decoder_stack'] = list()
                with tf.variable_scope('deconv_stack'):

                    for l, num_units in enumerate(self.cells_dec):

                        with tf.variable_scope('layer{}'.format(l)):

                            layer = dict()

                            if l == 0:
                                units_in = self.dim_latent
                            else:
                                units_in = self.cells_dec[l - 1]

                            units_out = num_units

                            layer['W'] = create_variable("W",
                                                         shape=[units_in, units_out],
                                                         initializer_type=self.initializer)
                            layer['b'] = create_bias_variable("b",
                                                              shape=[1, units_out])

                            var['decoder_stack'].append(layer)

                with tf.variable_scope('fully_connected'):
                    layer = dict()

                    layer['W_mu'] = create_variable("W_mu",
                                                    shape=[self.cells_dec[-1], self.input_dim],
                                                    initializer_type=self.initializer)
                    # layer['W_logvar'] = create_variable("W_logvar",
                    #                                     shape=[self.cells_dec[-1], self.input_dim])
                    layer['b_mu'] = create_bias_variable("b_mu",
                                                         shape=[1, self.input_dim])
                    # layer['b_logvar'] = create_bias_variable("b_logvar",
                    #                                          shape=[1, self.input_dim])

                    var['decoder_fc'] = layer

        return var

    def _create_network(self, input_batch, encode=False):

        # -----------------------------------
        # Encoder

        # Remove redundant dimension (weird thing to get PaddingFIFOQueue to work)
        # input_batch = tf.squeeze(input_batch)

        # Do encoder calculation
        encoder_hidden = input_batch
        # print('Encoder hidden state 0: ', encoder_hidden)
        for l in range(self.layers_enc):
            encoder_hidden = tf.nn.dropout(self.activation(tf.matmul(encoder_hidden,
                                                                     self.variables['encoder_stack'][l]['W'])
                                                           + self.variables['encoder_stack'][l]['b']),
                                           keep_prob=self.keep_prob)

            # print(f'Encoder hidden state {l}: ', encoder_hidden)

        # encoder_hidden = tf.reshape(encoder_hidden, [-1, self.conv_out_units])

        # Additional non-linearity between encoder hidden state and prediction of mu_0,sigma_0
        mu_logvar_hidden = tf.nn.dropout(self.activation(tf.matmul(encoder_hidden,
                                                                   self.variables['encoder_fc']['W_z0'])
                                                         + self.variables['encoder_fc']['b_z0']),
                                         keep_prob=self.keep_prob)

        # Split into parts for mean and variance
        mu_hidden, logvar_hidden = tf.split(mu_logvar_hidden, num_or_size_splits=2, axis=1)

        # Final linear layer to calculate mean and variance
        encoder_mu = tf.add(tf.matmul(mu_hidden, self.variables['encoder_fc']['W_mu']),
                            self.variables['encoder_fc']['b_mu'], name='ZMu')
        encoder_logvar = tf.add(tf.matmul(logvar_hidden, self.variables['encoder_fc']['W_logvar']),
                                self.variables['encoder_fc']['b_logvar'], name='ZLogVar')

        # Convert log variance into standard deviation
        encoder_std = tf.exp(0.5 * encoder_logvar)

        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(encoder_std), name='epsilon')

        if encode:
            z0 = tf.identity(encoder_mu, name='LatentZ0')
        else:
            z0 = tf.identity(tf.add(encoder_mu, tf.multiply(encoder_std, epsilon),
                                    name='LatentZ0'))

        # -----------------------------------
        # Latent flow

        # Lists to store the latent variables and the flow parameters
        nf_z = [z0]
        nf_sigma = [encoder_std]

        # Do calculations for each flow layer
        for l in range(self.param['iaf_flow_length']):

            W_flow = self.variables['iaf_flows'][l]['W_flow']
            b_flow = self.variables['iaf_flows'][l]['b_flow']

            nf_hidden = self.activation_nf(tf.matmul(encoder_hidden, W_flow) + b_flow)

            # Autoregressive calculation
            m_list = self.dim_latent * [None]
            s_list = self.dim_latent * [None]

            for j, flow_vars in enumerate(self.variables['iaf_flows'][l]['flow_vars']):

                # Go through computation one variable at a time
                if j == 0:
                    hidden_autoregressive = nf_hidden
                else:
                    z_slice = tf.slice(nf_z[-1], [0, 0], [-1, j])
                    hidden_autoregressive = tf.concat(axis=1, values=[nf_hidden, z_slice])

                W_flow_params_nl = flow_vars['W_flow_params_nl']
                b_flow_params_nl = flow_vars['b_flow_params_nl']
                W_flow_params = flow_vars['W_flow_params']
                b_flow_params = flow_vars['b_flow_params']

                # Non-linearity at current autoregressive step
                nf_hidden_nl = self.activation_nf(tf.matmul(hidden_autoregressive,
                                                       W_flow_params_nl) + b_flow_params_nl)

                # Calculate parameters for normalizing flow as linear transform
                ms = tf.matmul(nf_hidden_nl, W_flow_params) + b_flow_params

                # Split into individual components
                # m_list[j], s_list[j] = tf.split_v(value=ms,
                #                    size_splits=[1,1],
                #                    split_dim=1)
                m_list[j], s_list[j] = tf.split(value=ms,
                                                num_or_size_splits=[1, 1],
                                                axis=1)

            # Concatenate autoregressively computed variables
            # Add offset to s to make sure it starts out positive
            # (could have also initialised the bias term to 1)
            # Guarantees that flow initially small
            m = tf.concat(axis=1, values=m_list)
            s = self.param['initial_s_offset'] + tf.concat(axis=1, values=s_list)

            # Calculate sigma ("update gate value") from s
            sigma = tf.nn.sigmoid(s)
            nf_sigma.append(sigma)

            # Perform normalizing flow
            z_current = tf.multiply(sigma, nf_z[-1]) + tf.multiply((1 - sigma), m)

            # Invert order of variables to alternate dependence of autoregressive structure
            z_current = tf.reverse(z_current, axis=[1], name='LatentZ%d' % (l + 1))

            # Add to list of latent variables
            nf_z.append(z_current)

        z = tf.identity(nf_z[-1], name="LatentZ")

        # -----------------------------------
        # Decoder

        # Fully connected
        decoder_hidden = z

        for l in range(self.layers_dec):
            # print(decoder_hidden)
            decoder_hidden = tf.nn.dropout(self.activation(tf.matmul(decoder_hidden,
                                                                     self.variables['decoder_stack'][l]['W'])
                                                           + self.variables['decoder_stack'][l]['b']),
                                           keep_prob=self.keep_prob)
            decoder_hidden = self.activation(decoder_hidden)

        # Split into mu and logvar parts
        # decoder_hidden_mu, decoder_hidden_logvar = tf.split(decoder_hidden, num_or_size_splits=2, axis=1)

        # Final layer
        decoder_mu = tf.add(tf.matmul(decoder_hidden, self.variables['decoder_fc']['W_mu']),
                            self.variables['decoder_fc']['b_mu'],
                            name='XMu')
        # decoder_logvar = tf.add(tf.matmul(decoder_hidden_logvar, self.variables['decoder_fc']['W_logvar']),
        #                         self.variables['decoder_fc']['b_logvar'],
        #                         name='XLogVar')
        #
        # # Add clipping to avoid zero division
        # decoder_logvar = tf.clip_by_value(decoder_logvar,
        #                                   clip_value_min=-8.0,
        #                                   clip_value_max=+8.0)

        # Set decoder variance as fixed hyperparameter for stability; common assumption in Gaussian decoders
        decoder_logvar = tf.zeros_like(decoder_mu)

        # return decoder_output, encoder_hidden, encoder_logvar, encoder_std
        return decoder_mu, decoder_logvar, encoder_mu, encoder_logvar, encoder_std, epsilon, z, z0, nf_sigma

    def decode(self, z):

        decoder_hidden = z

        for l in range(self.layers_dec):
            # print(decoder_hidden)
            decoder_hidden = tf.nn.dropout(self.activation(tf.matmul(decoder_hidden,
                                                                     self.variables['decoder_stack'][l]['W'])
                                                           + self.variables['decoder_stack'][l]['b']),
                                           keep_prob=self.keep_prob)
            decoder_hidden = self.activation(decoder_hidden)

        decoder_mu = tf.add(tf.matmul(decoder_hidden, self.variables['decoder_fc']['W_mu']),
                            self.variables['decoder_fc']['b_mu'],
                            name='XMu')

        return decoder_mu

    def input_identity(self, input_batch):

        # return tf.matmul(input_batch, self.variables['encoder_stack'][0]['W'])

        return self.variables['encoder_stack'][0]['W']

    def loss(self,
             input_batch,
             name='vae',
             beta=1.0,
             test=False):

        with tf.name_scope(name):

            # Run computation
            decoder_mu, decoder_logvar, encoder_mu, encoder_logvar, encoder_std, epsilon, z, z0, nf_sigma = self._create_network(input_batch)

            # print("Output size: ", decoder_mu)

            # KL-Divergence loss
            _, div = kl_divergence(nf_sigma, epsilon, z, self.param, batch_mean=False)
            loss_latent = tf.identity(div, name='LossLatent')

            # Reconstruction loss assuming Gaussian output distribution
            decoder_var = tf.exp(decoder_logvar)
            loss_reconstruction = tf.identity(0.5 * tf.reduce_sum(tf.math.divide(tf.square(input_batch - decoder_mu),
                                                                                 decoder_var)
                                                                  + decoder_logvar, axis=1),
                                              name='LossReconstruction')

            # Small penalty to prevent z0 values from going to infinity
            z0_boundary = 10.0 * tf.ones_like(z0)
            z0_for_penalty = tf.maximum(z0_boundary, tf.abs(z0))
            z0_large = tf.reduce_mean(tf.square(z0_for_penalty - z0_boundary), axis=1)

            loss = tf.reduce_mean(loss_reconstruction + beta*loss_latent, name='Loss')

            if not test:
                tf.summary.scalar('loss_total', loss)
                tf.summary.scalar('loss_rec_per_feat', tf.reduce_mean(loss_reconstruction)/self.input_dim)
                tf.summary.scalar('loss_kl_per_dim', tf.reduce_mean(loss_latent)/self.dim_latent)
                tf.summary.scalar('beta', beta)

            return loss

    def embed(self, input_batch):

        # Run computation
        _, _, _, _, _, _, z, _, _ = self._create_network(input_batch, encode=True)

        return z
