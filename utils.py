import tensorflow as tf
from tensorflow.contrib import layers


class NeuralLayers:
    def __init__(self,
                 trainable,
                 is_train,
                 hparams):

        self.trainable = trainable
        self.is_train = is_train
        self.hparams = hparams

        self.conv_kernel_initializer = layers.xavier_initializer()

        if trainable and hparams.conv_kernel_regularizer_scale > 0:
            self.conv_kernel_regularizer = layers.l2_regularizer(scale=hparams.conv_kernel_regularizer_scale)
        else:
            self.conv_kernel_regularizer = None

        if trainable and hparams.conv_activity_regularizer_scale > 0:
            self.conv_activity_regularizer = layers.l1_regularizer(scale=hparams.conv_activity_regularizer_scale)
        else:
            self.conv_activity_regularizer = None

        self.fc_kernel_initializer = tf.random_uniform_initializer(
            minval=-hparams.dense_kernel_initializer_scale,
            maxval=hparams.dense_kernel_initializer_scale)

        if trainable and hparams.dense_kernel_regularizer_scale > 0:
            self.fc_kernel_regularizer = layers.l2_regularizer(
                scale=hparams.dense_kernel_regularizer_scale)
        else:
            self.fc_kernel_regularizer = None

        if trainable and hparams.dense_activity_regularizer_scale > 0:
            self.fc_activity_regularizer = layers.l1_regularizer(
                scale=hparams.dense_activity_regularizer_scale)
        else:
            self.fc_activity_regularizer = None

        self.dense_drop_rate = hparams.dense_drop_rate

    def conv2d(self,
               inputs,
               filters,
               kernel_size=(3, 3),
               strides=(1, 1),
               activation=tf.nn.relu,
               use_bias=True,
               name=None):
        """ 2D Convolution layer. """
        if activation is not None:
            activity_regularizer = self.conv_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.conv2d(inputs=inputs,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='same',
                                activation=activation,
                                use_bias=use_bias,
                                trainable=self.trainable,
                                kernel_initializer=self.conv_kernel_initializer,
                                kernel_regularizer=self.conv_kernel_regularizer,
                                activity_regularizer=activity_regularizer,
                                name=name)

    def max_pool2d(self,
                   inputs,
                   pool_size=(2, 2),
                   strides=(2, 2),
                   name=None):
        """ 2D Max Pooling layer. """
        return tf.layers.max_pooling2d(inputs=inputs,
                                       pool_size=pool_size,
                                       strides=strides,
                                       padding='same',
                                       name=name)

    def global_avg_pool2d(self,
                          inputs,
                          keepdims=True,
                          name=None):
        return tf.reduce_mean(inputs,
                              axis=(1, 2),
                              keepdims=keepdims,
                              name=name)

    def dense(self,
              inputs,
              units,
              activation=tf.tanh,
              use_bias=True,
              name=None):
        """ Fully-connected layer. """
        if activation is not None:
            activity_regularizer = self.fc_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.dense(inputs=inputs,
                               units=units,
                               activation=activation,
                               use_bias=use_bias,
                               trainable=self.trainable,
                               kernel_initializer=self.fc_kernel_initializer,
                               kernel_regularizer=self.fc_kernel_regularizer,
                               activity_regularizer=activity_regularizer,
                               name=name)

    def dropout(self,
                inputs,
                name=None):
        """ Dropout layer. """
        return tf.layers.dropout(inputs=inputs,
                                 rate=self.dense_drop_rate,
                                 training=self.is_train,
                                 name=name)

    def batch_norm(self,
                   inputs,
                   name=None):
        """ Batch normalization layer. """
        return tf.layers.batch_normalization(inputs=inputs,
                                             training=self.is_train,
                                             trainable=self.trainable,
                                             name=name)


class Optimizer:

    def __init__(self,
                 hparams):
        self.hparams = hparams

    def GradientDescent(self,
                        learning_rate=None):
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate or self.hparams.initial_learning_rate)

    def Momentum(self,
                 learning_rate=None,
                 momentum=None,
                 use_nesterov=None):
        hparams = self.hparams
        return tf.train.MomentumOptimizer(learning_rate=learning_rate or hparams.initial_learning_rate,
                                          momentum=momentum or hparams.momentum,
                                          use_nesterov=use_nesterov or hparams.use_nesterov)

    def RMSProp(self,
                learning_rate=None,
                decay=None,
                momentum=None,
                centered=None,
                epsilon=None
                ):
        hparams = self.hparams
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate or hparams.initial_learning_rate,
                                         decay=decay or hparams.decay,
                                         momentum=momentum or hparams.momentum,
                                         centered=centered or hparams.centered,
                                         epsilon=epsilon or hparams.epsilon)

    def Adam(self,
             learning_rate=None,
             beta1=None,
             beta2=None,
             epsilon=None):
        hparams = self.hparams
        return tf.train.AdamOptimizer(learning_rate=learning_rate or hparams.initial_learning_rate,
                                      beta1=beta1 or hparams.beta1,
                                      beta2=beta2 or hparams.beta2,
                                      epsilon=epsilon or hparams.epsilon)

    def build(self,
              name,
              learning_rate=None,
              momentum=None,
              use_nesterov=None,
              decay=None,
              centered=None,
              epsilon=None,
              beta1=None,
              beta2=None):
        name = name.lower()
        if name == 'sgd':
            return self.GradientDescent(learning_rate=learning_rate)
        elif name == 'momentum':
            return self.Momentum(learing_rate=learning_rate,
                                 momentum=momentum,
                                 use_nesterov=use_nesterov)
        elif name == 'rmsprop':
            return self.RMSProp(learning_rate=learning_rate,
                                decay=decay,
                                momentum=momentum,
                                centered=centered,
                                epsilon=epsilon)
        elif name == 'adam':
            return self.Adam(learning_rate=learning_rate,
                             beta1=beta1,
                             beta2=beta2,
                             epsilon=epsilon)
        else:
            raise ValueError('Unknown optimizer name')

    def compute_learning_rate(self,
                              global_step,
                              initial_learning_rate=None,
                              learning_rate_decay_factor=None,
                              num_steps_per_decay=None):
        hparams = self.hparams
        learning_rate_decay_factor = learning_rate_decay_factor or hparams.learning_rate_decay_factor
        if learning_rate_decay_factor < 1.0:
            learning_rate = tf.train.exponential_decay(
                learning_rate=initial_learning_rate or hparams.initial_learning_rate,
                global_step=global_step,
                decay_steps=num_steps_per_decay or hparams.num_steps_per_decay,
                decay_rate=learning_rate_decay_factor)
        else:
            learning_rate = initial_learning_rate or hparams.initial_learning_rate
        return learning_rate
