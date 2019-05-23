class HyperParameters(object):

    def __init__(self):

        self.dense_kernel_initializer_scale = 0.08
        self.dense_kernel_regularizer_scale = 1e-4
        self.dense_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.dense_drop_rate = 0.5

        # about the optimization
        self.num_epochs = 100
        self.batch_size = 64
        self.optimizer = 'Adam'  # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 0.01
        self.learning_rate_decay_factor = 1.0
        self.num_steps_per_decay = 20000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        self.license_number_list = [
            '0123456789',
            '0123456789',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            '0123456789',
            '0123456789',
            '0123456789',
            '0123456789',
        ]

        self.save_dir = '/content/drive/My Drive/deep-learning/models/license-plate-recognizer-model/'
        self.summary_period = 1
        self.summary_dir = './summary/'
        self.eval_period = 704

        self.train_image_dir = './data/train_images'
        self.val_image_dir = './data/val_images'
        self.test_image_dir = './data/test_images'
        self.test_result_dir = './data/test_result_images'
