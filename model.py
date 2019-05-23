import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import NeuralLayers, Optimizer


class Recognizer:
    def __init__(self,
                 hparams,
                 trainable):

        self.trainable = trainable
        self.hparams = hparams
        self.image_shape = [224, 224, 3]
        self.license_number_list = [
            '0123456789',
            '0123456789',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            '0123456789',
            '0123456789',
            '0123456789',
            '0123456789',
            '0123456789'
        ]
        self.is_train = tf.placeholder_with_default(False, shape=[], name='is_train')

        self.layers = NeuralLayers(trainable=self.trainable,
                                   is_train=self.is_train,
                                   hparams=self.hparams)
        self.optimizer_builder = Optimizer(hparams=hparams)
        self.saver = None
        self.build_resnet50()
        if trainable:
            self.build_optimizer()
            self.build_metrics()
            self.build_summary()

    def build_resnet50(self):
        hparams = self.hparams

        images = tf.placeholder(dtype=tf.float32,
                                shape=[None] + self.image_shape)

        conv1_feats = self.layers.conv2d(images,
                                         filters=64,
                                         kernel_size=(7, 7),
                                         strides=(2, 2),
                                         activation=None,
                                         name='conv1')
        conv1_feats = self.layers.batch_norm(conv1_feats, 'bn_conv1')
        conv1_feats = tf.nn.relu(conv1_feats)
        pool1_feats = self.layers.max_pool2d(conv1_feats,
                                             pool_size=(3, 3),
                                             strides=(2, 2),
                                             name='pool1')

        res2a_feats = self.identity_block_with_output_reduced(pool1_feats, 'res2a', 'bn2a', 64, (1, 1))
        res2b_feats = self.identity_block(res2a_feats, 'res2b', 'bn2b', 64)
        res2c_feats = self.identity_block(res2b_feats, 'res2c', 'bn2c', 64)

        res3a_feats = self.identity_block_with_output_reduced(res2c_feats, 'res3a', 'bn3a', 128)
        res3b_feats = self.identity_block(res3a_feats, 'res3b', 'bn3b', 128)
        res3c_feats = self.identity_block(res3b_feats, 'res3c', 'bn3c', 128)
        res3d_feats = self.identity_block(res3c_feats, 'res3d', 'bn3d', 128)

        res4a_feats = self.identity_block_with_output_reduced(res3d_feats, 'res4a', 'bn4a', 256)
        res4b_feats = self.identity_block(res4a_feats, 'res4b', 'bn4b', 256)
        res4c_feats = self.identity_block(res4b_feats, 'res4c', 'bn4c', 256)
        res4d_feats = self.identity_block(res4c_feats, 'res4d', 'bn4d', 256)
        res4e_feats = self.identity_block(res4d_feats, 'res4e', 'bn4e', 256)
        res4f_feats = self.identity_block(res4e_feats, 'res4f', 'bn4f', 256)

        res5a_feats = self.identity_block_with_output_reduced(res4f_feats, 'res5a', 'bn5a', 512)
        res5b_feats = self.identity_block(res5a_feats, 'res5b', 'bn5b', 512)
        res5c_feats = self.identity_block(res5b_feats, 'res5c', 'bn5c', 512)

        global_avg_pool = self.layers.global_avg_pool2d(res5c_feats,
                                                        keepdims=False,
                                                        name='global_avg_pool')
        global_avg_pool = self.layers.dropout(global_avg_pool,
                                              name='global_avg_pool_dropout')

        logits = []
        probabilities = []
        predictions = []
        for i, num_list in enumerate(self.license_number_list):
            logit = self.layers.dense(global_avg_pool,
                                      units=len(num_list),
                                      activation=None,
                                      name='num_{}'.format(i))
            probability = tf.nn.softmax(logit)
            prediction = tf.argmax(probability, axis=1)

            logits.append(logit)
            probabilities.append(probability)
            predictions.append(prediction)

        self.images = images
        self.logits = logits
        self.probabilities = probabilities
        self.predictions = predictions

    def identity_block_with_output_reduced(self, inputs, name1, name2, filters, strides=(2, 2)):
        """ A basic block of ResNet. """
        branch1_feats = self.layers.conv2d(inputs,
                                       filters=4 * filters,
                                       kernel_size=(1, 1),
                                       strides=strides,
                                       activation=None,
                                       use_bias=False,
                                       name=name1 + '_branch1')
        branch1_feats = self.layers.batch_norm(branch1_feats, name2 + '_branch1')

        branch2a_feats = self.layers.conv2d(inputs,
                                        filters=filters,
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        activation=None,
                                        use_bias=False,
                                        name=name1 + '_branch2a')
        branch2a_feats = self.layers.batch_norm(branch2a_feats, name2 + '_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.layers.conv2d(branch2a_feats,
                                        filters=filters,
                                        kernel_size=(3, 3),
                                        strides=strides,
                                        activation=None,
                                        use_bias=False,
                                        name=name1 + '_branch2b')
        branch2b_feats = self.layers.batch_norm(branch2b_feats, name2 + '_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.layers.conv2d(branch2b_feats,
                                        filters=4 * filters,
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        activation=None,
                                        use_bias=False,
                                        name=name1 + '_branch2c')
        branch2c_feats = self.layers.batch_norm(branch2c_feats, name2 + '_branch2c')

        outputs = branch1_feats + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def identity_block(self, inputs, name1, name2, filters):
        """ Another basic block of ResNet. """
        branch2a_feats = self.layers.conv2d(inputs,
                                        filters=filters,
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        activation=None,
                                        use_bias=False,
                                        name=name1 + '_branch2a')
        branch2a_feats = self.layers.batch_norm(branch2a_feats, name2 + '_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.layers.conv2d(branch2a_feats,
                                        filters=filters,
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        activation=None,
                                        use_bias=False,
                                        name=name1 + '_branch2b')
        branch2b_feats = self.layers.batch_norm(branch2b_feats, name2 + '_branch2b')
        branch2b_feats = tf.nn.relu(branch2b_feats)

        branch2c_feats = self.layers.conv2d(branch2b_feats,
                                        filters=4 * filters,
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        activation=None,
                                        use_bias=False,
                                        name=name1 + '_branch2c')
        branch2c_feats = self.layers.batch_norm(branch2c_feats, name2 + '_branch2c')

        outputs = inputs + branch2c_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def identity_block_without_bottleneck(self, inputs, name1, name2, filters):
        """ Another basic block of ResNet. """
        branch2a_feats = self.layers.conv2d(inputs,
                                        filters=filters,
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        activation=None,
                                        use_bias=False,
                                        name=name1 + '_branch2a')
        branch2a_feats = self.layers.batch_norm(branch2a_feats, name2 + '_branch2a')
        branch2a_feats = tf.nn.relu(branch2a_feats)

        branch2b_feats = self.layers.conv2d(branch2a_feats,
                                        filters=filters,
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        activation=None,
                                        use_bias=False,
                                        name=name1 + '_branch2b')
        branch2b_feats = self.layers.batch_norm(branch2b_feats, name2 + '_branch2b')

        outputs = inputs + branch2b_feats
        outputs = tf.nn.relu(outputs)
        return outputs

    def se_block(self, inputs, filters, name, ratio=16):
        avgpool = self.layers.global_avg_pool2d(inputs=inputs,
                                                keepdims=False,
                                                name=name + '_avgpool')
        dense1 = self.layers.dense(inputs=avgpool,
                                   units=filters / ratio,
                                   activation=tf.nn.relu,
                                   name=name + '_dense')
        weighted = self.layers.dense(inputs=dense1,
                                     units=filters,
                                     activation=tf.nn.sigmoid,
                                     name=name + '_weighted')
        weighted = tf.reshape(weighted, (-1, 1, 1, filters))
        outputs = tf.multiply(inputs, weighted)
        return outputs

    def build_optimizer(self):
        hparams = self.hparams

        global_step = tf.train.get_or_create_global_step()

        labels = tf.placeholder(dtype=tf.int64, shape=[None, len(self.license_number_list)])

        min_len = np.min([len(n) for n in self.license_number_list])
        losses = []
        for i, num_list in enumerate(self.license_number_list):
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels[:, i],
                                                          logits=self.logits[i])
            weight = len(num_list) / min_len
            loss = weight * loss
            losses.append(loss)

        cross_entropy_loss = tf.add_n(losses)

        regularization_loss = tf.losses.get_regularization_loss()

        total_loss = cross_entropy_loss + regularization_loss

        learning_rate = self.optimizer_builder.compute_learning_rate(global_step)

        optimizer = self.optimizer_builder.build(name=hparams.optimizer,
                                                 learning_rate=learning_rate)

        gradients, variables = zip(*optimizer.compute_gradients(total_loss, ))
        gradients, _ = tf.clip_by_global_norm(gradients, hparams.clip_gradients)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.apply_gradients(zip(gradients, variables),
                                             global_step=global_step)
        train_op = tf.group([train_op, update_ops])

        self.global_step = global_step
        self.labels = labels
        self.cross_entropy_loss = cross_entropy_loss
        self.regularization_loss = regularization_loss
        self.total_loss = total_loss
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.train_op = train_op

    def build_metrics(self):

        avg_cross_entropy_loss, avg_cross_entropy_loss_op = tf.metrics.mean_tensor(self.cross_entropy_loss)
        avg_reg_loss, avg_reg_loss_op = tf.metrics.mean_tensor(self.regularization_loss)
        avg_total_loss, avg_total_loss_op = tf.metrics.mean_tensor(self.total_loss)

        predictions = tf.stack(self.predictions, axis=1)
        partial_accuracy, partial_accuracy_op = tf.metrics.accuracy(labels=self.labels,
                                                                    predictions=predictions)
        matches = tf.reduce_all(tf.equal(self.labels, predictions), axis=1)
        accuracy, accuracy_op = tf.metrics.accuracy(labels=tf.ones_like(matches),
                                                    predictions=matches)

        self.metrics = {'cross_entropy_loss': avg_cross_entropy_loss,
                        'regularization_loss': avg_reg_loss,
                        'total_loss': avg_total_loss,
                        'partial_accuracy': partial_accuracy,
                        'accuracy': accuracy}
        self.metric_ops = {'cross_entropy_loss': avg_cross_entropy_loss_op,
                           'regularization_loss': avg_reg_loss_op,
                           'total_loss': avg_total_loss_op,
                           'partial_accuracy': partial_accuracy_op,
                           'accuracy': accuracy_op}

        self.metric_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
        self.reset_metric_op = tf.variables_initializer(self.metric_vars)

    def build_summary(self):

        with tf.name_scope('metric'):
            for metric_name, metric_tensor in self.metrics.items():
                tf.summary.scalar(metric_name, metric_tensor)

        with tf.name_scope('hyperparam'):
            tf.summary.scalar('learning_rate', self.learning_rate)

        self.summary = tf.summary.merge_all()

    def cache_metric_values(self, sess):
        metric_values = sess.run(self.metric_vars)
        self.metric_values = metric_values

    def restore_metric_values(self, sess):
        for var, value in zip(self.metric_vars, self.metric_values):
            sess.run(var.assign(value))

    def map_labels(self, labels):
        mapped_labels = []
        for label in labels:
            mapped_label = []
            for i, num in enumerate(label):
                assert len(label) == len(self.license_number_list)
                idx = self.license_number_list[i].index(num)
                mapped_label.append(idx)
            mapped_labels.append(mapped_label)
        mapped_labels = np.array(mapped_labels)

        return mapped_labels

    def prediction_decode(self, predictions):
        predictions = np.column_stack(predictions)
        decoded_predictions = []
        for prediction in predictions:
            decoded_prediction = []
            for i, num_idx in enumerate(prediction):
                decoded_prediction.append(self.license_number_list[i][num_idx])
            decoded_prediction = ''.join(decoded_prediction)
            decoded_predictions.append(decoded_prediction)

        return decoded_predictions

    def train(self, sess, train_dataset, val_dataset, test_dataset=None, load_previous=False):
        hparams = self.hparams

        if not os.path.exists(hparams.summary_dir):
            os.mkdir(hparams.summary_dir)
        train_writer = tf.summary.FileWriter(hparams.summary_dir + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(hparams.summary_dir + '/val')
        if test_dataset is not None:
            test_writer = tf.summary.FileWriter(hparams.summary_dir + '/test')

        train_fetches = {'train_op': self.train_op,
                         'global_step': self.global_step}
        train_fetches.update(self.metric_ops)
        val_fetches = self.metric_ops

        if test_dataset is not None:
            test_fetches = self.metric_ops

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if load_previous:
            self.load(sess)

        # Training
        for _ in tqdm(range(self.hparams.num_epochs), desc='epoch'):
            for _ in tqdm(range(train_dataset.num_batches), desc='batch', leave=False):
                images, labels = train_dataset.next_batch()
                labels = self.map_labels(labels)

                feed_dict = {self.images: images,
                             self.labels: labels,
                             self.is_train: True}

                train_record = sess.run(train_fetches, feed_dict=feed_dict)

                tqdm.write("Train step {}: total loss: {:>10.5f}   partial accuracy: {:8.2f}   accuracy: {:8.2f}"
                           .format(train_record['global_step'],
                                   train_record['total_loss'],
                                   train_record['partial_accuracy'] * 100,
                                   train_record['accuracy'] * 100))
                if train_record['global_step'] % hparams.summary_period == 0:
                    summary = sess.run(self.summary)
                    train_writer.add_summary(summary, train_record['global_step'])

                # Validation
                if (train_record['global_step'] + 1) % hparams.eval_period == 0:
                    self.cache_metric_values(sess)
                    sess.run(self.reset_metric_op)
                    for _ in tqdm(range(val_dataset.num_batches), desc='val', leave=False):
                        images, labels = val_dataset.next_batch()
                        labels = self.map_labels(labels)

                        feed_dict = {self.images: images,
                                     self.labels: labels}

                        val_record = sess.run(val_fetches, feed_dict=feed_dict)

                    tqdm.write("Validation step {}: total loss: {:>10.5f}   partial accuracy: {:8.2f}   accuracy: {:8.2f}"
                               .format(train_record['global_step'],
                                       val_record['total_loss'],
                                       val_record['partial_accuracy'] * 100,
                                       val_record['accuracy'] * 100))
                    summary = sess.run(self.summary)
                    val_writer.add_summary(summary, train_record['global_step'])
                    val_writer.flush()
                    val_dataset.reset()

                    self.restore_metric_values(sess)

            sess.run(self.reset_metric_op)

            self.save(sess, global_step=train_record['global_step'])

            train_dataset.reset()

        train_writer.close()
        val_writer.close()

        # Testing
        if test_dataset is not None:
            sess.run(self.reset_metric_op)
            for _ in tqdm(range(test_dataset.num_batches), desc='testing', leave=False):
                images, labels = val_dataset.next_batch()
                labels = self.map_labels(labels)

                feed_dict = {self.images: images,
                             self.labels: labels}

                test_record = sess.run(test_fetches, feed_dict=feed_dict)

            tqdm.write("Testing: total loss: {:>10.5f}   partial accuracy: {:8.2f}   accuracy: {:8.2f}"
                       .format(test_record['total_loss'],
                               test_record['partial_accuracy'] * 100,
                               test_record['accuracy'] * 100))
            summary = sess.run(self.summary)
            test_writer.add_summary(summary, train_record['global_step'])
            test_writer.flush()
            test_writer.close()

    def eval(self, sess, config, eval_data, eval_name='/eval', load_checkpoint=False):

        hparams = self.hparams

        if not os.path.exists(config.summary_dir):
            os.mkdir(config.summary_dir)
        eval_writer = tf.summary.FileWriter(config.summary_dir + eval_name, sess.graph)

        eval_dataset = self.build_dataset(eval_data)
        iterator = eval_dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()

        num_eval_batches = len(eval_data) // hparams.batch_size + 1

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if type(load_checkpoint) == str:
            self.load(sess, path=load_checkpoint)
        elif load_checkpoint:
            self.load(sess)

        sess.run(self.reset_metric_op)
        for _ in tqdm(range(num_eval_batches), desc='eval', leave=False):
            images, labels = sess.run(next_batch)

            feed_dict = {self.inputs: images,
                         self.labels: labels}

            eval_record = sess.run(self.metric_ops, feed_dict=feed_dict)

        summary, global_step = sess.run([self.summary, self.global_step])

        tqdm.write("Evaluation at step {}: loss: {:>10.5f}   {}: {:8.2f}%".format(global_step,
                                                                                  eval_record['loss'],
                                                                                  list(self.metrics)[1],
                                                                                  eval_record[list(self.metrics)[1]]))
        eval_writer.add_summary(summary, global_step)
        eval_writer.flush()
        eval_writer.close()

    def test(self, sess, test_dataset):
        hparams = self.hparams

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        self.load(sess)

        # Testing
        for _ in tqdm(range(test_dataset.num_batches), desc='batch', leave=False):
            images = test_dataset.next_batch()

            predictions = sess.run(self.predictions, feed_dict={self.images: images})

            predictions = self.prediction_decode(predictions)

            for image, prediction in zip(images, predictions):
                plt.imshow(image)
                plt.title(prediction)
                plt.savefig('{}/{}.jpg'.format(hparams.test_result_dir, prediction))

    def save(self, sess, path=None, global_step=None):
        if self.saver is None:
            self.saver = tf.train.Saver()
        path = path or self.hparams.save_dir
        global_step = global_step or self.global_step.eval(session=sess)

        self.saver.save(sess, path + '/recognizer-model.ckpt', global_step=global_step)

    def load(self, sess, path=None):
        if self.saver is None:
            self.saver = tf.train.Saver()
        path = path or self.hparams.save_dir
        latest_checkpoint = tf.train.latest_checkpoint(path)
        if latest_checkpoint is not None:
            self.saver.restore(sess, latest_checkpoint)
