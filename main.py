import tensorflow as tf

from model import Recognizer
from dataset import DataSet
from config import HyperParameters

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       '')
tf.flags.DEFINE_string('dataset_dir', '/dataset',
                       'Directory where the data needed for training is stored')
tf.flags.DEFINE_bool('load_checkpoint', True,
                     'Load previous checkpoint')
tf.flags.DEFINE_string('checkpoint', None,
                       'Path of checkpoint file')
tf.flags.DEFINE_string('summary_dir', '',
                       'Directory where to save summary data')


def main(args):
    hparams = HyperParameters()
    hparams.summary_dir = FLAGS.summary_dir if FLAGS.summary_dir else hparams.summary_dir

    if FLAGS.phase == 'train':
        train_dataset = DataSet(hparams.train_image_dir,
                                hparams.batch_size, [224, 224, 3],
                                len(hparams.license_number_list),
                                include_label=True,
                                shuffle=True,
                                augmented=True)
        val_dataset = DataSet(hparams.val_image_dir,
                              hparams.batch_size, [224, 224, 3],
                              len(hparams.license_number_list),
                              include_label=True,
                              shuffle=False,
                              augmented=False)
        # test_dataset = DataSet(hparams.test_image_dir,
        #                        hparams.batch_size, [224, 224, 3],
        #                        is_train=False,
        #                        shuffle=False,
        #                        augmented=False)

        with tf.Session() as sess:
            model = Recognizer(hparams,
                               trainable=True)
            model.train(sess,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        load_checkpoint=FLAGS.load_checkpoint,
                        checkpoint=FLAGS.checkpoint)
    elif FLAGS.phase == 'eval':
        test_dataset = DataSet(hparams.test_image_dir,
                               hparams.batch_size, [224, 224, 3],
                               len(hparams.license_number_list),
                               include_label=True,
                               shuffle=False,
                               augmented=False)
        with tf.Session() as sess:
            model = Recognizer(hparams,
                               trainable=True)
            model.eval(sess,
                       test_dataset,
                       checkpoint=FLAGS.checkpoint)
    else:
        test_dataset = DataSet(hparams.test_image_dir,
                              hparams.batch_size, [224, 224, 3],
                              len(hparams.license_number_list),
                              include_label=False,
                              shuffle=False,
                              augmented=False)
        with tf.Session() as sess:
            model = Recognizer(hparams,
                               trainable=True)
            model.test(sess,
                       test_dataset,
                       checkpoint=FLAGS.checkpoint)


if __name__ == '__main__':
    tf.app.run()