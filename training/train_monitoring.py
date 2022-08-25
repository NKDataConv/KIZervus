import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from training_args import training_args


def get_metrics():
    metrics = ["accuracy"] #, tf.keras.metrics.AUC(from_logits=False)]
    return metrics


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.output_dir)


def get_callbacks(log_dir, model_dir, retrain: bool = False):

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    save_pretrained_callback = SavePretrainedCallback(output_dir=model_dir)

    callbacks = [tensorboard_callback, save_pretrained_callback]

    if not retrain:
        log_hparams = hp.KerasCallback(log_dir, training_args)
        callbacks.append(log_hparams)

    return callbacks