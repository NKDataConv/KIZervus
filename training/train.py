import os
import datetime as dt
import numpy as np
import tensorflow as tf
from transformers import AutoConfig, TFAutoModelForSequenceClassification, create_optimizer, set_seed

from training_args import training_args
from data_transformations import load_data, get_tokenizer
from train_monitoring import get_callbacks, get_metrics


def get_loss_function():
    # return tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def get_model(metrics, length_data_train):
    config = AutoConfig.from_pretrained(
        training_args["MODEL_NAME"],
        num_labels=2,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
    )

    model = TFAutoModelForSequenceClassification.from_pretrained(
        training_args["MODEL_NAME"],
        config=config,
        from_pt=True
    )

    num_epochs = training_args["FREEZE_EPOCHS"]
    batches_per_epoch = length_data_train // training_args["BATCH_SIZE"]
    total_train_steps = int(batches_per_epoch * num_epochs)

    optimizer, schedule = create_optimizer(
        init_lr=training_args["LEARNING_RATE"], num_warmup_steps=0, num_train_steps=total_train_steps
    )

    loss_fn = get_loss_function()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    return model


def run_training():
    # setup logging directory
    now = dt.datetime.now().strftime("%Y%m%d%H%M")
    cur_dir = os.path.dirname(__file__)
    log_dir = os.path.join(cur_dir, "logs", now)
    model_dir = os.path.join(cur_dir, "models", now)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        tf.summary.text("Experiment description", training_args['DESCRIPTION'], step=0)

    # set seed in all libraries
    np.random.seed(training_args["SEED"])
    tf.random.set_seed(training_args["SEED"])
    set_seed(training_args["SEED"])

    tokenizer = get_tokenizer(training_args['MODEL_NAME'])
    data = load_data(tokenizer)

    length_data_train = len(data["train"])
    metrics = get_metrics()
    model = get_model(metrics, length_data_train)

    # model.layers[0].trainable = False

    callbacks = get_callbacks(log_dir=log_dir, model_dir=model_dir)

    model.fit(
        data["train"],
        validation_data=data["eval"],
        validation_steps=20,
        epochs=training_args["FREEZE_EPOCHS"],
        steps_per_epoch=training_args["STEPS_PER_EPOCH"], # use this only for testing purposes
        callbacks=callbacks
    )

    if training_args["PUSH_TO_HUB"]:
        model.push_to_hub("test-push-to-hub", use_temp_dir=True)
        tokenizer.push_to_hub("test-push-to-hub", use_temp_dir=True)

    output_folder = os.path.join(cur_dir, "models", f"{now}_export_model")
    output_folder_latest = os.path.join(cur_dir, "models", "latest")

    tokenizer.save_pretrained(output_folder)
    model.save_pretrained(output_folder)

    tokenizer.save_pretrained(output_folder_latest)
    model.save_pretrained(output_folder_latest)


if __name__ == '__main__':

    run_training()
    print("Training finished")
