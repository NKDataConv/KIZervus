import os
from transformers import AutoTokenizer, pipeline, TFAutoModelForSequenceClassification, DataCollatorWithPadding
import pandas as pd
import numpy as np
from datasets import Dataset

from data_transformations import split_data
from training_args import training_args
from train import get_loss_function

from typing import List


def get_predictions_huggingface(texts: List):

    model = TFAutoModelForSequenceClassification.from_pretrained("models/latest")
    tokenizer = AutoTokenizer.from_pretrained("models/latest")
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, framework="tf")

    preds = pipe(texts, return_all_scores=True)

    return preds


def evaluate_samples(df):
    """Predict with latest model and get loss for each sample.
    Input:
        df: pandas.DataFrame with columns "text" and "label"
    """
    model = TFAutoModelForSequenceClassification.from_pretrained("models/latest")
    tokenizer = AutoTokenizer.from_pretrained("models/latest")

    ds = Dataset.from_pandas(df)

    def _tokenize(row):
        result = tokenizer(row["text"], padding="max_length", truncation=True)
        result["label"] = row["label"]
        return result

    ds = ds.map(_tokenize, batched=True, batch_size=None)

    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")

    data = ds.to_tf_dataset(
        columns=[col for col in ds.column_names if col not in {"label", "__index_level_0__"}],
        shuffle=False,
        batch_size=training_args["BATCH_SIZE"],
        collate_fn=data_collator,
        drop_remainder=False,
        label_cols="label",
    )

    preds = model.predict(data)

    pred_class = np.argmax(preds["logits"], axis=-1)

    loss_fn = get_loss_function()

    losses = []
    for i, pred in enumerate(preds["logits"]):
        truth = df.reset_index().loc[i, "label"]
        losses.append(loss_fn(truth, pred).numpy())

    df["loss"] = losses
    df["pred_class"] = pred_class
    df = df.sort_values(by="loss", ascending=False)

    return df


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    df = pd.read_csv(os.path.join("..", "data", "data_train", "data.csv"))
    _, df_test, _ = split_data(df)

    df_test = df_test.iloc[:100, :]

    df = evaluate_samples(df_test)

    print(df[df.pred_class == 1])

    print(df)
