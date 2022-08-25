import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import Dataset, DatasetDict
from nlpaug.augmenter.word import ContextualWordEmbsAug

from training_args import training_args


def split_data(df: pd.DataFrame):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=training_args["SEED"])

    for train_index, temp_index in split.split(df, df[["label", "source"]]):
        df_train = df.loc[train_index]
        df_temp = df.loc[temp_index]

    df_temp = df_temp.reset_index()

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=training_args["SEED"])

    for test_index, eval_index in split.split(df_temp, df_temp[["label", "source"]]):
        df_eval = df_temp.loc[eval_index]
        df_test = df_temp.loc[test_index]

    cols = ["text", "label"]

    return df_train[cols], df_test[cols], df_eval[cols]


def load_data(tokenizer):

    cur_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(cur_dir, "..", "data", "data_train", "data.csv"))

    df = df.iloc[:1000, :]

    df_train, df_test, df_eval = split_data(df)

    ds_train = Dataset.from_pandas(df_train)
    ds_test = Dataset.from_pandas(df_test)
    ds_eval = Dataset.from_pandas(df_eval)

    ds = DatasetDict()

    ds['train'] = ds_train
    ds['test'] = ds_test
    ds['eval'] = ds_eval

    # remove index
    if '__index_level_0__' in ds["train"].column_names:
        ds = ds.map(remove_columns='__index_level_0__')

    # augment data
    if training_args["AUGMENT_DATA"]:
        # two kinds of augmentation
        aug_insert = ContextualWordEmbsAug(model_path="distilbert-base-german-cased", action="insert", aug_p=0.1)
        aug_sub = ContextualWordEmbsAug(model_path="distilbert-base-german-cased", action="substitute", aug_p=0.1)

        # augment with a factor such that both labels are approximately equally represented
        grouped = df.groupby("label").count()
        factor = grouped.text[0] / grouped.text[1]

        def _augment_text(batch):
            text_aug, label_aug = [], []
            for text, label in zip(batch["text"], batch["label"]):
                text_aug += [text]
                label_aug += [label]

                # augment only vulgar class
                if label == 1:

                    augmentation_count = int(factor) + np.random.binomial(n=1, p=factor % 1, size=1)[0] - 1

                    for _ in range(augmentation_count):
                        augmentor = np.random.choice([aug_insert, aug_sub])

                        new_text = augmentor.augment(text)
                        text_aug += new_text
                        label_aug += [label]

            return {"text": text_aug, "label": label_aug}

        ds = ds.map(_augment_text, batched=True, batch_size=training_args["BATCH_SIZE"])

    def _tokenize(row):
        result = tokenizer(row["text"], truncation=True)
        result["label"] = row["label"]
        return result

    ds = ds.map(_tokenize, batched=True, batch_size=training_args["BATCH_SIZE"])

    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")

    tf_data = dict()

    for key in ("train", "eval", "test"):
        if key == "train":
            drop_remainder = True  # Saves us worrying about scaling gradients for the last batch
        else:
            drop_remainder = False

        dataset = ds[key]
        data = dataset.to_tf_dataset(
            columns=[col for col in dataset.column_names if col not in {"label", "__index_level_0__"}],
            shuffle=True,
            batch_size=training_args["BATCH_SIZE"],
            collate_fn=data_collator,
            drop_remainder=drop_remainder,
            label_cols="label",
        )
        tf_data[key] = data

    return tf_data


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    return tokenizer
