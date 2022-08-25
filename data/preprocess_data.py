import os
import json

import pandas as pd


def prepare_fb_hate_speech():
    input_path = os.path.join('data_raw', 'fb-hate-speech.csv')
    df = pd.read_csv(input_path)

    input_path = os.path.join('data_raw', 'annotation-fb-hate-speech.csv')
    df_annotations = pd.read_csv(input_path)

    # filter for comments which have been annotated as vulgar
    vulgar_comment_ids = df_annotations.comment_id.unique()
    df = df[df.comment_id.isin(vulgar_comment_ids)]

    df = df[["message"]]
    df["label"] = 1
    df = df.rename(columns={"message": "text"})

    output_path = os.path.join("data_preprocessed", "fb-hate-speech.csv")
    df.to_csv(output_path, index=False)


def prepare_tweets_refugees():
    input_path = os.path.join('data_raw', 'tweets-refugees.csv')
    df = pd.read_csv(input_path, names=["text", "expert_1", "expert_2", "rating"], header=0)

    def _tweets_refugees_label(row):
        if row["expert_1"] == "YES" and row["expert_2"] == "YES":
            return 1
        elif row["expert_1"] == "NO" and row["expert_2"] == "NO":
            return 0
        else:
            return -1

    df["label"] = df.apply(_tweets_refugees_label, axis=1)
    df = df[df.label.isin([0, 1])]
    df = df[["text", "label"]]

    output_path = os.path.join("data_preprocessed", "tweets-refugees.csv")
    df.to_csv(output_path, index=False)


def prepare_telegram_data():
    input_path = os.path.join('data_raw', 'telegram.txt')
    with open(input_path) as f:
        telegram_data = json.load(f)
    df = pd.DataFrame(telegram_data["messages"])

    def _telegram_label(row):
        if row["gold_label"] == "OFFENSIVE_ABUSIVE":
            return 1

        # if all are neutral
        elif all([val == "NEUTRAL" for val in row["raw_annotations"].values()]):
            return 0

        # no angreement on classification, corresponding rows will be deleted later
        return -1

    df["label"] = df.apply(_telegram_label, axis=1)
    # delete -1 rows
    df = df[df.label.isin([0, 1])]
    df = df[["text", "label"]]

    output_path = os.path.join("data_preprocessed", "telegram.csv")
    df.to_csv(output_path, index=False)


def prepare_covid():
    input_path = os.path.join('data_raw', 'covid.csv')
    df = pd.read_csv(input_path, sep="\t")

    df["label"] = df.label.map(lambda x: 1 if x == "abusive" else 0)
    df = df[["text", "label"]]

    output_path = os.path.join("data_preprocessed", "covid.csv")
    df.to_csv(output_path, index=False)


def prepare_germeval2018():

    def _germ_label(row):
        if row["cat1"] == "OTHER" and row["cat2"] == "OTHER":
            return 0
        else:
            return 1

    dfs = []
    for split in ["training", "test"]:
        input_path = os.path.join('data_raw', f'germeval2018{split}.txt')
        df = pd.read_csv(input_path, sep="\t",
                         names=["text", "cat1", "cat2"])
        df["label"] = df.apply(_germ_label, axis=1)
        df = df[["text", "label"]]
        dfs.append(df)
    df = pd.concat(dfs)

    output_path = os.path.join("data_preprocessed", "germeval2018.csv")
    df.to_csv(output_path, index=False)


def prepare_germeval2021():
    dfs = []
    for split in ["test", "training"]:
        input_path = os.path.join('data_raw', f'germeval2021{split}.csv')
        df = pd.read_csv(input_path)
        df = df.rename(columns={"comment_text": "text", "Sub1_Toxic": "label"})
        df = df[["text", "label"]]
        dfs.append(df)
    df = pd.concat(dfs)

    output_path = os.path.join("data_preprocessed", "germeval2021.csv")
    df.to_csv(output_path, index=False)


def prepare_rp_mod():
    input_path = os.path.join('data_raw', 'rp-mod-crowd.csv')
    df = pd.read_csv(input_path)

    def _rp_label(row):
        if row["Reject Newspaper"] == 1 or row["Reject Crowd"] == 1:
            return 1
        else:
            return 0

    df["label"] = df.apply(_rp_label, axis=1)
    df.rename(columns={"Text": "text"}, inplace=True)
    df = df[~df.text.isna()]
    df = df[["text", "label"]]

    output_path = os.path.join("data_preprocessed", "rp-mod-crowd.csv")
    df.to_csv(output_path, index=False)


if __name__ == '__main__':

    directory = 'data_preprocessed'
    if not os.path.exists(directory):
        os.mkdir(directory)

    # prepare all datasets and convert to consistent format with columns text and label
    prepare_fb_hate_speech()
    prepare_tweets_refugees()
    prepare_germeval2018()
    prepare_germeval2021()
    prepare_rp_mod()

    # if corresponding data is available
    if os.path.exists("data_raw/telegram.txt"):
        prepare_telegram_data()
    if os.path.exists("data_raw/covid.csv"):
        prepare_covid()
