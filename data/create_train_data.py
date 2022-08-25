import os
import pandas as pd
import re


def remove_username_and_urls(text):

    # remove URLs
    text = re.sub(r'http\S+', '', text)

    # remove usernames (@username)
    text = re.sub('@[^\s]+', '', text)

    # remove " at beginning and end of text
    text = text.strip('"')

    # remove extra whitespace
    text = text.strip()

    return text


def create_train_data():
    directory = 'data_train'
    if not os.path.exists(directory):
        os.mkdir(directory)

    files = os.listdir('data_preprocessed')

    # files = ['fb-hate-speech.csv', 'germeval2018.csv', 'germeval2021.csv', 'rp-mod-crowd.csv', 'tweets-refugees.csv']
    #
    # if os.path.exists(os.path.join('data_preprocessed', 'covid.csv')):
    #     files.append('covid.csv')
    # if os.path.exists(os.path.join('data_preprocessed', 'telegram.csv')):
    #     files.append('telegram.csv')

    dfs = []
    for file in files:

        input_path = os.path.join('data_preprocessed', file)
        if file == "fb-hate-speech.csv":
            df = pd.read_csv(input_path, lineterminator='\n')
        else:
            df = pd.read_csv(input_path)
        df['source'] = file.split(".")[0]
        dfs.append(df)

    df = pd.concat(dfs)

    df["text"] = df["text"].map(remove_username_and_urls)
    df = df.dropna()
    mask_empty = df.text == ""
    df = df.loc[~mask_empty]
    df = df.drop_duplicates()

    output_path = os.path.join("data_train", "data.csv")
    df.to_csv(output_path, index=False)


if __name__ == '__main__':

    create_train_data()
    print("Creating train data finished.")