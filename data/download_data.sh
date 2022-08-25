#!/bin/bash

mkdir data_raw

# Download the data for GermEval-2018
wget https://raw.githubusercontent.com/uds-lsv/GermEval-2018-Data/master/germeval2018.training.txt -O data_raw/germeval2018training.txt
wget https://raw.githubusercontent.com/uds-lsv/GermEval-2018-Data/master/germeval2018.test.txt -O data_raw/germeval2018test.txt

# Download the data for GermEval-2021
wget https://raw.githubusercontent.com/germeval2021toxic/SharedTask/main/Data%20Sets/GermEval21_TrainData.csv -O data_raw/germeval2021training.csv
wget https://raw.githubusercontent.com/germeval2021toxic/SharedTask/main/Data%20Sets/GermEval21_TestData.csv -O data_raw/germeval2021test.csv

# Download the data for RP-Mod
wget https://zenodo.org/record/5291339/files/RP-Mod-Crowd.csv?download=1 -O data_raw/rp-mod-crowd.csv

# Download the data for Tweets refugees
wget https://raw.githubusercontent.com/UCSM-DUE/IWG_hatespeech_public/master/german%20hatespeech%20refugees.csv -O data_raw/tweets-refugees.csv

# Download the data for Facebook Hate Speech
wget http://ub-web.de/research/resources/fb_hate_speech_csv.zip -O data_raw/fb-hate-speech.zip
unzip data_raw/fb-hate-speech.zip -d data_raw
rm data_raw/fb-hate-speech.zip
mv data_raw/fb_hate_speech_csv/comments.csv data_raw/fb-hate-speech.csv
mv data_raw/fb_hate_speech_csv/annotated_comments.csv data_raw/annotation-fb-hate-speech.csv
rm -r data_raw/fb_hate_speech_csv