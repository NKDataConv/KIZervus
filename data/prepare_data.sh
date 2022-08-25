#!/bin/bash

source download_data.sh
python preprocess_data.py
python create_train_data.py