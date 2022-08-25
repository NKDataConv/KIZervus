#!/bin/bash

python -m transformers.onnx --model=training/models/latest --feature=sequence-classification deployment