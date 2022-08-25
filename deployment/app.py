from flask import Flask
from flask import request
from cachetools import cached, LFUCache

import numpy as np
from transformers import AutoTokenizer
from onnxruntime import InferenceSession

app = Flask(__name__)


@cached(cache=LFUCache(maxsize=1))
def _get_inference_moduls():

    tokenizer = AutoTokenizer.from_pretrained("./", local_files_only=True)
    # tokenizer = AutoTokenizer.from_pretrained("training/models/latest", local_files_only=True)
    session = InferenceSession("./model.onnx")
    # session = InferenceSession("deployment/model.onnx")

    return tokenizer, session


def softmax(logits):
    e = np.exp(logits)
    return np.round(e / e.sum(), 2)


@app.route('/api/v1/predict', methods=['GET', 'POST'])
def predict():

    # parse arguments
    text = request.args.get('text')
    text = text.replace("-", " ")
    boundary = request.args.get('decision_boundary')

    # get tokenizer and model (in session)
    tokenizer, session = _get_inference_moduls()

    # inference
    inputs = tokenizer(text, return_tensors="np")
    outputs = session.run(output_names=["logits"], input_feed=dict(inputs))

    # probabilities, Element at 0 is the probability of the non-vulgar class, Element at 1 is the probability of the vulgar class
    probabilities = softmax(outputs[0][0])

    return_dict = {"text": text, "score": str(probabilities[1])}

    # if a decision boundary is given, check if the probability for vulgar class is above the boundary
    if boundary:
        if probabilities[1] > float(boundary):
            return_dict["predicted_class"] = "vulgar"
            return return_dict
        else:
            return_dict["predicted_class"] = "non-vulgar"
            return return_dict

    # if no decision boundary is given, return the class which has the higher probability
    predicted_class = np.argmax(outputs[0], axis=-1)

    if predicted_class[0] == 1:
        return_dict["predicted_class"] = "vulgar"
        return return_dict
    return_dict["predicted_class"] = "non-vulgar"
    return return_dict


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
