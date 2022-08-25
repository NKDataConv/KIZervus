# KIZervus - **K**ünstliche **I**ntelligenz **z**ur **Er**kennung **vu**lgärer **S**prache
Artificial Intelligence for the detection of vulgar speech

This repository provides a framework for the development of an artificial intelligence system for the detection of vulgar speech in german. This repository contains three parts:
- **data** which is a collection of other sources of labeled text. You can find an overview of all sources [here](data/README.md).
- **training** which is a collection of scripts for the training of the artificial intelligence system. You can find an overview of all scripts [here](training/README.md).
- **deployment** which offers a simple way to deploy the artificial intelligence system as a REST API in a docker container.

# How to get started
Check out a quick demo of the latest model at huggingface.com: https://huggingface.co/KIZervus/KIZervus

There are three ways to get started with this repository.
- [Deploy the KIZervus model as REST API](#deploy) if you want to use the model in your own application.
- [Use the KIZervus Model in Python](#python) if you want to use the model in a Python application and further adapt to your own usecase.
- [Train your own model](#train) if you want to fine tune the model and adapt to your own data.
 

## <a name="deploy">Deploy the KIZervus model as a REST API</a> 
#### Prerequisite: 
- Docker installed

Run the following command to build the KIZervus Docker Container:

```bash
docker build . -t kizervus
```

Run the following command to start the KIZervus Docker Container:

```bash
docker run -p 80:80 -it kizervus
```
You can query the REST API at 
**http://localhost:80/api/v1/predict**
The REST API has the following parameters:
- **text**: the text to be classified whereby whitespaces are replaced by "-" and the text is encoded in UTF-8. Note that texts only up to a size of approximately 250 words can be classified.
- **decision_boundary** (optional): the decision boundary to be used for the classification. The default value is 0.5. Dependent on your use case, this parameter can be fine tuned. Internally a probability is calculated for each classification of non-vulgar vs vulgar text. Values close to 0 indicate a high probability for non-vulgar text whereas values close to 1 indicate a high probability for vulgar text. If you set the boundary to a low value (say 0.3) you lean towards classifying as vulgar text and the model is likely to classify texts which are non-vulgar into the vulgar category. Vice versa, if this threshold is set to a high value (say 0.7) you lean towards the non-vulgar class and only texts are classified into the category vulgar, if there is a high probability (>0.7).

Example calls:

```bash
http://127.0.0.1/api/v1/predict?text=Ich-liebe-dich
http://127.0.0.1/api/v1/predict?decision_boundary=0.6&text=Ich-liebe-dich
#http://127.0.0.1/api/v1/predict?text=Leck-mich-am-arsch
http://127.0.0.1/api/v1/predict?decision_boundary=0.6&text=Leck-mich-am-arsch
```

The response is a JSON object with the following fields:
- **text**: the original text
- **predicted_class**: the predicted class
- **score**: the probability score of the prediction. Values close to 0 indicate a high probability for non-vulgar text whereas values close to 1 indicate a high probability for vulgar text. The decision boundary for the predicted class is 0.5. A custom value can be added with the argument **decision_boundary**


## <a name="python">Import KIZervus model in python</a> 
Have a look at the Jupyter Notebook 'Python_Inference_Example.ipynb' for an example of how to use the KIZervus model in python.

## <a name="train"> Train your own model</a>
Install the required packages:

```bash
pip install -r requirements.txt
```

To train the model on your own data, there are basically two approaches:
1. Integrate data into the training process and improve the KIZervus model. To take this approach, follow the steps:
- Include your data in the data file. Follow the steps [here](data/README.md).
- Next, run the training script. This will automatically load and preprocess the data from data/data_train (where it will automatically be saved in the previous step). You can run the training script by issuing the following command:

```bash
python train.py
```

- We recommend to have a look into the training/training_args.py file. You can fine tune some parameters for the training process. For example, it is possible to push the trained model to the huggingface model hub by setting the parameter PUSH_TO_HUB to True. Be sure to change the description in training_args. This description will also be pushed to tensorboard and makes it easier to track the various training runs. Further explanations of the parameters are included in the comments in the training_args.py file.
- Your trained model will automatically be saved in training/models/latest and under your current timestamp in training/models. For deployment, you have to convert the model to a onnx file. This can be done by issuing the following command:

```bash
bash deployment/create_onnx_model.sh
```
- This will save the model in onnx file format under deployment. Now follow the steps [here](deploy) to deploy your model and serve as a REST API.

2. Use this model as a basis and fine tune on your data to get the best accuracy on your own data.
- Follow the steps [here](data/README.md) and make sure to skip all other data such that only your data file is placed in the data_raw folder. All other steps are still the same.
- Run the training script. This will automatically load and preprocess the data from data/data_train (where it will automatically be saved after following the previous step). You have to change the parameter 'MODEL_NAME' in training_args.py to 'KIZervus/KIZervus'. This is the model which was originally trained with this repository and will thus be used as a basis for fine tuning.
- After training, the model will be saved in training/models/latest and under your current timestamp in training/models. For deployment, you have to convert the model to an onnx file. This can be done by issuing the following command:

```bash
bash deployment/create_onnx_model.sh
```
- This will save the model in onnx file format under deployment. Now follow the steps [here](deploy) to deploy your model and serve as a REST API.


# Data
Contributors dissociate from all expressed opinions and speech contained in the data.
This repository uses a collection of data from other sources. For an overview of the data sources, have a look at the table in [data](data/README.md)

# How to contribute
- contribute data 
- run experiments and complement experiments

# License
This project is licensed under the [MIT license](https://opensource.org/licenses/MIT).

# Dependencies
To push your model to the huggingface hub, you will need to install git-lfs. Follow instructions here: https://git-lfs.github.com/

# Supporter

<p align="center">
  <img src="logo-bmbf.svg"/>
</p>