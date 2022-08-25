
training_args = {
    "DESCRIPTION": "Augment data with substitution and random insertion.",  # Insert a description of the experiment here. This will be saved and the logs and also shown in the tensorboard dashboard. It makes sense to change this for every training run.
    "SEED": 42,
    "MODEL_NAME": "distilbert-base-german-cased",  # this is the pretrained model from the huggingface model hub (https://huggingface.co/models). If you change this model, make sure it is the correct language.
    "BATCH_SIZE": 16,  # depending on your available memory, you can incrise this number
    "LEARNING_RATE": 5e-5,
    "FREEZE_EPOCHS": 2,  # normally two epochs are sufficient for fine tuning
    "UNFREEZE_EPOCHS": 1,
    "STEPS_PER_EPOCH": 2,  # only for testing purposes to speed up training
    "PUSH_TO_HUB": False,  # if True, the trained model will be pushed to the huggingface hub, given that you are registered and logged in.
    "AUGMENT_DATA": True,  # it True, data augmentation will be performed. Only to class vulgar data augmentation will be applied as this is the underrepresented class. Have a look at the data_transformations.py file for more details.
}
