# Training

To start training, run the following command:

```bash
python train.py
```

The Training Script will automatically load the data from the data/data_train directory and prepare for training. To tune hyperparameters, you can edit the training_args.py file.

To monitor the training and look at other previous training runs, start tensorboading by running the following command (inside the training directory):

```bash
tensorboard --logdir=logs
```
