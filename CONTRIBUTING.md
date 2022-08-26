# Contributing

Thanks your for considering contributing to KIZervus.

You can contribute in the following ways:
- Extending data
- Run experiments and report your results by adding your training logs and model
- Finding and reporting bugs
- Contributing code to KIZervus by fixing bugs or implementing features
- Improving the documentation

### Extending data
You can add further data. Please add only data which has been labeled by more than one person (i.e. there are multiple annotators, for sound procedure of annotating see for example [here](https://aclanthology.org/2020.lrec-1.626/)). At the moment this repository has only german data. This may change in the future as multi language models might provide a benefit.

### Run experiments
Feel free to use this repository as a basis to run your own experiments. You can both submit your code for running experiments and the logs of your training run such that others can observe in detail which hyperparameters and setting were already tested and which impact on performance it had.
For the former, we recommend to either extend the existing funtionalities or to use dedicated jupyter notebook which you can push in the /notebook directory. For the latter case, please use the directory /training/logs and make sure your logs are readable by tensorboard. We recomment using or extending the existing tensorflow callbacks.

### Finding and reporting bugs
You are very welcome to report any sort of bug or issue. Please either open an issue in the [Github repository](https://github.com/NKDataConv/KIZervus/issues) or send an email to the [contact email](mailto:nk@data-convolution.de).

### Contributing code
You are welcome to fix bugs, fix open issues or other features you may find valuable for this project. Please push your code in a dedicated branch and give a meaningful commit message.

### Improving the documentation
You can complement documentation, add type hints or comments. Please push your changes in a dedicated branch.
