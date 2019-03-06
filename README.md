# PyTorch Image Classifier

## Updates

As for many users request, I released a new version of standared pytorch immage classification example at here: http://codes.strangeai.pro/aicodes_detail.html?id=30

It's more standared with **pytorch 1.0** support. It contains those features which is really useful to write a standared AI application:

- standared dataload to load data;
- separate net work defination and data process code (less couple);
- catch keyboard interrupt and resume training.

## Less Than 200 Line Train Codes and 25 Epochs, Got 98% Accuracy!

In this repo, I managed classify images into 2 kinds which is ants and
bees, but it's also very straightforward to train on more classes images.
The amazing thing is, using PyTorch, we can use less than 200 line code to get
a very hight accuracy of classify on images!!!


## Usage

To using this repo, some things you should to know:

* Compatible both of CPU and GPU, this code can automatically train on CPU or GPU;
* Models trained on GPU can also predict on CPU using predict.py;
* First run please run `bash download_datasets.sh` to obtain datasets;
* Model will be save after epochs;
* Image Size can be set in `global_config.py`.

## Future Work

This version if fine tune on ResNet18, in the future maybe implement some own network,
also fine tune on famous networks.
As well as more datasets.

## Copyright

This repo implement by Jin Fagang, and ofcourse PyTorch authors.
if you have any question, you can find me on wechat: `jintianiloveu`
