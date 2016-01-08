# mmd-caffe

This is the implementation of PAMI paper "Learning Transferable Visual Features with Very Deep Adaptation Networks". We fork the repository of version ID `c6414ea` from [Caffe](https://github.com/BVLC/caffe) and make our modifications. The main modifications are listed as follow:

- Change the label from a single integer to a tuple, which contains the origin label and an indicator to distinguish source and target
- Add mmd layer described in the paper to neuron layers
- Add entropy loss layer described in the paper to loss layers

In `models/google_net/amazon_to_webcam`, we give an example model based on GoogLeNet. In `data/office/amazon_to_webcam/*.txt`, we give an example to show how to prepare the train and test data file. Please note that in this task, `amazon` dataset is the source domain and `webcam` dataset is the target domain
