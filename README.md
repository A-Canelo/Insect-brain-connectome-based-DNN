# Insect-brain-connectome-based-DNN
Fruit fly Drosophila connectome-based deep neural network. This model is made up by a combination of CNN-RNN and mimics the visual system of Drosophila in detail following the information in the existing connectomes of the mentioned insect. The network preocesses the given frames and predicts the 3 dimensions position (x, y, z) of the object which appears in the frames.

Files and folders
>>	"connectome_CNN_RNN_training.py"
>>>	- This file contains the DNN model and it loads the data and trains the model. It also shows the performance of the training and the prediction of the training data.
>>	"connectome_CNN_RNN_evaluation.py"
>>>	- This file loads the data and the pre-trained model. Then it runs the simulation with the test data and shows the prediction and performance. It also shows the convolution filter initial weights and the learned weights.
>>	"data"
>>>	- This folder contains the file "DAVIS_CNNRNN_data.mat", which is the preprocessed DAVIS 2016 dataset (https://davischallenge.org/), including the labels for the training and evaluation files.
>>	"connectome_model_CNNRNN_v3"
>>>	- This folder contains the pre-trained model. The model is loaded by the "connectome_CNN_RNN_evaluation.py" file mentioned above.
