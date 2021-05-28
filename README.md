# Insect-brain-connectome-based-DNN
Fruit fly Drosophila connectome-based deep neural network. This model is made up by a combination of CNN-RNN and mimics the visual system of Drosophila in detail following the information in the existing connectomes of the mentioned insect. The network preocesses the given frames and predicts the 3 dimensions position (x, y, z) of the object which appears in the frames.

Files and folders
>>	"connectome_CNN_RNN_training.py"
>>>	- This file contains the DNN model and it loads the data and trains the model. It also shows the performance and the prediction of the training data.
>>>	- "Seleccion del tipo de velocidad angular de la plataforma giratoria" menu allows to choose the stimulus angular velocity among constant, trapezoidal or sinusoidal.  
>>>	- "Considerar movimiento de la cabeza?" menu allows to choose whether you want to consider angular motion of the hypothetical "head" or consider it static  
>>>	- "Seleccion del tipo de velocidad angular de la cabeza en cada eje" menu will be available in the case you chose in the previous menu the option "con movimiento",	that is considering angular motion. Choose between "alabeo", "cabeceo", "guinada" or "combinado" for roll, pich, yaw or combined respectively.  
>>>	- "Considerar aceleracion lineal?" menu allows to choose whether you want to inroduce linear acceleration or not as stimulus.  
>>>	- Check "Ver respuesta de los otolitos" to show the otolith organs resulting plot.  
>>>	- Finally, press "Simular" to run your simulation.  
>>	- 2 Robot simulation  
>>>	- Introduce the desired parameters of the controller. the parameters used in the paper are:  
>>>> - Case 1: Kp = 2, Ki = 5, Ki1 = 5  
>>>> - Case 2: Kp = 2, Ki = 8, Ki1 = 8  
>>> - Check "Ver seguimiento de la posicion" to show the error angles yaw, pitch and roll resulting plots.  
>>> - Finally, press "Simular" to run your simulation.   

Figures folder  
>>	Figures obtained from the simulation and shown in the paper.
