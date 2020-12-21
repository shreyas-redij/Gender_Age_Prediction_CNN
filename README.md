<h2>Gender-and-Age-Prediction</h2>

**Description**

Convolutional Neural Networks are heavily used for image classification tasks. 
Here, we are using VGG-16 for Gender Classification.

**Dependencies:**

We are creating a virtualenv and loading neceassary libraires.

Tensorflow==2.3.0
opencv-python>=4.2.0.34
opencv-contrib-python>=4.2.0.34
numpy>=1.18.3
h5py>=2.10.0
matplotlib>=3.2.1


**DataSet:**

This dataset contains real world images with following Specifications

Statistics and info
Total number of photos: 26,580
Total number of subjects: 2,284
Number of age groups / labels: 8 (0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60-)
Gender labels: Yes
In the wild: Yes
Subject labels: Yes

Download the Entire Adience Dataset using the following link :

http://www.openu.ac.il/home/hassner/Adience/data.html

**Steps to run the project:**

1. Data Preparation:
		Data_Generation_Adience.ipynb -> This File converts our data to H5 file named adience_1.h5

2. Model Conversion for Transfer Learning:
		ModelConversion_VGG.ipynb-> Converts VGG face model to tensorflow format named vgg_face_weights.h5 

3. Training:
		For Training I have used the Google Colab GPU. I have uploaded the weights and the model to a google drive and then read them into a Google Colab Notebook.

		VGG_Train.ipynb->This file loads the pretrained weights and data obtained from previous steps. It then further trains the model and saves it.
		
		VGG_Architecture.h5 has both the weights and model architecture. Final_model.h5 has just the saved weights. 

4. Evaluation:
		VGG_Train.ipynb file also has scripts to evaluate the model accuracy and loss. It also plots the training graphs for accuracy and loss with respect to epochs.

5. Prediction:
		Perdict.ipynb file has predict function that returns prediction after taking images as an input.
		
**Results**

![alt text](https://github.com/shreyas-redij/Gender_Age_Prediction_CNN/blob/master/Images/Lady.JPG)

![alt text](https://github.com/shreyas-redij/Gender_Age_Prediction_CNN/blob/master/Images/Oldman.JPG)

![alt text](https://github.com/shreyas-redij/Gender_Age_Prediction_CNN/blob/master/Images/Sachin.JPG)


**References:**
1. Model Conversion from matlab to tensorflow https://sefiks.com/2019/07/15/how-to-convert-matlab-models-to-keras/
2. Deep Learning for face recognition https://sefiks.com/2018/08/06/deep-facte-recognition-with-keras/
3. VGG face literature http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
4. Adience dataset https://talhassner.github.io/home/projects/Adience/Adience-data.html
5. VGG in Keras https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
6. Transfer Learning for CNNs https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/	
7. Google colab for Deep Learning https://medium.com/analytics-vidhya/google-collab-deep-learning-keras-with-hdf5-must-know-for-those-who-use-collab-for-building-img-e5aa2f6ef4fd






