# SRGAN Introduction
Welcome to my SRGAN (Super-Resolution Generative Adversarial Network) repository!
SRGAN is a deep learning-based approach to perform single image super-resolution. It uses a generative adversarial network (GAN) to generate high-resolution images from low-resolution images. 

The SRGAN model is trained on pairs of low-resolution and high-resolution images, where the goal is to learn a mapping function from the low-resolution space to the high-resolution space. Once trained, the SRGAN model can be used to generate 128x128 sized images from 32x32 sized images.

In this repository, I'll provide an implementation of the SRGAN model using TensorFlow and Keras. The implementation includes the generator and discriminator models, as well as the training loop to train the models on your own data. This script was done through Google Colab.

# SRGAN Model
Starting off, a dataset was established and the resolution was be compressed to 32x32. So a mount to Google Drive was done with the code shown below.

![Mount](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/Mount.PNG)

The dataset was stored in a .zip so it was unzipped and used in the script.

![Unzip](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/Unzip.PNG)

Next, an import of the libraries as well as make directories were made to call later.

![Directories](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/Directories.PNG) 
![Libraries](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/Libraries.PNG)

This code defines a custom dataset class called "MyDataset" that loads and preprocesses images from a specified directory. The dataset class has two transforms: "train_transforms" and "high_res_transforms", which are used to augment the training data and resize the images to a high-resolution format, respectively.

The "getitem" method loads an image from the given directory, applies the specified transforms (if any), and returns a tuple containing two versions of the image: one with low resolution and the other with high resolution.

The "len" method returns the number of images in the dataset.

![Dataset](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/Dataset.PNG)

This section of code defines the generator and discriminator models that are used for the SRGAN algorithm. The generator model takes as input a low-resolution image and outputs a high-resolution image. It consists of a series of convolutional layers with a kernel size of 9 and strides of 1, followed by PReLU. The discriminator model takes as input a high-resolution image and outputs a binary classification score, indicating whether the image is real or fake. The model consists of a series of convolutional layers with a kernel size of 3 and batch normalization, followed by leaky ReLU activation functions. An example of the layers are shown below.

![Gen_Disc](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/Gen_Disc.PNG)
![Layers](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/Layers.PNG)

This script is part of the implementation of the SRGAN. The training parameters and the models are defined, and the optimization process is set up. The training parameters are defined such as the number of epochs, and batch size. The generator and discriminator were built then compiled and the model was created, which after it was compiled.

![Compile](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/CompileSRGAN.PNG)

This part is only if you saved the epochs of your model and you would like to load it again.

![Load](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/LoadingEpochs.PNG)

The next part is the training loop. Basically this trains the discriminator, generator and prints the progress. Then every 5 epochs it'll save the .h5 files. Some epochs are displayed below during training.

![Train](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/TrainingLoop.PNG)
![Epochs](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/Epochs.PNG)

Next the images that were produced by the SRGAN as well as the original high and low resolution images. As you can see the results were not the best, and I tried to greyscale it to maybe get rid of some noise. But it didn't go very well.

![RGB](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/DisplayRGB.PNG)
![RGB](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/Display1.PNG)
![Grey](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/DisplayGrey.PNG)
![Grey](https://github.com/jpham11/stunning-robot/blob/main/Images_SRGAN/Display2.PNG)


