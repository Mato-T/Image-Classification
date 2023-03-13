# Project Description
## Introduction
- This project is about classifying images of different types of fruits and vegetables. Most of the time I spent developing deep learning models focused on natural language processing, but I wanted to extend my knowledge to computer vision, so I created this project.
- The dataset can be found on Kaggle at this URL: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition. I chose this dataset because it was created by scraping from Bing image search. This is the simplest (perhaps too simple, as explained later on) approach to collecting a diverse dataset, and it is probably the approach I would have used if no datasets were available.
- I created two models for this dataset. One was created along with augmentations to generalize it to real data, while the other uses the original dataset and forms a sort of baseline. 
- Please note that the following information should be read along with the Python code in this repository.
## Preparation and Analysis
- As mentioned earlier, I have created a dataset that contains augmented images. In this case, I am performing rotation, translation, and shear on each image. Since the dataset does not contain many images, I concatenated it with the original dataset to create a single dataset. 
- Here are some sample images from the dataset to get a feel for how the dataset is structured and how that affects the decisions that need to be made when building the model. 

  ![image](https://user-images.githubusercontent.com/127037803/224535400-c6f7b9fb-9423-4401-996a-e634e0d040bd.png)
- As seen in the plot, the images vary greatly. Some images look like stock images (high quality images with the object in the center on a white background), others are taken from a greater distance and show the plant in nature (possibly introducing noise from surrounding objects), and some show the object in large numbers in one image.
- This can present a problem for simple models, since a lemon, for example, can be shown in very different ways. A lemon on a tree is very different from a collection of lemons on a table, which in turn is different from a stock image of a lemon cut open. Especially with a small sample size, it will be difficult for simple models to detect patterns in the images.
- Ultimately, it depends on what application is being built. For example, for an app that classifies plants in nature, you would need images like the lemon tree rather than stock images of a lemon. In my case, I want to create a model that can classify fruits/vegetables no matter what. This may be an unrealistic expectation given the size of the data set, but it is definitely worth a try.
- Let us now look at the class distribution.

  ![image](https://user-images.githubusercontent.com/127037803/224536042-375f288d-33c2-4d16-a478-f4d3a68d9a29.png)
- Since the data set contains many different classes, I decided to use descriptive measures instead of actual counts. As seen in this diagram, the sample size appears to be evenly distributed among the classes, so there is no need to use special sampling techniques.

## Training the Model
### Model in Augmentation
- As a model, I used several Residual BottleNeck blocks that help pass information from the gradient to earlier layers. They consist of a combination of convolutional layers, leaky ReLU activation functions, and batch normalization. I have also included max-pooling for additional generalization (see the Concept.md file for more information).
- I usually prefer ADAM as the optimizer, but I have had better results with Stochastic Gradient Descent and Momentum. I used Cosine Annealing as the scheduler for the learning rate, as this is my default choice and worked best on this dataset.
- I only used 6 training epochs because the dataset is relatively large (for me, as a single individual) and my GPU does not allow for larger batch sizes as I was running out of memory using more than 32 samples per batch. Also, the performance did not improve much.
### Baseline
- The baseline model is similar to the model used in augmentation. It also uses the exact same Residual BottleNeck blocks. However, this baseline model lacked a more generalized data set, so I added dropout and max-pooling layers. I also experimented with different kernel sizes and number of filters to achieve better accuracy.
- In this case, ADAM is used as the optimizer instead of SGD, but the learning rate scheduler remains the same. I achieved slightly better accuracy with training (96%) than with the model used along augmentation (92%). This is not necessarily a good or bad sign, as the model is prone to overfitting.

## Evaluation
### Model in Augmentation
- Note that I saved the model after the training so that the progress would not be lost after closing the notebook. For this reason, I reloaded it during the evaluation. First, I wanted to see what problems my model was facing. Consider the following misclassifications:

  ![image](https://user-images.githubusercontent.com/127037803/224536759-32900466-062f-4468-b9c0-095d63de250e.png)
- As suspected, the model does not generalize very well. It relies heavily on color, as seen in the example of cabbage. The colors seem to match those of bananas (probably immature bananas). However, because of the different shapes and sizes in the training sample, the model is not able to focus more on shape than just color.
- Some misclassifications are very understandable, such as the distinction between corn and sweet corn. Also, there is no clear distinction between bell peppers and capsicums. Capsicum is the name of the genus from which the plant originates, which includes sweet, bell, and hot peppers. Since the dataset was created from Bing images, it also includes misclassified images, such as the Apple Inc. logo, which was classified as an apple.
- Another interesting visualization is the creation of a confusion matrix along with a heat map showing how different fruits or vegetables are misclassified:

  ![image](https://user-images.githubusercontent.com/127037803/224537436-94ccdba3-838a-4f7e-a3c2-5ed51bd2f9a4.png)
- Surprisingly, the model seems to classify most fruits and vegetables without any problems, which is strange since it doesn't seem to recognize the shape very well considering the previous examples.
For this reason, I created my own "validation dataset" using self-selected images off the Internet. I scrolled down further to reduce the risk of selecting images that were included in the training/test set. I focused on images where the fruit/vegetable was clearly visible and in the most common color (e.g., yellow banana, red tomato, etc.). I chose vegetables with unique colors where the model could predict the correct class (e.g., cauliflower and eggplant), but also images where the model needs to focus on shape rather than color (e.g., green vegetables such as lettuce, chilies, and cabbage). These are the predictions made:

  ![image](https://user-images.githubusercontent.com/127037803/224537709-df6516a4-cb4d-4b18-9543-cd8c19f84dc9.png)
- In this case, the model misclassified 5 out of 12, indicating that the model is overfitting the test and training data. As seen in the plot, the model has trouble distinguishing between vegetables that are of the same color (e.g., green). However, it did recognize the pear and was able to distinguish between ginger and soybeans, which have similar colors.

### Baseline
- When evaluated, the model appears to have the same number of misclassifications as the model used in augmentation (6% misclassifications). Let's now take a look at some of the misclassifications:

  ![image](https://user-images.githubusercontent.com/127037803/224541167-b7dcdc62-966c-4914-8208-f97ee3cd9be9.png)
- As mentioned earlier, the dataset contains some inappropriate images, so some of the misclassifications are understandable. Again, high confidence misclassifications occur when distinguishing sweet corn from corn. However, predictions such as watermelon for chili pepper and eggplant for bell pepper appear to be way off. However, the confusion matrix shows good results:

  ![image](https://user-images.githubusercontent.com/127037803/224541443-7955d97d-5321-449a-8824-c26e398a28f7.png)
- All these factors lead to the conclusion that the model strongly overfits the training/test data. The following figure shows the performance on images that I selected myself:
  
  ![image](https://user-images.githubusercontent.com/127037803/224541598-452d20ef-c9e6-4c4a-a52c-0ae2f30dad30.png)
- The results also confirm the hypothesis. The model seems to have major problems when it comes to classifying images that it has never seen before. It does make some predictions that are correct, but it still performs worse than the model used with augmented data. 


## Conclusion
- While I suppose it's naive to think that I can create a well-performing fruit/vegetable classifier by randomly taking a few images from the Web, it was a great opportunity to learn more about data augmentation and how to approach any computer vision task.
- As it turns out, data augmentation can be a great tool to generalize a model and expand the data set when there is little labeled data. Although the model used in augmentation performs worse in training and testing, it is still an improvement over the base model when it comes to data it has never seen.
- I think this model can be further improved by collecting more data and tuning the hyperparameters. However, my computational resources were limited and it took a lot of time to train and experiment with different models and hyperparameters.






