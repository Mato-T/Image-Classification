# Project Description
## Introduction
- This project is about classifying pictures of various fruits and vegetables. Most time I spent developing Deep Learning models was focused on Natural Language Processing but I wanted to expand my knowledge to computer vision so I created this project.
- The dataset is found on Kaggle using this URL: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition. I chose this dataset because it was created by scraping from Bing Image Search. This is the easiest (maybe too easy as explained later on) approach to collecting a diverse dataset and is probably the approach I would have used if no datasets were available.
- I have created two models for this dataset. One is created along with transformations to generalize it towards real-world data, while the other is just using the plain dataset, forming some kind of baseline. 
- Note that the following should be read along with the Python code found in this repository.
## Preparation and Analysis.
- As mentioned before, I have created a dataset that includes augmented images. In this case, I perform rotation, translation, and shearing on every single image. Since the dataset does not contain many images, I concatenated it with the original dataset to form a single dataset. 
- Here are some sample images from the dataset to get a feel for how the dataset is structured and how that influences any decisions to make about building the model. 

  ![image](https://user-images.githubusercontent.com/127037803/224535400-c6f7b9fb-9423-4401-996a-e634e0d040bd.png)
- As seen in the plot, the images vary alot. Some pictures look like stock images (high quality pictures with the object centered on a white background), some are images are taken from greater distance picturing the plant in nature (potentially introducing noise due to surrounding objects), and some display the object in great number on one image.
- This can present a problem for simple models since a lemon, for example, can be presented in very different ways. A lemon on a tree is very different from a collection of lemons on a table, which in turn is different from a stock image of a lemon cut open. Especially with a low sample size, it will be hard for simple models to detect patterns in the images.
- Moving on, lets look at the at the class distribution.

  ![image](https://user-images.githubusercontent.com/127037803/224536042-375f288d-33c2-4d16-a478-f4d3a68d9a29.png)
- Since the dataset contains many different classes, I decided to use descriptive measures instead of actual counts. As seen in this chart, the sample size seams to be evenly distributed among classes, so no need for applying specific sampling techniques.

## Training the Model
### Augmented Model
- As a model, I used several Residual BottleNeck blocks which help passing on the information from the gradient to earlier layers. They are composed of a combination of Convolutional layers, leaky ReLU activation functions and batch normalization (for more information, check out the Concept.md file)
- Usually, I prefer using ADAM as optimizer but I kept getting better results using Stochastic Gradient Descent along with momentum. As a learning rate scheduler, I used cosine annealing as it is my default choice and it worked best for me on this dataset.
- I used only 6 epochs of training since the dataset is relatively large (for me, as a single individual) and my GPU did not allow for larger batch sizes and I ran out of memory using anything above 32. Also, performance did not increase significantly.

## Evaluation
### Augmented Model
- Note that after training, I saved the model so that no progress is lost after closing the notebook. This is why I loaded it back in during evaluation. The first thing I wanted to see is what were the problems that my model is facing. Consider the following misclassifications:

  ![image](https://user-images.githubusercontent.com/127037803/224536759-32900466-062f-4468-b9c0-095d63de250e.png)
- As suspected, the model is not generalizing very well. It heavily relies on color as seen on the cabbage example. The colors seem to match the ones found in bananas (probably unripe bananas). However, due to the different forms and sizes presented in the training sample, the model is not able to focus more on the shape than just on color
- Some misclassifications are very understandable like the classification between sweet corn and corn. In addition, there is not a clear distinction between capsicum and bell pepper. Capsicum is the name of the genus that the plant comes from which includes sweet, bell, and hot peppers. Lastly, since the dataset was created crawling Bing images, misclassified images are included, such as the image of the logo of Apple Inc. being classified as an apple.
- Another intersting visualization is creating a confusion matrix along with a heatmap that shows how different fruits or vegetables are misclassified:

  ![image](https://user-images.githubusercontent.com/127037803/224537436-94ccdba3-838a-4f7e-a3c2-5ed51bd2f9a4.png)
- Surprisingly, the model seems to have no problem classifying most fruits and vegetables, which is strange since it does not seem to detect shape very well, considering the previous examples.
For this reason, I created my own "validation dataset" where I went on the Internet and picked images myself. I was scrolling further down to decrease the risk of picking images that were included in the training/test set. I focused on images where the fruit/vegetable was clearly seen and in their most common color. I chose images with distinct colors, where the model was like to predict the right class (e.g., cauliflower and eggplant) but also included images where the model must focus on the shape, not color (e.g., green vegetables, like lettuce, chillis, and cabbage). These are the predictions made:

  ![image](https://user-images.githubusercontent.com/127037803/224537709-df6516a4-cb4d-4b18-9543-cd8c19f84dc9.png)
- In this case, the model misclassified 5 out of 12, indicating that the model is heavily overfitting to the test and training data. As seen in the plot, the model has problems distinguishing between vegetables that share the same color (e.g., green). However, it did recognize the pear and it was able to distinguish between ginger and soy beans which share similar colors.





