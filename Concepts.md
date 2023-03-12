# Convolutional Layer
## Convolutions
- Neural networks are very useful when they are use to impose a prior belief. Prior belief is when it is believed that something is true about how the data/problem works prior to ever looking at the data
- When working with images, a prior should be added into the model: the spatial relationship. The prior that convolutions encode is that elements near each other are related, and elements far from each other have no relationship
- A convolution is a mathematical function with two inputs. Convolutions take an input image and a filter (also called a kernel) and output a new image. The goal is for the filter to recognize certain patterns from the input and highlight them in the output
- A convolution takes an image and filter and produces one output, where the filter is defining what kind of pattern to look for. The more convolutions there are, the more different patterns can be detected
  
  ![2d](https://user-images.githubusercontent.com/127037803/224548425-7ebd2026-3fe7-4100-a4f5-24dd73d3bd4e.png)
- The idea of a convolution stays the same, no matter the dimension of the input: slide a filter around the input, multiply the values in the filter with each area of the input, and then make the sum
## Convolutional Neural Network
- The mathematical notation used for describing a network with one convolutional hidden layer is similar to a fully connected layer

  $$f(x)=tanh(x\_{C\_{in}, W, H}\otimes W^{h\_1}\_{C\_{out}, C\_{in}, K, K}+b^{h\_1})W^{out}\_{(C\_{out}\*w\*h), C}+b^{out})$$

- The main thing that has changed in this equation is that the dot product is replaced by a convolution (a spatially linear operation denoted by $\otimes$)
- One can add more filters to the model, where $C_{out}$ represents the number of different filters (where each filter outputs one channel) and $C_{in}$ is used to indicate the number of channels in the input
- One issue occurs because the output of the convolution has a shape of $(C, W, H)$, but the linear layer ($W^{out}+b^{out}$) expects something of the shape $(C\*W\*H)$ — that is one dimension with all three dimensions collapsed into it
- Essentially, the output of the convolutions must be reshaped to remove the spatial interpretation so that the linear layer can process the result and compute some predictions

# Leaky ReLU
## Vanishing Gradient
- The tanh and sigmoid are two of the original activation functions used for neural networks, but they are not the only options because both of them can lead to a problem called vanishing gradient
- At the end of each layer, the activation $a$ is passed into a sigmoid function to produce the output vector of that layer: $h=\sigma(a)$. This means when calculating the derivative of the loss function with respect to the first weight vector, for instance, the derivative of $\sigma(a)$ is included for this and all consecutive layers
- Note that the derivative of $\sigma(a)$ is equal to $\sigma(a)(1-\sigma(a))$ and because that term is bound in the range $[0,1]$, the largest this value can get is if $\sigma(a)=0.5$, meaning $0.5(1-0.5)=0.25$ is the largest possible outcome
- Based on these calculations, consecutive sigmoid activation functions will decrease the the gradient of the loss function with respect to the first weight vector, diminishing the effect it can have on the loss function. This property also applies to the tanh() activation function

## Recitfied Linear Units
- The most common approach to fixing vanishing gradients is to use an activation function known as the rectified linear unit, which has a very simple definition

  $\text{ReLU}(x)=\text{max}(0,x)$

- That is all the ReLU does. If the input is positive, the return is unaltered. If the input is negative, the return is zero, instead. This means, for backpropagation, the derivative of these zero outputs is just 0, and for all other values the derivative is 1
- This is because the function is linear: $ReLU(x)=x$, and the derivative of a linear function is defined to be 1. This makes it a simple activation function, however this also means that ReLU often performs worse for very small networks
- This is because ReLU has no gradients for $x\leq0$, meaning some neurons will stop activating, which is not a problem if there are many other neurons; but if there are not enough extra neurons, this becomes a problem
- This again can be solved with a simple modification: instead of returning 0 for negative inputs, something else will be returned, leading to what is called the leaky ReLU
- The leaky ReLU takes a “leaking” factor $\alpha$, which is supposed to be small, usually in range $\alpha\in[0.01, 0.3]$

  $\text{LeakyReLU}(x)=\text{max}(\alpha*x,x)$

# Batch Normalization
## Normalization
- Usually, before feeding a dataset $X$ with $n$ rows and $d$ features into a machine-learning algorithm, the features are standardized in some way. This can be done by subtracting the mean $\mu$ and dividing by the standard deviation

  $$\mu=\frac {1}{n}\sum^n_{i=1}x_i$$

  $$\sigma=\sqrt{\epsilon+\frac {1}{n}\sum^n_{i=1}(\mu-x_i)^2}$$

  $$\hat X={...\frac{x_i-\mu}{\sigma},...}$$

- This results in the data $\hat X$ having a mean of zero and a standard deviation of 1. This is done because most algorithms are sensitive to the scale of the input data

## Normalization Layers
- For neural networks, the normalization process can be applied before every layer of a neural network. This can be done by either inserting the normalization layer at the beginning of every hidden layer block or in the middle of the block (before the activation function and after the linear layer, for instance)
- Normalization layers are applied at every layer with one extra trick: let the network learn how to scale the data instead of assuming that a mean of 0 and standard deviation of 1 are the best choices ($l$ denotes $l$ th layer):

  $$\frac{x_l-\mu_l}{\sigma_l}\*\gamma_l+\beta_l$$

- The first term remains the same but the crucial additions are $\gamma$, which lets the network change the scale of the data (change standard deviation), and $\beta$, which lets the network shift the data left/right (change average)
- Since the networks controls $\gamma$ and $\beta$, they are learned parameters and thus included in the set of parameters $\theta$. It is common to initialize $\gamma=\vec1$ and $\beta=\vec0$ to start with simple standardization

## Batch Normalization
- The most popular type of normalization is batch normalization (BN). BN is applied differently depending on the structure of the input data. When working with fully connected layers (dimension (B, D)), take the average and standard deviation of the feature values D over the B items in the batch
- Hence, it is normalized over the data features in a given batch. This means that $\mu, \sigma, \gamma, \beta$ have a shape of D, and each item in the batch is normalized by the mean and standardization of just that batch of data
