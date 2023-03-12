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

# Residual Connections
## Skip Connections
- With a normal feed-forward network, an output from one layer goes directly to the next layer. With skip connections, this is still true, but the next layer is also skipped and connected to a preceding layer as well

  ![skip](https://user-images.githubusercontent.com/127037803/224552290-e33e5684-12bd-4533-881a-c555b2d66180.png)
- The right two diagrams show two different ways to implement skip connections. The black dots on the connections indicate concatenation of the outputs (works with other kind of layers as well)
- So if $x$ and $h$ connect in the diagram, they input to the next layer: $[x, h]$. That way, the two inputs $x$ and $h$ have shapes (B, D) and (B, H). These features shall be stacked so the result will have shape (B, D + H)
- Every operation done makes the network more complex, but also makes the gradient more complex, creating a tradeoff between depth (capacity) and learnability (optimization). Skip connections create a short path with fewer operations, and can make the gradient easier to learn as exploding or vanish gradients are less likely to occur

## 1x1 Convolutions
- Convolutions are used to capture information about the relationship of values near each other. As such, the convolutions have a kernel with some size k so information about [k/2] neighbors
- However, when k is set to 1, no information about the neighbors is obtained and thus no spatial information is captured
- One particular application, where this would be useful, is to change the number of channels at a given layer. In the previous section, a Conv2d layer was inserted to convert the number of channels C into a more convenient value
- This is possible because there are $C_{in}$ input channels and $C_{out}$ output channels when performing a convolution. So a convolution with $k=1$ is looking not at a spatial neighbors but at spatial channels by grabbing a stack of $C_{in}$ values and processing them all at once

  ![1x1](https://user-images.githubusercontent.com/127037803/224552210-89493c44-382a-43fa-b8b9-ffe825ff7286.png)

## Residual Bottlenecks
- The residual layer is a simple extension of the skip connection idea that works by making the short path do as little work as possible to help with the gradient flow and minimize noise
- 1 x 1 convolutions are a fast and practical way to change the number of channels in the input without  changing its size. Shrinking and then expanding is the general idea behind residual bottlenecks. There are two main reasons for implementing a bottleneck
- The first is a design choice, where the original authors wanted to make their networks deeper as a way to increase their capacity. Making the bottlenecks shrink and then expand keeps the number of parameters down, saving GPU memory for adding more layers
- The second reason draws on the concept of compression. The idea is that the model is forced to go from a large number of parameters to a small number, this will force the model to create more meaningful and compact representations

  ![bottleneck](https://user-images.githubusercontent.com/127037803/224552533-4f2350a6-0518-49cb-acf9-0cac4c6267eb.png)
- In the residual bottleneck approach, the short path is still short and has no activation function but simply performs a 1 x 1 convolution followed by batch normalization to change the original number of channels C into the desired number C’ to match the output of the subnetwork
- The first hidden layer of the subnetwork uses a 1 x 1 convolution to shrink the number of channels C before doing a normal hidden layer in the middle, followed by a final 1 x 1 convolution to expand the number of channels back up to the original account

# Stochastic Gradient Descent with Momentum
- The learning rate should be increased if the gradient for parameter $j$ consistently heads in the same direction. This can be described as wanting the gradient to build momentum
- If the gradient in one direction keeps returning similar values, it should start taking bigger and bigger steps in the same direction
- The normal gradient update equation looks like this

  $$\theta^{t+1}=\theta^t-\eta g^t$$

- The initial version of the momentum idea is to have a momentum $\mu$, which has some value in range (0, 1). Next, a velocity term called $v$ is added, which accumulates the momentum. The new velocity term $v^{t+1}$ consists of the current momentum and gradient

  $$v^{t+1}=\mu v^t+\eta g^t$$

- Because $v^{t+1}$ depends on the previous velocity and gradient, it always takes all previous gradient updates into account (that altered $v$ over the time). Next, $v$ simply replaces $g$ to perform the gradient update

  $$\theta^{t+1}=\theta^t-v^{t+1}$$

- The velocity always consists of the old gradients and after many iterations, many rounds of momentum has been applied to older gradients but only once for the gradient of the previous time step. And because $µ$ is smaller than one, it has a shrinking effect on these old gradients
- There is another form of momentum called Nesterov momentum, which has the advantage if the loss is moving in a wrong direction, Nesterov will push back the model back in the correct direction sooner
- The way it work is to first calculate the updated parameters ($\theta^{t’}$) based on the current parameters and velocity

  $$\theta_{t’}=\theta^t-\mu v^t$$

- Second, the training data is passed into the network with these new updated parameters. This will check where the model is heading. If the current momentum ($\mu v^t$) is pushing the model in the wrong direction the gradient ($\nabla_{\theta^{t’}}f_{\theta^{t’}}(x)$) would be much higher pushing it back in the correct direction
- Combined, these two will result in a smaller step that is closer to the solution

  $$v^{t+1}=\mu v^t+\eta\nabla_{\theta^{t’}}f_{\theta^{t’}}(x)$$

- Finally, the new velocity will update the current parameters resulting in the ultimate updated weights

  $$\theta^{t+1}=\theta^t-v^{t+1}$$


# Adding Variance to Momentum
- One of the most popular optimization techniques is called adaptive moment estimation (Adam), which is strongly related to the SGD with momentum
- A few terms have to me renamed: the velocity $v$ becomes $m$, and the momentum weight $\mu$ becomes $\beta_1$

  $$m^t=\beta_1*m^{t-1}+(1-\beta_1)*g^t$$

- The idea behind using the term $(1-\beta_1)$ is to give more weight to the current gradient in the exponential moving average, and less weight to the previous estimate. This helps to adaptively adjust the learning rate for each parameter based on the gradient history
- It is called exponential because the gradient from $z$ steps ago, $g^{t-z}$, is exponential contributing less at each iteration (because the equation is multiplied by $1-\beta_1$ at each iteration). It is a moving average because it is includes an average of all previous gradients and as more and more $\beta_1$ are added to older gradients, the weight of the average is determined by newer gradients; thus moving along as the epochs progress

  $$m_4=(1-\beta)[g_4+\beta¹g_3+\beta²g_2+\beta³g_1]$$

- Note that $\beta$ controls the moving average. A common value is $\beta=0.9$, meaning the average is taken over the last 10 iterations’ gradients ($1-0.9=\frac 1{10}$). When talking about momentum as an average gradient over multiple updates, variance can be considered as well. For one single data point, variance is just the residual (distance to the mean)
- If one parameter $g^t_j$ has high variance, it should not contribute too much to the momentum because it will likely change again. If a parameter $g^t_j$ has low variance, it is a reliable direction to head in, so it should have more weight in the momentum calculation
- To implement this idea, information about the variance over time can be calculated by looking at squared gradient values divided by the number of iterations (formula for variance). A momentum term is added to provide more accurate information about variance over time
- Using $\odot$ to denote the elementwise multiplication between two vectors leads to this equation for variance of velocity $v$, where $(1-\beta_2)*g^t\odot g^t$ is an approximate of variance. Adding momentum into account, this will provide a good representation of variance

  $$v^t=\beta_2*v^{t-1}+(1-\beta_2)*g^t\odot g^t$$

- There must be one alteration done to $m^t$ and $v^t$ before combining them into one equation. Because at the early stage of optimization, $t$ is a small value, $m^t$ and $v^t$ would give a biased estimate of the mean and variance
- They are biased because the mean and variance are initialized to zero, so early estimates will be too small if they are used naively. For example, if $t=1$ the value of $m_1$ would be

  $$m_1=(1-\beta_1)*g_1$$

- However this doesn’t represent the actual average. The actual average would simply be just $g_1$ (because $g_1/1=g_1$). To balance the equation, add the following

  $$\hat m=\frac{m^t}{1-\beta^t_1}\\\qquad \hat v=\frac{v^t}{1-\beta^t_2}$$

- With $\hat v$ and $\hat m$ at hand, they can be used together to update the weights $\theta$

  $$\theta^{t+1}=\theta^t-\eta\frac{\hat m}{\sqrt{\hat v} + \epsilon}$$

- The numerator $\eta*\hat m$ is computing the SGD with momentum term, but now, each dimension is normalized by the variance. That way, if a parameter has naturally large swings in its value, it will not have a huge impact ($\epsilon$ is a very small term so division by zero is avoided)

# Cosine Annealing
- This approach to adjusting the learning rate is very effective for getting the best possible results but does not provide the same degree of stabilization as other methods, so it might not work on poorly behaved datasets and networks
- Cosine annealing has an initial learning rate $\eta_0$ and a minimum rate $\eta_{min}$; the learning rate alternates between the minimum and maximum learning rates. $T_{max}$ is the number of epochs between cycles

$\qquad \eta_t=\eta_{min}+(\eta_0-\eta_{min})\frac12(1+\cos(\frac{t}{T_{max}}\pi))$

- The cosine terms fluctuates up and down like a cosine function normally does, but it is rescaled to have a maximum at $\eta_0$ and a minimum at $\eta_{min}$
- The cosine approach makes sense as a solution because a neural network can have several local minima and if the model first heads towards one of these local minima and the learning rate is decreased only, the model might get stuck in this sub-optimal area

# Sources
Huge thank you to Manning Publications and their "Inside Deep Learning (Math, Algorithms, Models)". Very detailed and well explained.

https://www.manning.com/books/inside-deep-learning
