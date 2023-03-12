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

  $$f(x)=tanh(x_{C_{in}, W, H}\otimes W^{h_1}_{C_{out}, C_{in}, K, K}+b^{h_1})W^{out}_{(C_{out}*w*h), C}+b^{out})$$

- The main thing that has changed in this equation is that the dot product is replaced by a convolution (a spatially linear operation denoted by $\otimes$)
- One can add more filters to the model, where $C_{out}$ represents the number of different filters (where each filter outputs one channel) and $C_{in}$ is used to indicate the number of channels in the input
- One issue occurs because the output of the convolution has a shape of $(C, W, H)$, but the linear layer ($W^{out}+b^{out}$) expects something of the shape $(C*W*H)$ — that is one dimension with all three dimensions collapsed into it
- Essentially, the output of the convolutions must be reshaped to remove the spatial interpretation so that the linear layer can process the result and compute some predictions
