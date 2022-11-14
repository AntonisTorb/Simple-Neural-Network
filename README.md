# Simple-Neural-Network
A simple neural network for basic operations and learning. Supports multiple hidden layers and 3 different activation functions. All weights and biases are initiated randomly in the range (-1, 1).

## 1) Initializing

After importing the `snn` module in your script, create an instance of the network by calling `SimpleNeuralNetwork`. The arguments required are:

- `inputs_no`: Number of input nodes.
- `hidden_no`: Number of hidden nodes.
- `outputs_no`: Number of output nodes.

You can optionaly provide:

- `hidden_layers_no`: Number of hidden layers (default = 1).
- `activation_function`: Activation function to use (default = `"tanh"`).

## 2) Training

To train the model, call the `train` method for the network instance and provide the following parameters:

- `input`: The input data in a list.
- `target`: The target data in a list.

You can optionaly provide:

- `learning_rate`: The learning rate of the model (default = 0.1).

At this point, the `train` method returns the final result of the calculation, in order to monitor the accuracy during the learning process, so make sure to assign it to a variable.

## 3) Predictions

In order to make a prediction, call the `predict` method for the network instance and provide the following parameter:

- `input`: The input data in a list.

At this point, the `predict` method returns a list of the results from every hidden layer, not just the output, as these results are used for the training process. You can access the output result by slicing the returned object as such: 

- `predict(input)[-1][0][0]`.

Included is an XOR example with a simple accuracy logging during the training phase.

If the results are not accurate enough, try adjusting the amount of hidded layers and/or hidden nodes, or try a different activation function. Please keep in mind this is intended as an introduction to Machine Learning, so depending on the complexity of your project, results can be extremelly inacurrate.

Resources used to create this:

- [Neural Networks - The Nature of Code by "The Coding Train"](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh)

- [Neural networks by "3Blue1Brown"](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

- [Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math) by "Samson Zhang"](https://www.youtube.com/watch?v=w8yWXqWQYmU)

## Future plans

- [ ] Attempt to use the network for an OCR (Optical Character Recognition) application and apply improvements and/or adjustments if necessary.
- [ ] Add more error handling for invalid inputs.
- [ ] Add option to adjust the weights and biases after a mini batch of training data, rather than after every iteration.
- [ ] Look into different activation functions.
- [ ] Option to save weights and biases in JSON file after training, as well as loading them from a JSON file.
- [ ] Improve the accuracy logging in the XOR example and maybe include a visualization of the learning process.
- [ ] General code improvements where possible.