import random

from snn import SimpleNeuralNetwork
import time

def main() -> None:
    '''Main function.'''

    # Non-batch training.
    training_iterations = 10000
    logging_step = 10000
    learning_rate = 0.1

    # Batch training.
    training_iterations_b = 50000
    learning_rate_b = 0.4
    logging_step_b = 50000
    batch_size = 10

    # Non-batch neural network.
    nn = SimpleNeuralNetwork(2, 2, 1, 1, "tanh")

    # Batch neural network.
    nn_b = SimpleNeuralNetwork(2, 2, 1, 1, "tanh")

    training_data = [
        {
            "input" : [-1, -1],
            "target": [-1]
        },
        {
            "input" : [-1,  1],
            "target": [1]
        },
        {
            "input" : [ 1, -1],
            "target": [1]
        },
        {
            "input" : [ 1,  1],
            "target": [-1]
        }
    ]

    # Training the non-batch network.
    st = time.time()
    for training_index in range(training_iterations):
        data = random.choice(training_data)
        result = nn.train(data["input"], data["target"], learning_rate)[0][0]
        if not training_index % logging_step and not training_index == 0:
            target = data["target"][0]
            print(f"Iteration: {training_index},\nAccuracy = {(1 - abs(target - result)) * 100}%")
    print(f"Training time for non-batch network: {round(time.time() - st, 4)} seconds.")
    
    # Testing the non-batch network predictions.
    result1 = nn.predict([-1,  1])[0][0]
    result2 = nn.predict([ 1,  1])[0][0]
    result3 = nn.predict([ 1, -1])[0][0]
    result4 = nn.predict([-1, -1])[0][0]
    
    print("Non-batch network results:")
    print(f"[0,1]: {result1} (1)")
    print(f"[1,1]: {result2} (-1)")
    print(f"[1,0]: {result3} (1)")
    print(f"[0,0]: {result4} (-1)")

    # Training the batch network.
    st = time.time()
    for training_index in range(training_iterations_b):
        data = random.choice(training_data)
        result = nn_b.train(data["input"], data["target"], learning_rate_b, batch_size, training_index)[0][0]
        if not training_index % logging_step_b and not training_index == 0:
            target = data["target"][0]
            print(f"Iteration: {training_index},\nAccuracy = {(1 - abs(target - result)) * 100}%")
    print(f"Training time for batch network: {round(time.time() - st, 4)} seconds.")

    # Testing the batch network predictions.
    result1 = nn_b.predict([-1,  1])[0][0]
    result2 = nn_b.predict([ 1,  1])[0][0]
    result3 = nn_b.predict([ 1, -1])[0][0]
    result4 = nn_b.predict([-1, -1])[0][0]
    
    print("Batch network results:")
    print(f"[0,1]: {result1} (1)")
    print(f"[1,1]: {result2} (-1)")
    print(f"[1,0]: {result3} (1)")
    print(f"[0,0]: {result4} (-1)")

if __name__ == "__main__":
    main()
    