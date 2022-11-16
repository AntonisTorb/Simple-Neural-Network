import random

from snn import SimpleNeuralNetwork


def main() -> None:
    '''Main function.'''

    training_iterations = 10000
    learning_rate = 0.1
    logging_step = 100

    nn = SimpleNeuralNetwork(2, 2, 1, 1, "tanh")

    training_data = [
        {
            "input" : [-1, -1],
            "target": [0]
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
            "target": [0]
        }
    ]

    # Training the network.
    for i in range(training_iterations):
        data = random.choice(training_data)
        result = nn.train(data["input"], data["target"], learning_rate)[0][0]
        if not i % logging_step:
            target = data["target"][0]
            print(f"Iteration: {i},\nAccuracy = {(1 - abs(target - result)) * 100}%")

    # Testing the network predictions.
    result1 = nn.predict([-1,  1])[0][0]
    result2 = nn.predict([ 1,  1])[0][0]
    result3 = nn.predict([ 1, -1])[0][0]
    result4 = nn.predict([-1, -1])[0][0]

    print(f"[0,1]: {result1} (1)")
    print(f"[1,1]: {result2} (0)")
    print(f"[1,0]: {result3} (1)")
    print(f"[0,0]: {result4} (0)")


if __name__ == "__main__":
    main()
    