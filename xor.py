import random

from snn import SimpleNeuralNetwork

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

for i in range(10000):
    data = random.choice(training_data)
    result = nn.train(data["input"], data["target"], 0.1)
    if not i % 100:
        target = data["target"][0]
        print(f"Iteration: {i},\nAccuracy = {(1 - abs(target - result)) * 100}%")

result1 = nn.predict([-1,  1])[-1][0][0]
result2 = nn.predict([ 1,  1])[-1][0][0]
result3 = nn.predict([ 1, -1])[-1][0][0]
result4 = nn.predict([-1, -1])[-1][0][0]

print(f"[0,1]: {result1} (1)")
print(f"[1,1]: {result2} (0)")
print(f"[1,0]: {result3} (1)")
print(f"[0,0]: {result4} (0)")
