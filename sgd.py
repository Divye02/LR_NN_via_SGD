from math import exp
import math
import random
import numpy as np

# TODO: Calculate logistic
def logistic(x):
    # print
    # print "x is :", x
    val = 1.0 / (1 + np.exp(-1.0*x))
    # print "val is :", val
    return val
# TODO: Calculate accuracy of predictions on data

def accuracy(data, predictions):
    total = len(data)
    correct = 0
    for point, pred in zip(data, predictions):
        if (pred[0] < 0.5 and point["label"][0] == 0) or (pred[0] >= 0.5 and point["label"][0] == 1):
            correct += 1
    return 1.0 * float(correct) / total

class model:
    def __init__(self, structure):
        self.weights=[]
        self.bias = []
        for i in range(len(structure)-1):
            self.weights.append(np.random.normal(size=(structure[i], structure[i+1])))
            self.bias.append(np.random.normal(size=(1, structure[i+1])))
            
    # TODO: Calculate prediction based on model
    def predict(self, point):
        ans = self.feedforward(point)
        # print "ans is: " , ans
        return  ans[-1]

    # TODO: Update model using learning rate and L2 regularization
    def update(self, a, delta, eta, lam):
        for i in range(0, len(self.weights)):
            grad_weights = lam * self.weights[i] - (a[i] * delta[i].item(0)).transpose()
            self.weights[i] -= eta*grad_weights
            grad_bias =  - np.transpose(delta[i].item(0))
            self.bias[i] -= eta*grad_bias

    # TODO: Perform the forward step of backpropagation
    def feedforward(self, point):
        a = []
        a.append(point["features"])
        for weight, bias in zip(self.weights, self.bias):
            M = logistic(a[-1]*weight + bias)
            a.append(M)
        return a
        
    
    # TODO: Backpropagate errors
    def backpropagate(self, a, label):
        L = len(a) - 1
        sigma = label - a[-1]
        result = []
        result.insert(0, label - a[-1])
        while L > 1:
            sigma = np.dot(sigma, np.multiply(a[L - 1], 1 - a[L - 1]))
            sigma = np.multiply(self.weights[L -1], sigma.transpose())
            L = L - 1
            result.insert(0, sigma)
        return result
            
        
        

    # TODO: Train your model
    def train(self, data, epochs, rate, lam):
        length = len(data)
        for i in range(epochs*length):
            random_index = random.randrange(0,length)
            point = data[random_index]
            a = self.feedforward(point)
            delta = self.backpropagate(a, point["label"])
            self.update(a, delta, rate, lam)

def logistic_regression(data, lam=0.00001):
    m = model([data[0]["features"].shape[1], 1])
    m.train(data, 100, 0.05, lam)
    return m
    
def neural_net(data, lam=0.00001):
    m = model([data[0]["features"].shape[1], 15, 1])
    m.train(data, 100, 0.05, lam)
    return m
