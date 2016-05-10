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
        if (np.array_equal(point["label"], pred)):
            correct += 1
    return float(correct) / total

class model:
    def __init__(self, structure):
        self.weights=[]
        self.bias = []
        for i in range(len(structure)-1):
            self.weights.append(np.random.normal(size=(structure[i], structure[i+1])))
            self.bias.append(np.random.normal(size=(1, structure[i+1])))
            
    # TODO: Calculate prediction based on model
    def predict(self, point):
        ans = logistic(point["features"] * self.weights[-1] + self.bias[-1])
        # print "ans is: " , ans
        return  np.matrix(ans)

    # TODO: Update model using learning rate and L2 regularization
    def update(self, a, delta, eta, lam):
        # print "a is: ", type(a["features"])
        # print "deltaa is: ", type(delta)
        # print "eta is: ", type(eta)
        # print "lam is: ", type(lam)
        # print "self.weights is: ", type(self.weights)
        # m = map(lambda x: lam * x, self.weights)
        # print self.weights[0].shape
        # f = (a["features"] * delta.item(0)).shape
        # print f
        grad_weights = lam * self.weights[0] - (a["features"] * delta.item(0)).transpose()
        self.weights[0] -= eta*grad_weights
        grad_bias = (np.ones(a["features"].shape)*delta.item(0)).transpose()
        self.bias -= eta*grad_bias

    # TODO: Perform the forward step of backpropagation
    def feedforward(self, point):
        a = []
        a.append(point["features"])
        for weight, bias in zip(self.weights, self.bias):
            # print weight.shape
            # print a[-1].shape
            # print bias.shape
            M = logistic(a[-1]*weight + bias)
            # print M.shape
            a.append(M)
        return a
        
    
    # TODO: Backpropagate errors
    def backpropagate(self, a, label):
        print a
        print label

    # TODO: Train your model
    def train(self, data, epochs, rate, lam):
        length = len(data)
        for i in range(epochs):
            d = list(data)
            for i in range(length):
                random_index = random.randrange(0,len(d))
                point = d.pop(random_index)
                # print "d[\"label\"] is: ", point["label"]
                # print "self.predict(point) is:", self.predict(point)
                delta = point["label"] - self.predict(point)
                self.update(point, delta, rate, lam)

def logistic_regression(data, lam=0.00001):
    m = model([data[0]["features"].shape[1], 1])
    m.train(data, 100, 0.05, lam)
    return m
    
def neural_net(data, lam=0.00001):
    m = model([data[0]["features"].shape[1], 15, 1])
    m.train(data, 100, 0.05, lam)
    return m
