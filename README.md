# One layer perceptron
implementation of One Layer Perceptron algorithm in python


## Installation

its script dont need installation

## Usage
we have data set for seven letters that maped on array of 1 and -1 
```python

# this is out put classes 
Classes = {
    "A": [1, -1, -1, -1, -1, -1, -1],
    "B": [-1, 1, -1, -1, -1, -1, -1],
    "C": [-1, -1, 1, -1, -1, -1, -1],
    "D": [-1, -1, -1, 1, -1, -1, -1],
    "E": [-1, -1, -1, -1, 1, -1, -1],
    "J": [-1, -1, -1, -1, -1, 1, -1],
    "K": [-1, -1, -1, -1, -1, -1, 1],
}

PNN = Perceptron(63, 7, Sampels, Classes, lr=0.0007, teta=15)
print(PNN.train())
M1.test()


```

and you can change the output layer size
in this case our output laye has 3 neuron that should match with this classes 

```python 
 Classes = {
     "A": [-1, -1, 1],
     "B": [-1, 1, -1],
     "C": [-1, 1, 1],
     "D": [1, -1, -1],
     "E": [1, -1, 1],
     "J": [1, 1, -1],
     "K": [1, 1, 1],
}

PNN = Perceptron(63, 3, Sampels, Classes, lr=0.0007, teta=15)
```
