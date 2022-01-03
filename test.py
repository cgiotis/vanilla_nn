import numpy as np

y = np.array([0, 0, 1, 0, 1, 0, 1, ])
y_hat = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
predictions = np.around(y_hat).astype(int)

def calculate_accuracy(y, predictions):
    accuracy = (y == predictions)
    print(accuracy)
    print(np.sum(accuracy))
    print(np.mean(accuracy))

# calculate_accuracy(y, predictions)
dict = {}
foo = np.random.randn(4,2)
dict['foo'] = foo
print(foo)
foo *= 5
print(foo)
print(dict['foo'])
