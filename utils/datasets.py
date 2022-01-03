import numpy as np
import sklearn.datasets as datasets
from sklearn.preprocessing import MinMaxScaler

def generate_circle_data(*args, **kwargs):
    print('Generating circle data...')
    x, y = datasets.make_circles(*args, **kwargs)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x = scaler.fit_transform(x)
    return x, y
