__author__ = "https://www.linkedin.com/in/bongsang/"
__license__ = "MIT"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window : window.batch(5))
dataset = dataset.map(lambda window : (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(batch_size=2).prefetch(1)

for x, y in dataset:
    print("-"*50)
    print("x = ", x.numpy())
    print("y = ", y.numpy())

