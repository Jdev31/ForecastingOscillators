import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from Oscillators import Oscillator
import time

reloaded_model = tf.keras.models.load_model("1753473770.h5")
reloaded_model.summary()