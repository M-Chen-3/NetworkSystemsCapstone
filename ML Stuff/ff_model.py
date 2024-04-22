import tensorflow as tf
import os
import sys
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import backend

x = [(i+2, i*2) for i in range(10)]
y = [["Google", "Amazon"][i%2] for i in range(10)]

