from sdv.tabular.copulagan import CopulaGAN
from flaml import AutoML
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sdv.tabular.copulagan import CopulaGAN
from scipy.stats import ks_2samp

def classificationDataStream(sample_size):
    class_gen = CopulaGAN.load('Classification_demo/new_data_generator.sav')
    return class_gen.sample(sample_size).reset_index().iloc[:, 1:]