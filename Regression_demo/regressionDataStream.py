from sdv.tabular.copulagan import CopulaGAN
from flaml import AutoML
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error as mse
from sdv.tabular.copulagan import CopulaGAN
from scipy.stats import ks_2samp


def regressionDataStream(sample_size):
    reg_gen = CopulaGAN.load('Regression_demo/new_data_generator.sav')
    return reg_gen.sample(sample_size).reset_index().iloc[:, 1:]
