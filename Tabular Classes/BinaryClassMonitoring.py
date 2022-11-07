from sqlite3 import Timestamp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import psutil
import collections
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from collections import deque
from copy import deepcopy


class MultiTSMonitoring(object):

    def __init__(self, y_pred, y_true, timestamp, batch_size, interval=1000):

        # checking if inputs are valid
        assert np.shape(y_pred) == np.shape(
            y_true) == np.shape(timestamp), "y_pred, y_true, and timestamp must have the same dimensions."
        assert batch_size <= np.shape(
            y_pred)[0], "batch_size must be smaller than length of y."
        assert np.shape(y_pred)[
            0] >= 2, "The length of y_pred must be at least 2."
        assert interval >= 0, "Interval must be positive"

        self.num_of_batches = math.ceil(np.shape(y_pred) / batch_size)
        self.batches = {}
        y_pred = list(y_pred)
        y_true = list(y_true)
        timestamp = list(timestamp)

        for i in range(self.num_of_batches):
            if i == self.num_of_batches - 1:
                y_pred_batch = y_pred[i * batch_size:]
                y_true_batch = y_true[i * batch_size:]
                timestamp_batch = timestamp[i * batch_size:]
            else:
                y_pred_batch = y_pred[i * batch_size:(i + 1) * batch_size]
                y_true_batch = y_true[i * batch_size:(i + 1) * batch_size]
                timestamp_batch = timestamp[i * batch_size:(i + 1) * batch_size]
            self.batches[i: (y_pred_batch, y_true_batch, timestamp_batch)]

        self.interval = interval
        self.batch_size = batch_size
        self.MAEScoreDF = pd.DataFrame(
            columns=['Batch Number', 'Timestamp Range', 'Score'])
        self.MSEScoreDF = pd.DataFrame(
            columns=['Batch Number', 'Timestamp Range', 'Score'])
        self.MAPEScoreDF = pd.DataFrame(
            columns=['Batch Number', 'Timestamp Range', 'Score'])

    def addData(self, new_y_pred, new_y_true, new_timestamp, batch_size=None):
        # checking if inputs are valid
        assert np.shape(new_y_pred) == np.shape(
            new_y_true) == np.shape(new_timestamp), "y_pred, y_true, and timestamp must have the same dimensions."
        if batch_size is None:
            batch_size = self.batch_size

        num_of_batches = math.ceil(np.shape(new_y_pred) / batch_size)
        prev_num_of_batches = deepcopy(self.num_of_batches)
        self.num_of_batches += num_of_batches
        
        new_y_pred = list(new_y_pred)
        new_y_true = list(new_y_true)
        new_timestamp = list(new_timestamp)

        for i in range(prev_num_of_batches, self.num_of_batches):
            if i == self.num_of_batches - 1:
                y_pred_batch = new_y_pred[i * batch_size:]
                y_true_batch = new_y_true[i * batch_size:]
                timestamp_batch = new_timestamp[i * batch_size:]
            else:
                y_pred_batch = new_y_pred[i * batch_size:(i + 1) * batch_size]
                y_true_batch = new_y_true[i * batch_size:(i + 1) * batch_size]
                timestamp_batch = new_timestamp[i * batch_size:(i + 1) * batch_size]
            self.batches[i: (y_pred_batch, y_true_batch, timestamp_batch)]

    def removeData(self, batch_num):
        pass
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

    def setInterval(self, interval):
        self.interval = interval

    def getMAEScoreDF(self):
        return self.MAEScoreDF

    def getMSEScoreDF(self):
        return self.MSEScoreDF

    def getMAPEScoreDF(self):
        return self.MAPEScoreDF

    def modelPerformancePlots(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle('Multivariate Time Series Model Performance')

        # set up plot 1
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('y')
        ax1.set_title('y_pred VS. y_true')
        y_pred, = ax1.plot([], [], 'b', lw=2)
        y_true, = ax1.plot([], [], 'r', lw=2)
        ax1.legend(['y_pred', 'y_true'])

        # set up plot 2
        ax2.set_xlabel('Batch Number')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Mean Absolute Error')
        mae, = ax2.plot([], [], 'b', lw=2)

        # set up plot 3
        ax3.set_xlabel('Batch Number')
        ax3.set_ylabel('Mean Squared Error')
        ax3.set_title('Mean Squared Error')
        mse, = ax3.plot([], [], 'b', lw=2)

        # set up plot 4
        ax4.set_xlabel('Batch Number')
        ax4.set_ylabel('Mean Absolute Percentage Error')
        ax4.set_title('Mean Absolute Percentage Error')
        mape, = ax4.plot([], [], 'b', lw=2)
        
        self.current_batch = 0
        def helper(i):
            if self.current_batch not in self.batches:
                self.current_batch -= 1
                
            y_pred_batch, y_true_batch, timestamp_batch = self.batches[self.current_batch]
            timestamp_range = (timestamp_batch[0], timestamp_batch[-1])
            # calculate scores
            mae = mean_absolute_error(y_true_batch, y_pred_batch)
            self.MAEScoreDF.loc[len(self.MAEScoreDF)] = [
                self.current_batch, timestamp_range, mae]
            mse = mean_squared_error(y_true_batch, y_pred_batch)
            self.MSEScoreDF.loc[len(self.MSEScoreDF)] = [
                self.current_batch, timestamp_range, mse]
            mape = mean_absolute_percentage_error(y_true_batch, y_pred_batch)
            self.MAPEScoreDF.loc[len(self.MAPEScoreDF)] = [
                self.current_batch, timestamp_range, mape]

            # Graph 1: prediction values vs true values
            ax1.plot(timestamp_batch, y_pred_batch)

        plot = FuncAnimation(fig, helper, interval=self.interval)
        plt.show()

    def dataPropertiesPlots(self):
        pass
