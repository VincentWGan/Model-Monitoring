import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import psutil
import collections
import math
from sklearn.metrics import r2_score
from collections import deque

class multiTSModelPerformance(object):

    def __init__(self, y_pred, y_true, batch_size, interval=1000):
    
        # checking if inputs are valid
        assert np.shape(y_pred) == np.shape(y_true), "y_pred and y_true must have the same dimensions."
        assert batch_size <= np.shape(y_pred)[0], "batch_size must be smaller than length of y."
        assert np.shape(y_pred)[0] >= 2, "The length of y_pred must be at least 2."
        assert np.shape(y_true)[0] >= 2, "The length of y_pred must be at least 2."
        assert interval >= 0, "Interval must be positive"
        

        # store the inputs if they are valid
        self.y_pred = y_pred
        self.y_true = y_true
        self.batch_size = batch_size
        self.interval = interval
        
    def addData(self, new_y_pred, new_y_true):
        pass
        
    def showPlots(self):
        scores = {'r2': [], 'mse': [], 'mape': []}
        fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(2, 2)
        def helper(i):
            y_pred_batch = 
            scores['r2'].append(r2_score(self.y_true, self.y_pred))
                
        plot = FuncAnimation(fig, helper, interval=self.interval)
        plt.show()