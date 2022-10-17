import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import psutil
import collections
import math
from sklearn.metrics import r2_score

class multiTSModelPerformance(object):

    def __init__(self, y_pred, y_true, batch_size, interval=1000):
    
        # checking if inputs are valid
        assert np.shape(y_pred) == np.shape(y_true), "y_pred and y_true must have the same dimensions."
        assert batch_size <= np.shape(y_pred)[0], "batch_size must be smaller than length of y."
        assert np.shape(y_pred)[0] >= 2, "The length of y_pred must be at least 2."
        assert np.shape(y_true)[0] >= 2, "The length of y_pred must be at least 2."
        assert interval >= 0, "frequency must be positive"
        

        # store the inputs if they are valid
        self.y_pred = y_pred
        self.y_true = y_true
        self.batch_size = batch_size
        self.interval = interval
        
        
    def showPlots(self):
        scores = {'r2': [], 'mse': [], 'mape': []}
        fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(2, 2)
        def helper(i):
            
                
        plot = FuncAnimation(fig, helper, interval=self.interval)
        plt.show()