"""
This file includes tracker which can be used to track stopping points during training.
"""

#---------------------------------------------------------------------------------------
# Dependencies
import numpy as np

#---------------------------------------------------------------------------------------
# moving average class
class MovingAverageTracker:
    """
    This class tracks the moving average for a given metric.
    The window size can be defined during initialization.
    New values are added via the update function.
    The current moving average can be accessed via the value property.
    """
    def __init__(self, window_size: int = 20):
        """
        Initialization of the Moving Average Tracker.
        :param window_size: int, nbr of values to consider for the moving average.
        """
        super().__init__()
        self.window_size = window_size
        self._value_buffer = list()
        self.filling_buffer_stage = True

    def update(self, new_value:float):
        """
        With this functions new values can be added to the value buffer, i.e. a list of
        values which have to be considered for the moving average.
        
        If the buffer is not yet filled, the new value is simply appended to the buffer.
        If the buffer is already filled, the oldest value is replaced by the one one.
        
        The pre-defined window size defines the length of the buffer.
        """
        if self.filling_buffer_stage:
            self._value_buffer.append(new_value)
            if len(self._value_buffer) >= self.window_size:
                self.filling_buffer_stage = False
        else:
            self._value_buffer = self._value_buffer[1:] + [new_value]
    @property
    def value(self):
        return np.mean(self._value_buffer)

# train val delta tracker class
class TrainValDeltaTracker:
    """
    For a given metric, this class tracks the difference between its values on the
    training and validation.
    Precisely, the absolute difference between training and validation value is
    returned.
    """
    def __init__(self):
        self.train_value = None
        self.val_value = None
    
    def set_train_value(self, new_value:float):
        """
        With this function the training value can be set. Having set the training value,
        the training value cannot be set again before the absolute-delta value is
        returned. This is to ensure that compared values really belong to the same epoch
        since training and validation values can just set once per epoch.
        """
        #assert self.train_value is None, "train value already set"
        self.train_value = new_value
    
    def set_val_value(self, new_value:float):
        """
        With this function the val. value can be set. Having set the val. value,
        the val. value cannot be set again before the absolute-delta value is
        returned. This is to ensure that compared values really belong to the same epoch
        since training and validation values can just set once per epoch.
        """
        #assert self.val_value is None, "val value already set"
        self.val_value = new_value
    
    @property
    def absolute_delta(self):
        """
        This property returns the absolute difference between previously set training
        and validation values.
        Querying this property resets the set training and validation values.
        """
        
        if self.train_value is not None: # this has to be done for pl sanity check
            assert self.train_value is not None, "train value not set"
            assert self.val_value is not None, "val value not set"
            
            delta = self.train_value - self.val_value
            abs_delta = np.abs(delta)
            
            # Reset values
            self.train_value = None
            self.val_value = None
            
            return abs_delta
        else:
            return 1 # max delta is 1