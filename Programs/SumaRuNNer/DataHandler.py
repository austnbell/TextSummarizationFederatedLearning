
import numpy as np

class SumaDataHandler(object):
    """
    Data handler for federated learning
    """
    def __init__(self, data_config=None):
        #super().__init__()
        self.X_train_file = data_config['X_train_file']
        self.y_train_file = data_config['y_train_file']
        self.X_test_file = data_config['X_test_file']
        self.y_test_file = data_config['y_test_file']
        
        # Load the data
        self.load_data()
        
    def load_data(self):
        """
        Load training and and test data to memory
        """
        self.X_train = np.load(self.X_train_file)
        self.y_train = np.load(self.y_train_file)
        
        self.X_test = np.load(self.X_test_file)
        self.y_test = np.load(self.y_test_file)


    def get_training_data(self):
        """
        Retrieves pre-processed data
        """
        
        return (self.X_train[:80000], self.y_train[:80000]), (self.X_test, self.y_test)
