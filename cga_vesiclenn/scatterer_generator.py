import crease_ga as cga
import numpy as np
import sys
from tensorflow import keras
import tensorflow as tf
from cga_vesiclenn import NNMODEL_DIR_PATH
class scatterer_generator:
    '''
    shape specific descriptors (shape_params):
    ------------------------------------------
    *None.*

    Input parameters to be predicted:
    ------------------------------------------
    *TODO*
    '''
    def __init__(self,shape_params = [NNMODEL_DIR_PATH+'/1.h5'],
                 minvalu = (50,0,0,0,0,0,0,2),
                 maxvalu = (2000,1,1,1,1,1,1,7)):
        self.minvalu = minvalu
        self.maxvalu = maxvalu
        self.numvars = 8
        self.model_path = shape_params[0]

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)
        self.nn_minvalu = np.array([0,0,0,0,0,0,-2])
        self.nn_maxvalu = np.array([1,1,1,1,1,1,2])

    def converttoIQ(self,qrange,param):
        self.load_model()
        param = np.array(param)
        R = param[0]
        nn_input = np.zeros((len(qrange),7))
        nn_input[:,6] = (np.log10(qrange*R)-self.nn_minvalu[-1])/(self.nn_maxvalu[-1]-self.nn_minvalu[-1])
        nn_input[:,:6] = (param[1:7]-self.nn_minvalu[:6])/(self.nn_maxvalu[:6]-self.nn_minvalu[:6])
        nn_output = np.array([10**i for i in self.model(nn_input).numpy()]).flatten()
        nn_output = nn_output + 10**(-param[-1])
        return nn_output/nn_output[0]
            

