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
    def __init__(self,
                 shape_params = [NNMODEL_DIR_PATH+'/1.h5'],
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
            
class scatterer_generator_2parts:
    '''
    shape specific descriptors (shape_params):
    ------------------------------------------
    *None.*

    Input parameters to be predicted:
    ------------------------------------------
    *TODO*
    '''
    def __init__(self,
                 shape_params = [NNMODEL_DIR_PATH+'/1.h5',NNMODEL_DIR_PATH+'/2.h5',qRbreak],
                 minvalu = (50,0,0,0,0,0,0,2),
                 maxvalu = (2000,1,1,1,1,1,1,7)):
        self.minvalu = minvalu
        self.maxvalu = maxvalu
        self.numvars = 8
        self.left_model_path = shape_params[0]
        self.right_model_path = shape_params[1]
        self.qRbreak = qRbreak

    def load_model(self):
        self.left_model = tf.keras.models.load_model(self.left_model_path)
        self.right_model = tf.keras.models.load_model(self.right_model_path)
        self.nn_minvalu_left = np.array([0,0,0,0,0,0,-1])
        self.nn_maxvalu_left = np.array([1,1,1,1,1,1,2])
        self.nn_minvalu_right = np.array([0,0,0,0,0,0,5])
        self.nn_maxvalu_right = np.array([1,1,1,1,1,1,10])

    def converttoIQ(self,qrange,param):
        self.load_model()
        param = np.array(param)
        R = param[0]
        qrange = np.array(qrange)
        qRrange = qrange*R
        qRrange_left = qRrange[np.where(qRrange < self.qRbreak)]
        qRrange_right = qRrange[np.where(qRrange >= self.qRbreak)]
        nn_input_left = np.zeros((len(qRrange_left),7))
        nn_input_right = np.zeros((len(qRrange_right),7))
        nn_input_left[:,:6] = (param[1:7]-self.nn_minvalu_left[:6])/(self.nn_maxvalu_left[:6]-self.nn_minvalu_left[:6])
        nn_input_right[:,:6] = (param[1:7]-self.nn_minvalu_right[:6])/(self.nn_maxvalu_right[:6]-self.nn_minvalu_right[:6])

        nn_input_left[:,6] = (np.log10(qRrange_left)-self.nn_minvalu_left[-1])/(self.nn_maxvalu_left[-1]-self.nn_minvalu_left[-1])
        nn_input_right[:,6] = (qRrange_right-self.nn_minvalu_right[-1])/(self.nn_maxvalu_right[-1]-self.nn_minvalu_right[-1])
        
        nn_output_left = np.array([10**i for i in self.left_model(nn_input_left).numpy()]).flatten()
        nn_output_right = np.array([10**i for i in self.right_model(nn_input_right).numpy()]).flatten()
        nn_output = np.hstack((nn_output_left,nn_output_right))
        nn_output = nn_output + 10**(-param[-1])
        return nn_output/nn_output[0]
            

