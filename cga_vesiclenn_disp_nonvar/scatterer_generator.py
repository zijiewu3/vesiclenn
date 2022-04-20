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
    def __init__(self,shape_params = [NNMODEL_DIR_PATH+'/1.h5',0.6,25],
                 minvalu = (20, 0.1, 0.01, 0.01, 0.5,0.05, 1),#Rtotal fcore fAin fAout sAin pd
                 maxvalu = (3000,0.95,0.99,0.99,0.1,0.5,6)):
        self.minvalu = minvalu
        self.maxvalu = maxvalu
        self.numvars = 7
        self.model_path = shape_params[0]
        self.sB = shape_params[1]
        self.dp = shape_params[2]

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)
        self.nn_minvalu = np.array([0,0,0,0.5,0.5,np.log(0.5)])
        self.nn_maxvalu = np.array([1,1,1,1,0.75,np.log(10)])

    def converttoIQ(self,qrange,param):
        self.load_model()
        param = np.array(param)
        R_total_mu = param[0]
        R_core_mu = R_total_mu*param[1]
        tAin = (R_total_mu-R_core_mu)*param[2]
        tAout = (R_total_mu-R_core_mu-tAin)*param[3]
        tB = R_total_mu-R_core_mu-tAin-tAout
         
        R_core_sd = R_total_mu*param[-2]
        nn_output_sum = np.zeros(len(qrange))
        for Rcore in np.linspace(R_core_mu-1.5*R_core_sd, R_core_mu+1.5*R_core_sd,num = 20):
            if Rcore < 5:
                continue
            Rtotal = Rcore+tAin+tAout+tB
            qRrange = qrange*Rtotal
            fcore = Rcore/Rtotal
            fAin = tAin/(Rtotal-Rcore)
            fAout = param[3]
            sAin = param[4]

            nn_input = np.zeros((len(qrange),6))
            input_nn = [fcore,fAin,fAout,sAin,self.sB]
            nn_input[:,5] = (np.log10(qRrange)-self.nn_minvalu[-1])/(self.nn_maxvalu[-1]-self.nn_minvalu[-1])
            nn_input[:,:5] = (input_nn-self.nn_minvalu[:5])/(self.nn_maxvalu[:5]-self.nn_minvalu[:5])
            nn_output_sum += np.array([10**i for i in self.model(nn_input).numpy()]).flatten()
        nn_output_sum += 10**(-param[-1])
        return nn_output_sum/nn_output_sum[0]
            
class scatterer_generator_log_normal:
    '''
    shape specific descriptors (shape_params):
    ------------------------------------------
    *None.*

    Input parameters to be predicted:
    ------------------------------------------
    *TODO*
    '''
    def __init__(self,shape_params = [NNMODEL_DIR_PATH+'/1.h5',NNMODEL_DIR_PATH+'/2.h5',5],
                 minvalu = (50,0,0,0,0,0,0,6),
                 maxvalu = (2000,1,1,1,1,1,1,7)):
        self.minvalu = minvalu
        self.maxvalu = maxvalu
        self.numvars = 8
        self.model_path_left = shape_params[0]
        self.model_path_right = shape_params[1]
        self.qRbreak = shape_params[2]

    def load_model(self):
        self.model_left = tf.keras.models.load_model(self.model_path_left)
        self.nn_minvalu_left = np.array([0,0,0,0,0,0,-1])
        self.nn_maxvalu_left = np.array([1,1,1,1,1,1,2])
        self.model_right = tf.keras.models.load_model(self.model_path_right)
        self.nn_minvalu_right = np.array([0,0,0,0,0,0,5])
        self.nn_maxvalu_right = np.array([1,1,1,1,1,1,10])

    def converttoIQ(self,qrange,param):
        self.load_model()
        param = np.array(param)
        R = param[0]
        qRrange = np.array(qrange)*R
        qRrange_left = qRrange[np.where(qRrange < self.qRbreak)]
        qRrange_right = qRrange[np.where(qRrange >= self.qRbreak)]
        nn_input_left = np.zeros((len(qRrange_left),7))
        nn_input_right = np.zeros((len(qRrange_right),7))
        nn_input_left[:,6] = (np.log10(qRrange_left)-self.nn_minvalu_left[-1])/(self.nn_maxvalu_left[-1]-self.nn_minvalu_left[-1])
        nn_input_right[:,6] = (qRrange_right-self.nn_minvalu_right[-1])/(self.nn_maxvalu_right[-1]-self.nn_minvalu_right[-1])
        nn_input_left[:,:6] = (param[1:7]-self.nn_minvalu_left[:6])/(self.nn_maxvalu_left[:6]-self.nn_minvalu_left[:6])
        nn_input_right[:,:6] = (param[1:7]-self.nn_minvalu_right[:6])/(self.nn_maxvalu_right[:6]-self.nn_minvalu_right[:6])
        nn_output_left = np.array([10**i for i in self.model_left(nn_input_left).numpy()]).flatten()
        nn_output_right = np.array([10**i for i in self.model_right(nn_input_right).numpy()]).flatten()
        nn_output = np.hstack((nn_output_left,nn_output_right))
        nn_output = nn_output + 10**(-param[-1])
        return nn_output/nn_output[0]
 
class scatterer_generator_log_log:
    '''
    shape specific descriptors (shape_params):
    ------------------------------------------
    *None.*

    Input parameters to be predicted:
    ------------------------------------------
    *TODO*
    '''
    def __init__(self,shape_params = [NNMODEL_DIR_PATH+'/1.h5',NNMODEL_DIR_PATH+'/2.h5',5],
                 minvalu = (50,0,0,0,0,0,0,6),
                 maxvalu = (2000,1,1,1,1,1,1,7)):
        self.minvalu = minvalu
        self.maxvalu = maxvalu
        self.numvars = 8
        self.model_path_left = shape_params[0]
        self.model_path_right = shape_params[1]
        self.qRbreak = shape_params[2]

    def load_model(self):
        self.model_left = tf.keras.models.load_model(self.model_path_left)
        self.nn_minvalu_left = np.array([0,0,0,0,0,0,-1])
        self.nn_maxvalu_left = np.array([1,1,1,1,1,1,2])
        self.model_right = tf.keras.models.load_model(self.model_path_right)
        self.nn_minvalu_right = np.array([0,0,0,0,0,0,-1])
        self.nn_maxvalu_right = np.array([1,1,1,1,1,1,2])

    def converttoIQ(self,qrange,param):
        self.load_model()
        param = np.array(param)
        R = param[0]
        qRrange = np.array(qrange)*R
        qRrange_left = qRrange[np.where(qRrange < self.qRbreak)]
        qRrange_right = qRrange[np.where(qRrange >= self.qRbreak)]
        nn_input_left = np.zeros((len(qRrange_left),7))
        nn_input_right = np.zeros((len(qRrange_right),7))
        nn_input_left[:,6] = (np.log10(qRrange_left)-self.nn_minvalu_left[-1])/(self.nn_maxvalu_left[-1]-self.nn_minvalu_left[-1])
        nn_input_right[:,6] = (np.log10(qRrange_right)-self.nn_minvalu_right[-1])/(self.nn_maxvalu_right[-1]-self.nn_minvalu_right[-1])
        nn_input_left[:,:6] = (param[1:7]-self.nn_minvalu_left[:6])/(self.nn_maxvalu_left[:6]-self.nn_minvalu_left[:6])
        nn_input_right[:,:6] = (param[1:7]-self.nn_minvalu_right[:6])/(self.nn_maxvalu_right[:6]-self.nn_minvalu_right[:6])
        nn_output_left = np.array([10**i for i in self.model_left(nn_input_left).numpy()]).flatten()
        nn_output_right = np.array([10**i for i in self.model_right(nn_input_right).numpy()]).flatten()
        nn_output = np.hstack((nn_output_left,nn_output_right))
        nn_output = nn_output + 10**(-param[-1])
        return nn_output/nn_output[0]
 
