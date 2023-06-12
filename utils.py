import os
import math
from time import time

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize

from geneticalgorithm import geneticalgorithm as ga
import scipy.optimize as opt
from numdifftools import Jacobian, Hessian

import matplotlib.pyplot as plt
from pyearth import Earth
from matk import matk, pest_io

from Uncertainty import *
from uncertaintyMetric import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
import keras
import keras.backend as K
from keras import Model
from keras.layers import Input, Dense, BatchNormalization, PReLU, Dropout
from keras.regularizers import L1, L2

class FullOpt:
    def __init__(self, select=0):
        self.dims = 2
        self.measure_type = 1
        self.ROM_obj = LinearRegression()
        self.ROM_data = LinearRegression()
        self.nMCSamples = 750
        self.nDataRealization = 25
        self.verbose = False
        
        def f1(x):  return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
        def f2(x):  return x[1]*x[0]**2 - 2*x[0]*x[1]**2 + 3*x[0]*x[1] + 4
        def f3(x):  return -x[0]*x[1]*np.exp(-(x[0]**2+x[1]**2)/2)
        def f4(x):  return 1 - ((1 + np.cos(12*np.sqrt(x[0]**2+x[1]**2)))/(0.5 * (x[0]**2+x[1]**2) + 2))
        def f5(x):  return np.abs(x[0]*np.sin(x[0])+0.1*x[0]) + np.abs(x[1]*np.sin(x[1])*0.1*x[1])
        def f6(x):  return -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2))) - np.exp(0.5*(np.cos (2*np.pi*x[0])+np.cos(2*np.pi*x[1]))) + np.exp(1) + 20
        def f7(x):  return ((x[0]**2+x[1]**2)/4000) - np.cos(x[0])*np.cos(x[1]/np.sqrt(2))+1
        def f8(x):  return x[0]**2 + x[1]**2 - 10*np.cos(2*np.pi*x[0]) - 10*np.cos(2*np.pi*x[1]) + 20
        def f9(x):  return (np.sqrt(np.sqrt(x[0]**2+x[1]**2)) + np.sqrt(np.sqrt(x[0]**2+x[1]**2))*np.sin(50*np.sqrt(x[0]**2+x[1]**2)**(1/5))**2)**2
        def f10(x):  return -x[0]*np.sin(np.sqrt(np.abs(x[0]))) - x[1]*np.sin(np.sqrt(np.abs(x[1]))) + 837.9657745448674
        def f11(x): return (np.abs(x[0])+np.abs(x[1])) * np.exp(-np.sin(x[0]**2) - np.sin(x[1]**2))
        def f12(x): return 1 - np.cos(2*np.pi*np.sqrt(x[0]**2+x[1]**2)) + 0.1*np.sqrt(x[0]**2+x[1]**2)
        
        def f0(obj):
            res = proxy(ncol_data        = int(obj), 
                        measure_type     = self.measure_type,
                        rom_obj          = self.ROM_obj, 
                        rom_data         = self.ROM_data, 
                        nMCSamples       = self.nMCSamples, 
                        nDataRealization = self.nDataRealization, 
                        verbose          = self.verbose)
            return res.value
        
        if   select==0:  self.fun = f0
        elif select==1:  self.fun = f1
        elif select==2:  self.fun = f2
        elif select==3:  self.fun = f3
        elif select==4:  self.fun = f4
        elif select==5:  self.fun = f5
        elif select==6:  self.fun = f6
        elif select==7:  self.fun = f7
        elif select==8:  self.fun = f8
        elif select==9:  self.fun = f9
        elif select==10: self.fun = f10
        elif select==11: self.fun = f11
        elif select==12: self.fun = f12
        else: print('please select a function between 0 and 12 - [0=CO2-Optim-Proxy, 1-12=Test-Functions]')
        
    def fun_jac(self, x):
        self.Jac = Jacobian(lambda x: self.fun(x))(x).ravel()
        return self.Jac
    
    def fun_hess(self, x):
        self.Hess = Hessian(lambda x: self.fun(x))(x)
        return self.Hess
        
    def global_opt(self, 
                   varbounds=[-10,10], vartype='real', maxiter=1000, pop_size=50,
                   cross_prob=0.25, cross_type='uniform', mut_prob=0.2, parent_prop=0.2, elitratio=0.01, 
                   maxiter_no_improv=100, Convergence_Curve=False, Progress_Bar=False):
        params = {'max_num_iteration':      maxiter ,   
                  'population_size':        pop_size,
                  'crossover_probability':  cross_prob, 
                  'crossover_type':         cross_type, 
                  'mutation_probability':   mut_prob,  
                  'parents_portion':        parent_prop, 
                  'elit_ratio':             elitratio,
                  'max_iteration_without_improv': maxiter_no_improv}
        self.global_model = ga(function             = self.fun,    
                               dimension            = self.dims,
                               variable_type        = vartype,     
                               variable_boundaries  = np.array([varbounds]*self.dims),
                               algorithm_parameters = params,      
                               convergence_curve    = Convergence_Curve,
                               progress_bar         = Progress_Bar)
        self.global_model.run()
        xsol, ysol, fsol = self.global_model.best_variable[0], self.global_model.best_variable[1], self.global_model.best_function
        self.global_res_df = pd.DataFrame(np.array((xsol, ysol, fsol)), index=['X','Y','f']).T
        return self.global_model, self.global_res_df
    
    def local_opt(self, x0=[-1,1], method='CG', tol=None, options={'maxiter':None, 'disp':False, 'gtol':1e-6}):
        '''
        Methods: 
            'Nelder-Mead', 'Powell', 'CG - conjugate gradient', 'Newton-CG', 'BFGS', 'L-BFGS-B (bounded)', 
            'TNC - truncated Newton', 'COBYLA (constrained opt by linear approx)', 'SLSQP (sequential least squares programming)', 
            'dogleg', 'trust-constr', 'trust-ncg (newtorn conjugate gradient)', 'trust-exact', 'trust-krylov']
        '''
        all_x, all_y, all_f = [x0[0]], [x0[1]], [self.fun(x0)]
        def store(X_):
            x, y = X_
            all_x.append(x)
            all_y.append(y)
            all_f.append(self.fun(X_))
        self.local_res = opt.minimize(fun=self.fun, x0=x0, method=method, tol=tol,
                                      jac=self.fun_jac, hess=self.fun_hess, 
                                      callback=store, options=options)
        print('Method: {} | Solution: (x={:.3f}, y={:.3f}), f(x,y)={:.5f} | niter: {}'.format(method, all_x[-1], all_y[-1], all_f[-1], len(all_f)))
        self.local_res_df = pd.DataFrame({'X':all_x, 'Y':all_y, 'f':all_f})
        return self.local_res, self.local_res_df  
    
    def make_plot(self, global_res, local_res, labels=['Global','Local'], npts=500, levels=30,
                    delta_lims = 1, showcontours=False, filled=False,
                    mbounds=[-3,3], angle=[30,45], rstride=8, cstride=8, alpha=0.8, msize=[60,60], 
                    markers=['d','*'], cmaps=['viridis','jet','coolwarm'], colors=['r','b'], figsize=(25,8)):

        xx = np.linspace(mbounds[0], mbounds[1], npts)
        X, Y = np.meshgrid(xx, xx)
        fig = plt.figure(figsize=figsize)
        ax1 = plt.subplot(121)
        if filled:  
            im1 = plt.contourf(X, Y, self.fun([X,Y]), levels=levels, cmap=cmaps[0], alpha=alpha)
        else:       
            im1 = plt.contour(X,  Y, self.fun([X,Y]), levels=levels, cmap=cmaps[0], alpha=alpha)
        ax1.scatter(global_res['X'], global_res['Y'], c=colors[0], s=msize[0], marker=markers[0], label=labels[0])
        ax1.plot(local_res['X'], local_res['Y'],  c=colors[1], linestyle='-', marker=markers[1], label=labels[1])
        ax1.set(xlabel='X', ylabel='Y', title='Contours $f(x,y)$'); plt.colorbar(im1, label='$f(x,y)$'); plt.legend()
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.view_init(elev=angle[0], azim=angle[1])
        im2 = ax2.plot_surface(X, Y, self.fun([X,Y]), cmap=cmaps[1], alpha=alpha, cstride=cstride, rstride=rstride)
        if showcontours:
            ax2.contour(X, Y, self.fun([X,Y]), zdir='z', offset=-1, cmap=cmaps[2])
            ax2.set(xlim=(mbounds[0]-delta_lims,mbounds[1]+delta_lims), ylim=(mbounds[0]-delta_lims,mbounds[1]+delta_lims))
        ax2.scatter(global_res['X'], global_res['Y'], global_res['f'],                c=colors[0], marker=markers[0], s=msize[0], label=labels[0])
        ax2.scatter(local_res.iloc[-1,0], local_res.iloc[-1,1], local_res.iloc[-1,2], c=colors[1], marker=markers[1], s=msize[1], label=labels[1])
        ax2.set(xlabel='X', ylabel='Y', zlabel='Z', title='Surface $f(x,y)$'); plt.colorbar(im2, label='$f(x,y)$'); plt.legend()
        plt.show()

###############################################################################################################################################################################
class proxy:
    def __init__(self, ncol_data=7, data_dir=os.path.join(os.getcwd(),'data'), measure_type=1, obj_fname='run_co2mt.his', 
                 ncol_obj=50, tot_time=1800, nTime=60, nIntv=1, nTrain=500, nParam=4, nMCSamples=100000, nDataRealization=200,
                 xmin=[1e-19, 1e-19, 1e-19, 0.5], xmax=[1e-14, 1e-14, 1e-14, 2.0], error_option=3, time_sensitivity=1,
                 titles=['.', 'Pressure', 'CO2 Saturation (l)', 'Temperature', 'Pressure + CO2 Saturation'], 
                 rom_val=0, rom_obj=Earth(), rom_data=Earth(),
                 NN=False, verbose=False):
        self.nColumn_obj, self.nColumn_data = ncol_obj, [ncol_data]
        self.rom_obj, self.rom_data = rom_obj, rom_data
        self.Data_Directory, self.Obj_Filename, self.MeasureType, self.titles = data_dir, obj_fname, measure_type, titles
        self.Total_time , self.nTimeSeries, self.nInterval, self.nDataRealization = tot_time, nTime, nIntv, nDataRealization
        self.nTrain,  self.nParam, self.nMCSamples = nTrain, nParam, nMCSamples
        self.x_min, self.x_max = xmin, xmax
        self.verbose, self.NN, self.seed = verbose, NN, 787878
        self.ROMs_validation, self.time_sensitivity, self.err_option = rom_val, time_sensitivity, error_option
        # Load X, Y, Data
        self.training_data, self.raw_data, self.parameters = read_train_sim_results(self.Data_Directory, self.MeasureType, self.Obj_Filename, self.nColumn_obj, 
                                                                                    self.Total_time, self.nTimeSeries, self.nColumn_data, self.nInterval, 
                                                                                    self.nTrain, self.nParam, self.x_min, self.x_max, self.titles, self.verbose)
        # Extract reuslts
        self.data_train, self.x_train, self.x_train_scaled, self.y_train, self.y_train_scaled = self.training_data
        self.data_train_read_raw, self.data_train_read_raw0                                   = self.raw_data
        self.eps, self.nData, self.time_point                                                 = self.parameters
        # Train ROMS
        self.ROM_data, self.ROM_obj = rom_validation(self.ROMs_validation, self.rom_obj, self.rom_data, self.x_train_scaled, self.y_train, self.data_train, 
                                                     self.nTrain, self.nData, self.data_train_read_raw, self.NN, self.verbose)
        # Evaluate ROMS with MC
        np.random.seed(self.seed); self.mc_design = 2*np.random.rand(self.nMCSamples, self.nParam)-1
        self.mc_data, self.mc_obj = evaluate_MC_with_ROM(self.ROM_data, self.ROM_obj, self.data_train_read_raw, self.mc_design, self.nData, self.nMCSamples, self.verbose)
        # Calculation Uncertainty
        self.prior_mean, self.prior_p90mp10, self.synthetic_data, self.results = uncertainty_reduction(self.mc_obj, self.mc_data, self.ROM_data, self.data_train_read_raw, 
                                                                                                       self.MeasureType, self.nData, self.nDataRealization, self.nParam, 
                                                                                                       self.nColumn_data, self.x_min, self.x_max, self.eps, self.err_option, 
                                                                                                       self.time_sensitivity, self.verbose)
        self.post_p90mp10_mean, self.post_p90mp10_time, self.post_p90mp10_iData, self.post_mean, self.post_mean_iData, self.nSamples, self.mc_obj_post = self.results
        self.value = -(self.prior_p90mp10 - self.post_p90mp10_mean) #minimize this objective
        

############################## BASICS ##############################
def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    print('Checking Tensorflow Version:')
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print('Tensorflow version:', tf.__version__)
    print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
    print(tf.config.list_physical_devices())
    return None


############################## PLOTTING ##############################
def plot_train_data(datatrain, xtrain, ytrain, titles, MeasureType, Obj_filename, unc_params, figsize=(35,4)):
    plt.figure(figsize=figsize)
    plt.subplot(1,6,1); 
    plt.plot(datatrain, alpha=0.15, c='k', linewidth=2); plt.plot(datatrain.mean(1), 'c-', label='mean', linewidth=3)
    plt.legend(); plt.grid('on'); plt.title('Data Train'); plt.ylabel(titles[MeasureType]); plt.xlabel('Timestep')
    plt.subplot(1,6,2); plt.plot(ytrain); plt.title('Y Train'); plt.xlabel(Obj_filename[:-4])
    k=0
    for i in range(3,7):
        plt.subplot(1,6,i); plt.plot(xtrain[:,k], c='C{}'.format(k+1)); plt.title('X Train {}'.format(k)); plt.xlabel(unc_params[k]); k+=1
    plt.show()
    
def plot_loss(fit, title='', figsize=None):
    if figsize:
        plt.figure(figsize=figsize)
    loss, val = fit.history['loss'], fit.history['val_loss']
    epochs = len(loss)
    iterations = np.arange(epochs)
    plt.plot(iterations, loss, '-', label='loss')
    plt.plot(iterations, val, '-', label='validation loss')
    plt.title(title+' Training: Loss vs epochs'); plt.legend()
    plt.ylabel('Epochs'); plt.ylabel('Loss'); plt.xticks(iterations[::epochs//10])
    plt.show()
    
def make_rom_predictions(models, x_train_scaled, y_train_scaled, nTrain, figsize=(20,6), labels=['ANN','MARS','RFR'], plot=True):
    y_train_pred, y_test_pred = {}, {}
    mse_train,    mse_test    = {}, {}
    for i in range(len(models)):
        y_train_pred[i] = models[i].predict(x_train_scaled[:nTrain])
        y_test_pred[i]  = models[i].predict(x_train_scaled[nTrain:])
        mse_train[i]    = mean_squared_error(y_train_scaled[:nTrain], y_train_pred[i])
        mse_test[i]     = mean_squared_error(y_train_scaled[nTrain:], y_test_pred[i])
    if plot:
        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.axline([0,0],[1,1], c='r', linewidth=3); plt.grid('on')
        for i in range(len(models)):
            plt.scatter(y_train_scaled[:nTrain], y_train_pred[i], alpha=0.5, label='{}, MSE={:.4f}'.format(labels[i], mse_train[i]))  
        plt.title('TRAINING'); plt.xlabel('true'); plt.ylabel('pred'); plt.xlim([-1,1]); plt.ylim([-1,1]); plt.legend()
        plt.subplot(122)
        plt.axline([0,0],[1,1], c='r', linewidth=3); plt.grid('on')
        for i in range(len(models)):
            plt.scatter(y_train_scaled[nTrain:], y_test_pred[i], alpha=0.5, label='{}, MSE={:.4f}'.format(labels[i], mse_test[i]))
        plt.title('TESTING'); plt.xlabel('true'); plt.ylabel('pred'); plt.xlim([-1,1]); plt.ylim([-1,1]); plt.legend()
        plt.show()
    return y_train_pred, y_test_pred, mse_train, mse_test

############################## PROXY MODEL ##############################
def make_proxy(reg=L1(1e-5), drop=0.2, opt='adam', loss='mse'):
    def dense_block(inp, units):
        _ = Dense(units, kernel_regularizer=reg)(inp)
        _ = BatchNormalization()(_)
        _ = PReLU()(_)
        _ = Dropout(drop)(_)
        return _
    inp = Input(shape=(4))
    x = dense_block(inp, 64)
    x = dense_block(x, 128)
    x = dense_block(x, 32)
    x = dense_block(x, 16)
    out = Dense(1, activation='tanh')(x)
    proxy = Model(inp, out)
    proxy.compile(optimizer=opt, loss=loss, metrics=['mse','mae'])
    return proxy

############################## MAKE FUNCTIONS ##############################
def read_train_sim_results(Data_Directory, MeasureType, Obj_filename, nColumn_obj, Total_time, nTimeSeries, nColumn_data, nInterval, nTrain, nParam, 
                           x_min, x_max, titles, verbose=True):
    '''
    data_train: measure_type at node nColumn_data
    x_train:    perm4/perm5/perm6/kmult -> random permeabilities for each realization
    y_train:    cumulative co2 leakage in kiloTons for each realization
    '''
    # Read data (data_train) from fehm training simulations
    if MeasureType == 1:    # for pressure
        data_filename = 'run_presWAT.his'            # monitoring data file name
        eps = 0.002                                  # History match error tolerance
    elif MeasureType == 2:  # for liquid CO2 saturation
        data_filename = 'run_co2sl.his'
        eps =  0.05 
    elif MeasureType == 3:  # for tempeture
        data_filename = 'run_temp.his' 
        eps = 0.002 
    elif MeasureType == 4:  # for pressure+CO2 saturation
        data_filename = ['run_presWAT.his','run_co2sl.his']
        eps = [0.002, 0.05]
    else:
        print("No such measurement option, optimization will be terminated")
    print('Measure Type: {} | Data File Name: {} | Description: {} | nColData: {} | HM-epsilon: {}'.format(MeasureType, data_filename, titles[MeasureType], nColumn_data, eps))
    nData_read =  nTimeSeries*len(nColumn_data)*nInterval  # The actural number of data read from simulation output
    nData      =  nTimeSeries*len(nColumn_data)            # The number of measurement data points
    data_train_read_raw  = np.zeros((nTrain,nData_read))
    data_train_read_raw0 = np.zeros((nTrain,len(nColumn_data)))
    data_train_read      = np.zeros((nData_read,nTrain))
    data_train           = np.zeros((nData,nTrain))
    time_set             = []
    time_point_read      = np.zeros(nTimeSeries*nInterval+1)
    time_point           = np.zeros(nTimeSeries+1)
    for i in range(0, len(nColumn_data)):
        for itrain in range(0, nTrain):
            if MeasureType==4:
                train_data_filename = os.path.join(Data_Directory, 'workdir.{}'.format(itrain+1), data_filename[i])
            else:
                train_data_filename = os.path.join(Data_Directory, 'workdir.{}'.format(itrain+1), data_filename)
            count = 0
            for lines in open(train_data_filename):
                count += 1
                lines = lines.rstrip('\n')
                lines = lines.split(None, 1)
                times, features = lines
                dim_count = 0
                if count==5:
                    for e in features.split():
                        dim_count += 1
                        val = e
                        if dim_count == nColumn_data[i]-1:
                            xi = float(val)
                    data_train_read_raw0[itrain][i] = xi
                if count>5:
                    for e in features.split():
                        dim_count += 1
                        val = e
                        if dim_count == nColumn_data[i]-1:
                            xi = float(val)                        
                    time_set += [float(times)]
                    data_train_read_raw[itrain][count-6+i*nTimeSeries*nInterval] = xi
    time_point_read[1:nTimeSeries*nInterval+1] = time_set[:nTimeSeries*nInterval]
    data_train_read = data_train_read_raw.T
    if nData_read==nData:
        data_train = data_train_read
        time_point = time_point_read
    else:
        multiplier=nData_read/nData
        for i in range(0,nData):
            data_train[i] = data_train_read[multiplier*i+1]
            if i<=nTimeSeries:
                time_point[i] = time_point_read[multiplier*i]
        time_point[nTimeSeries] = Total_time
    if verbose:
        print('\nRead data from fehm traning simulations: Done!')
        print('Time Points: {}'.format(time_point.shape))
        print('Data Train: {}'.format(data_train.shape))    
    # Read response of interest (y_train) from fehm training simulations
    y_train = np.zeros(nTrain)
    for itrain in range(0, nTrain):
        train_filename = os.path.join(Data_Directory, 'workdir.{}'.format(itrain+1), Obj_filename)
        with open(train_filename) as f1:
            lines1 = f1.readlines()
            lines1 = [line.rstrip('\n') for line in open(train_filename)]
            lastlines = lines1[-1]
            Objective = float(lastlines.split()[nColumn_obj-1])
            y_train[itrain] = Objective
    if verbose:
        print('\nRead y_train from fehm traning simulations: Done!')
        print('y_train: {}'.format(y_train.shape))
    # Scale y_train to [-1,1]
    y_train_scaled = 2*(y_train-y_train.min()) / (y_train.max()-y_train.min())-1
    if verbose:
        print('Rescale y_train to [-1,1]: Done!')
    # Read data (x_train) from 'sample.txt' file
    m       = matk()
    ss      = m.read_sampleset('sample.txt')
    x_train = ss.samples.values
    if verbose:
        print('\nRead x_train from fehm traning simulations: Done!')
        print('x_train: {}'.format(x_train.shape))
    # Scale x_train to [-1,1]
    x_train_scaled = np.zeros((nTrain,nParam))
    for i in range(0,nTrain):
        for j in range(0,nParam):
            x_train_scaled[i][j] = 2*(x_train[i][j] - x_min[j])/(x_max[j] - x_min[j]) - 1
    if verbose:
        print('Rescale x_train to [-1,1]: Done!')
    # Return data
    training_data = [data_train, x_train, x_train_scaled, y_train, y_train_scaled]
    raw_data      = [data_train_read_raw, data_train_read_raw0]
    parameters    = [eps, nData, time_point]
    return training_data, raw_data, parameters

def rom_validation(ROMs_validation, ROM_o, ROM_d, x_train_scaled, y_train, data_train, nTrain, nData, data_train_read_raw, NN=False, verbose=True):
    ## Step 3: 10-fold cross-validation of ROMs
    if ROMs_validation==0:
        ## Step 4: Construct the Mars ROMs for data and response of interest
        # ROMs for data points
        ROM_data = {}
        for iData in range(0, nData):
            ROM_data[iData] = ROM_d
            if NN:
                ROM_data[iData].fit(x_train_scaled[:nTrain], MinMaxScaler((-1,1)).fit_transform(data_train[iData].reshape(-1,1)),
                                    epochs=300, verbose=0, batch_size=60, validation_split=0.2)
            else:
                ROM_data[iData].fit(x_train_scaled[:nTrain], data_train[iData])
        if verbose:
            print('Build the ROMs for data points: Done!')
        # ROMs for obj
        ROM_obj = ROM_o
        if NN:
            ROM_obj.fit(x_train_scaled[:nTrain], MinMaxScaler((-1,1)).fit_transform(y_train.reshape(-1,1)),
                        epochs=300, verbose=0, batch_size=60, validation_split=0.2)
        else:
            ROM_obj.fit(x_train_scaled[:nTrain], y_train)
        if verbose:
            print('Build the ROMs for objective of interests: Done!')
    
    elif ROMs_validation==1:
        if verbose:
            print('ROMs accuracy validation: 10-fold cross-validation')
        # ROMs validation for objs
        Interval     = int(nTrain/10)
        predict_obj  = np.zeros(nTrain)
        predict_data = np.zeros((nData,nTrain))
        for i in range(0,10): # 10-fold cross-validation
            x_train_scaled_v = x_train_scaled.tolist()
            y_train_v        = y_train.tolist()
            data_train_raw_v = data_train.T.tolist()
            del x_train_scaled_v[Interval*i:Interval*(i+1)]
            del y_train_v[Interval*i:Interval*(i+1)]
            del data_train_raw_v[Interval*i:Interval*(i+1)]
            x_train_scaled_v = np.array(x_train_scaled_v)
            data_train_v     = np.array(data_train_raw_v).T
            data_train_v1    = data_train_v
            # build ROMs
            ROM_obj = ROM_o   
            ROM_obj.fit(x_train_scaled_v, y_train_v)
            ROM_data = {}
            for iData in range(0,nData):
                ROM_data[iData] = ROM_d
                ROM_data[iData].fit(x_train_scaled_v, data_train_v1[iData])            
            # predict   
            predict_obj[Interval*i:Interval*(i+1)] = ROM_obj.predict(x_train_scaled[Interval*i:Interval*(i+1)])
            for iData in range(0,nData):
                predict_data[iData][Interval*i:Interval*(i+1)] = ROM_data[iData].predict(x_train_scaled[Interval*i:Interval*(i+1)])
        for i in range(0,nData):
            for j in range(0,nTrain):
                if predict_data[i][j]<np.amin(data_train_read_raw):
                    predict_data[i][j] = np.amin(data_train_read_raw)
                if predict_data[i][j]>np.amax(data_train_read_raw):
                    predict_data[i][j] = np.amax(data_train_read_raw)
        for i in range(0,nTrain):
            if predict_obj[i]<0:
                predict_obj[i] = 0
        CorreCoeff = np.corrcoef(predict_obj,y_train)[0][1]
        if verbose:
            print('The correlation coefficient between the true values and the predicted values for Obj ROMs: '+ str(CorreCoeff))
        # plot1
        CorreCoef_data = np.zeros(nData)
        for i in range (0,nData):
            CorreCoeff2 = np.corrcoef(predict_data[i],data_train[i])[0][1]      
            if verbose:
                print('The correlation coefficient between the true values and the predicted values for ROM data point '+str(i+1)+': '+ str(CorreCoeff2))
            CorreCoef_data[i] = CorreCoeff2
            plt.figure()
            plt.scatter(predict_data[i],data_train[i],marker='*',color='blue')
            real_max0, pred_max0 = max(data_train[i]),       max(predict_data[i])
            real_min0, pred_min0 = min(data_train[i]),       min(predict_data[i])
            maxV0,     minV0     = max(real_max0,pred_max0), min(real_min0,pred_min0)
            plt.plot([minV0,maxV0],[minV0,maxV0],ls="-",c="0.3")
            plt.xlabel('ROM Prediction (MPa)',fontsize=16,fontweight="bold"); plt.ylabel('True Value from Simulation (MPa)',fontsize=16,fontweight="bold")
            plt.rc('xtick',labelsize=14); plt.rc('ytick',labelsize=14)
            plt.xlim([minV0,maxV0]); plt.ylim([minV0,maxV0])
            figname = 'figures/ROMsData-validation'+str(i+1); plt.savefig(figname,bbox_inches='tight') #; plt.show()
            plt.close()
        plt.figure()
        plt.scatter(np.arange(1,nData+1,1),CorreCoef_data,marker='d',color='red')
        plt.xlabel('Monitoring Data Point',fontsize=16,fontweight="bold"); plt.ylabel('Correlation Coefficient',fontsize=16,fontweight="bold")
        plt.rc('xtick',labelsize=14); plt.rc('ytick',labelsize=14)
        plt.xlim([0,nData+1]); plt.ylim([0.9,1])
        plt.savefig("figures/CorrelationCoeff_data",bbox_inches='tight'); plt.show()
        # plot2
        plt.figure()
        plt.scatter(predict_obj/1e6,y_train/1e6,marker='*',color='blue')
        real_max1, pred_max1 = max(y_train/1e6), max(predict_obj/1e6)
        maxV1 = max(real_max1,pred_max1)+5
        plt.plot([0,maxV1],[0,maxV1],ls="-",c="0.3")
        plt.xlabel('ROM Prediction (Kt)',fontsize=16,fontweight="bold"); plt.ylabel('True Value from Simulation (Kt)',fontsize=16,fontweight="bold")
        plt.rc('xtick',labelsize=14); plt.rc('ytick',labelsize=14)
        plt.xlim([0,maxV1]); plt.ylim([0,maxV1])
        plt.savefig("figures/ROMs-obj",bbox_inches='tight'); plt.show()
    
    else:
        print('Please Select ROMs_validation={0,1}!')
    return ROM_data, ROM_obj

def evaluate_MC_with_ROM(ROM_data, ROM_obj, data_train_read_raw, mc_design, nData, nMCSamples, verbose=True):
    ## Step 6: Evaluate the MC samples using the built ROMs for data points/objs
    if verbose:
        print('Evaluating Monte Carlo samples: ing... ing ...')
    # data
    mc_data = np.zeros((nData, nMCSamples))
    for iData in range(0, nData):
        mc_data[iData] = ROM_data[iData].predict(mc_design).squeeze()
        for iMCSamples in range(0,nMCSamples):
            if mc_data[iData][iMCSamples]<np.amin(data_train_read_raw):
                mc_data[iData][iMCSamples] = np.amin(data_train_read_raw)
            if mc_data[iData][iMCSamples]>np.amax(data_train_read_raw):
                mc_data[iData][iMCSamples] = np.amax(data_train_read_raw)
    # objective
    mc_obj = np.zeros(nMCSamples)   
    mc_obj = ROM_obj.predict(mc_design).squeeze()
    for i in range(0,nMCSamples):
        if mc_obj[i]<0:
            mc_obj[i] = 0
    if verbose:
        print('Evaluate the Monte Carlo samples: Done!')
    return mc_data, mc_obj
    
def uncertainty_reduction(mc_obj, mc_data, ROM_data, data_train_read_raw, MeasureType, nData, nDataRealization, nParam, nColumn_data, 
                          x_min, x_max, eps, err_option, time_sensitivity, verbose=True):
    prior_mean    = sum(mc_obj)/len(mc_obj)
    prior_p90mp10 = uncertaintyMetric(mc_obj)
    # Generate synthetic data
    #synthetic_data = mc_data.T[:nDataRealization]
    def fehm(p):
        print('')
    p = matk(model=fehm)
    p.add_par('perm4',min=x_min[0], max=x_max[0])
    p.add_par('perm5',min=x_min[1], max=x_max[1])
    p.add_par('perm6',min=x_min[2], max=x_max[2])
    p.add_par('kmult',min=x_min[3], max=x_max[3])
    s = p.lhs(siz=nDataRealization, seed=1000)
    LHSamples        = s.samples.values
    LHSamples_scaled = np.zeros_like(LHSamples)
    for i in range(0,nDataRealization):
        for j in range(0,nParam):
            LHSamples_scaled[i][j] = 2*(LHSamples[i][j] - x_min[j]) / (x_max[j] - x_min[j]) - 1
    synthetic_data_raw = np.zeros((nData, nDataRealization))
    for iData in range(0, nData):
        synthetic_data_raw[iData] = ROM_data[iData].predict(LHSamples_scaled).squeeze()
    synthetic_data = synthetic_data_raw.T
    for i in range(0,nDataRealization):
        for j in range(0,nData):
            if synthetic_data[i][j]<np.amin(data_train_read_raw):
                synthetic_data[i][j] = np.amin(data_train_read_raw)
            if synthetic_data[i][j]>np.amax(data_train_read_raw):
                synthetic_data[i][j] = np.amax(data_train_read_raw)
    if verbose:
        print('Generate synthetic monitoring data: Done!')
    # Calculate posterior metrics
    results = Uncertainty_calc(mc_data, synthetic_data, mc_obj, err_option, eps, MeasureType, time_sensitivity, len(nColumn_data), verbose)
    if verbose:
        print('\nData assimilation is done!')
    return prior_mean, prior_p90mp10, synthetic_data, results

def make_post_processing(post_processing, results, synthetic_data, mc_obj, prior_p90mp10, data_train_read_raw0, time_point, Total_time, nDataRealization, nMCSamples, nColumn_data, nTimeSeries):
    if post_processing==1:
        print('Post-processing')
        post_p90mp10_mean, post_p90mp10_time, post_p90mp10_iData, post_mean, post_mean_iData, nSamples, mc_obj_post = results
        
        # plot posterior uncertainty change with time
        U = np.insert(post_p90mp10_time/1e6,0,prior_p90mp10/1e6)
        #T = np.insert(time_point,0,0)
        print('The prior uncertainty: {:.3f}'.format(U[0]))
        print('The end point posterior uncertainty: {}'.format(U[len(U)-1]))

        plt.figure()
        plt.plot(np.array(time_point)/360,U,marker='o',markersize=4,markerfacecolor='none',color='red',label='M1')
        #plt.plot(np.array(time_point)/360,U_2,marker='^',markersize=4,markerfacecolor='none',color='blue',label='M2')
        #plt.plot(np.array(time_point)/360,U_4,marker='x',markersize=4,color='black',label='M1+M2')
        #plt.plot(np.array(time_point)/360,U_3,marker='+',markersize=5,color='green',label='M1+M2+M3')    
        plt.xlabel('Time (Years)',fontsize=16,fontweight="bold")
        plt.ylabel('U of Cumulative CO$_2$ Leak (Kt)',fontsize=16,fontweight="bold")
        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)
        plt.xlim([0,Total_time/360])
        plt.ylim([0,40])
        plt.legend(loc='upper right',fontsize=14)
        plt.savefig("figures/UR_ab.png",bbox_inches='tight')
        plt.show()
       
        # plot U values of prior and posterior distributions
        plt.figure()
        plt.scatter(1,prior_p90mp10/1e6,marker='d',color='red')
        xl=np.zeros(nDataRealization)
        for i in range(0,nDataRealization):
            xl[i]= 2
        plt.scatter(xl,post_p90mp10_iData/1e6,marker='d',color='blue')
        plt.plot((0.7,2.3),(post_p90mp10_mean/1e6,post_p90mp10_mean/1e6),ls='--',color='blue')
        #plt.plot((1.1,1.1),(post_p90mp10_mean,prior_p90mp10),ls='-',color='red')
        plt.annotate('',xy=(1.1,post_p90mp10_mean/1e6),xycoords='data',xytext=(1.1,prior_p90mp10/1e6),textcoords='data',arrowprops={'arrowstyle':'<->','color':'red','lw':'1.5'})
        plt.figtext(0.45,0.7,"Uncertainty Reduction",fontsize=14, fontweight='bold', color='red')
        #plt.xlabel('',fontsize=14)
        plt.ylabel('U of Cumulative CO$_2$ Leak (Kt)',fontsize=16,fontweight="bold")
        plt.xticks(np.arange(4),('','Prior','Posterior',''),fontsize=16,fontweight="bold")
        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)
        plt.ylim([0,40])
        #plt.legend(loc='center right')
        plt.savefig("figures/U of prior and posterior",bbox_inches='tight')
        plt.show()
        
        # plot CDF for prior and posterior distribution of obj
        plt.figure()
        num_bins =100
        hist, bin_edges = np.histogram(mc_obj/1e6, bins=num_bins)
        cdf = np.cumsum(hist)
        cdf1=np.zeros(num_bins)
        for i in range(0,num_bins):
            cdf1[i]=float(cdf[i])/nMCSamples
        plt.plot(bin_edges[1:],cdf1,'r',label="Prior")
        for i in range(0,len(mc_obj_post)):
            hist_post, bin_edges_post = np.histogram(mc_obj_post[i]/1e6, bins=num_bins)
            cdf_post = np.cumsum(hist_post)
            cdf1_post=np.zeros(num_bins)
            for j in range(0,num_bins):
                cdf1_post[j]=float(cdf_post[j])/len(mc_obj_post[i])
            if i==0:
                plt.plot(bin_edges_post[1:],cdf1_post,'b',label="Posterior")
            else:
                plt.plot(bin_edges_post[1:],cdf1_post,'b')
        plt.xlabel('Cumulative CO$_2$ Leak (Kt)',fontsize=16,fontweight="bold")
        plt.ylabel('CDF',fontsize=16,fontweight="bold")
        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)
        plt.legend(loc='upper left',fontsize=14)
        plt.savefig("figures/CDF.png",bbox_inches='tight')    
        plt.show()
        
        #plot histogram
        plt.figure()
        num_bins1=50
        plt.hist(mc_obj/1e6,bins=num_bins1,color='blue',label='Prior')
        plt.hist(mc_obj_post[0]/1e6,bins=num_bins1,color='orange',label='Posterior_R1')
        plt.hist(mc_obj_post[99]/1e6,bins=num_bins1,color='red',label='Posterior_R100')
        plt.xlabel('Cumulative CO$_2$ Leak (Kt)',fontsize=16,fontweight="bold")   
        plt.ylabel('Frequency',fontsize=16,fontweight="bold")
        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)
        plt.legend(loc='upper left')
        plt.savefig("figures/histogram.png",bbox_inches='tight')
        plt.show()
        
        # plot the data realization
        for iLoc in range(0, len(nColumn_data)):
            plt.figure()
            for i in range(0,nDataRealization):
                data_plot=np.insert(synthetic_data[i][nTimeSeries*iLoc:nTimeSeries*(iLoc+1)],0,data_train_read_raw0[0][iLoc])
                plt.plot(np.array(time_point)/360,data_plot)
            plt.xlabel('Time (Years)',fontsize=16,fontweight="bold")
            plt.ylabel('Monitoring Pressure (MPa)',fontsize=16,fontweight="bold")
            #plt.ylabel('CO$_2$ Saturation',fontsize=16,fontweight="bold")
            plt.rc('xtick',labelsize=14)
            plt.rc('ytick',labelsize=14)
            plt.xlim([0,Total_time/360])
            #plt.ylim([10.200,10.235])
            figname = 'figures/data_realizations_Loc_'+str(iLoc+1)
            plt.savefig(figname, bbox_inches='tight')
            plt.show()
            
        # plot number of samples remained
        plt.figure()
        plt.scatter(np.arange(1,nDataRealization+1,1),nSamples,marker='o',color='blue')
        plt.xlabel('Data Realization',fontsize=16,fontweight="bold")
        plt.ylabel('Samples Remained',fontsize=16,fontweight="bold")
        plt.rc('xtick',labelsize=14)
        plt.rc('ytick',labelsize=14)
        plt.xlim([0,nDataRealization])
        #plt.ylim([0,nMCSamples])
        plt.savefig("figures/samples_remained.png",bbox_inches='tight')
        plt.show()
        # plot boxplot for the number of samples remained
        #plt.figure()
        #data_box=[nSamples_1,nSamples]
        #plt.boxplot(data_box, 0, '')
        #plt.ylabel('Samples Remained',fontsize=16,fontweight="bold")
        #plt.rc('ytick',labelsize=14)
        #plt.xticks([1, 2], ['M1', 'M2'],fontsize=16,fontweight="bold")
        #plt.savefig("figures/boxplot_samples_remained.png",bbox_inches='tight')
        #plt.show()
        print('\nWorkflow: done successfully!\n\n')
    else:
        print('\nNo Post-processing - Workflow: done successfully!\n\n')
    return None