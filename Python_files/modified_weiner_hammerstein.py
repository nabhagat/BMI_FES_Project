# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:30:14 2016

@author: nikunj
"""
#import serial, io, time
#import sys, select
#import logging, os

import csv
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
#%matplotlib gtk
from scipy.signal import resample
import statsmodels.api as sm

#%% Load data
mat_data = scipy.io.loadmat("/home/nikunj/Data/PhD_Dissertation/NJBT_pilot_data/NJBT_ses1_stim_block4.mat",matlab_compatible = True)
stim_array = np.array(mat_data['NJBT_ses1_stim_block3'])

# Convert ADC counts to integers
# Force sensor equation = 20.21 x + 0.02775
force_output = (stim_array[:,2]*5/float(1023))*20.21 + 0.02775 # Unit: Newtons
stim_input = (stim_array[:,3]*5/float(1023))*15 # voltage divider ratio

Fs_orig = 100 # Sampling frequency
Fs_new = 5000  # Pulse_width = 200 usec, Fs = 5 KHz; for Pulse_width = 1ms, Fs = 2KHz
No_of_stim_pulses = 12 # In 1 burst - verify from plot
stim_freq = 20
hill_delay = 0.01 # Delay from Hill-Huxley model, assume 10 ms

# Select time interval for model testing
sample_nos = [60.00,70.00]
ind1 = np.where(stim_array[:,0] == sample_nos[0])[0]            # [0] added to get the value, np.where 
ind2 = np.where(stim_array[:,0] == sample_nos[1])[0]            # will return a tuple
force_output = force_output[ind1:ind2]
stim_input = stim_input[ind1:ind2]
stim_time = stim_array[ind1:ind2,0]

# Plot PSD of force signal; compute frequency with 95% power and high pass filter with this frequency cutoff
f_den,Pxx_den = sp.signal.welch(force_output,fs = Fs_orig, nperseg=256, noverlap=128, nfft = 256, detrend='constant',scaling='density')
#plt.subplot(2,1,1)
#plt.semilogy(f_den,Pxx_den)
#plt.ylim([0.5e-3, 10])
#plt.xlabel('frequency [Hz]')
#plt.ylabel('PSD [V**2/Hz]')

# Compute power vs frequency
total_signal_power = np.trapz(Pxx_den,f_den)
band_signal_power = np.zeros(f_den.shape)
for f_index,f in enumerate(f_den):
    band_signal_power[f_index] = np.trapz(Pxx_den[0:f_index], f_den[0:f_index])
    #print(str(f_index) + ', ' + str(band_signal_power[f_index]))

#plt.subplot(2,1,2)
#plt.plot(f_den,band_signal_power/total_signal_power,'-r')
#plt.xlabel('frequency [Hz]')
#plt.ylabel('Normalized Power')
min_low_pass_freq = f_den[band_signal_power == min(band_signal_power, key=lambda x:abs(x-0.99*total_signal_power))]
#plt.plot([min_low_pass_freq,min_low_pass_freq], [0, 1],'-k', linewidth = 2)
#plt.text(min_low_pass_freq-2,-0.1,str(min_low_pass_freq))
#plt.show()

# Low pass filter force output at 
b_lpf, a_lpf = sp.signal.butter(8, 10/float(Fs_orig/2), 'low', analog=False)
force_output_filt = sp.signal.filtfilt(b_lpf,a_lpf, force_output)
#plt.figure()
#plt.plot(stim_time,force_output,'-b',stim_time,force_output_filt,'-r')

#w,h = sp.signal.freqz(b_lpf,a = a_lpf)
#fig = plt.figure()
#plt.title('Digital filter frequency response')
#ax1 = fig.add_subplot(111)
#plt.plot(w*(Fs_orig/float(np.pi)/float(2)), 20 * np.log10(abs(h)), 'b')
#plt.ylabel('Amplitude [dB]', color='b')
#plt.xlabel('Frequency [rad/sample]')
#ax2 = ax1.twinx()
#angles = np.unwrap(np.angle(h))
#plt.plot(w, angles, 'g')
#plt.ylabel('Angle (radians)', color='g')
#plt.grid()
#plt.axis('tight')
#plt.grid(axis='both')
#plt.show()

# resample to 10KHz
force_output_orig = force_output
(force_output,samp_time) = resample(force_output_filt,(ind2 - ind1)*(Fs_new/Fs_orig),stim_time)      # Output is a tuple
(stim_input_orig,samp_time) = resample(stim_input,(ind2 - ind1)*(Fs_new/Fs_orig),stim_time)

#%% Create simulated stim_input from original before resampling the signal
tau = 1  # 1 sample, i.e. 10 ms @ 100 Hz
stim_width = int(200E-6*Fs_new) # 200 usec, change_here
tau_count = 0
stim_ON = False
stim_start_indexes = []
stim_stop_indexes = [] 

for ind,u in enumerate(stim_input):
    if stim_ON == True:
        if u <= 0:
            tau_count += 1
            if tau_count > tau:
                stim_ON = False # stimulation has stopped
                stim_stop_indexes.append(ind)
        else:
            tau_count = 0            
    else:
        tau_count = 0
        if u > 5:  # Stimulation has started
            stim_ON = True
            stim_start_indexes.append(ind)        
            
#stim_start_marker = np.zeros(stim_time.shape)
#stim_stop_marker = np.zeros(stim_time.shape)
#stim_start_marker[stim_start_indexes] = 5
#stim_stop_marker[stim_stop_indexes] = 5
#plt.figure()
#plt.plot(stim_time,stim_input,'-b')
#plt.hold(True)
#plt.plot(stim_time,stim_start_marker,'-r')
#plt.plot(stim_time,stim_stop_marker,'-g')
#plt.hold(False)

sim_stim_input = np.zeros(force_output.shape)
sig_amp = 40

for start_ind in stim_start_indexes:
    #print np.where(samp_time == stim_time[start_ind])[0]
    #print stim_time[start_ind], start_ind, samp_time[start_ind*20]
    pulse_width_ind = int(start_ind*Fs_new/Fs_orig) 
    for pulse_cnt in range(0,No_of_stim_pulses):
        #print pulse_width_ind
        sim_stim_input[pulse_width_ind:pulse_width_ind+int(200E-6*Fs_new)] = sig_amp # 
        pulse_width_ind += int(Fs_new/float(stim_freq)) # pulsewidth = 50ms for stim_freq = 20 Hz; change_here
        
plt.figure()
plt.plot(samp_time,force_output,'-r')
plt.hold(True)
plt.plot(samp_time,sim_stim_input,'-b')

#%% Compute the nonlinear poylnomial signal
def f_polynomial(U_k, B_q):
    if U_k.ndim == 1: # U_k is a vector...used for prediction
        f_beta1 = sum(B_q)
    else:
        f_beta1 = sum(B_q)*np.ones((len(U_k),1)) # U_k is a matrix...used for optimization
    f_beta2 = np.dot(U_k,B_q)
    f_beta3 = np.dot(U_k**2,B_q)
    f_beta4 = np.dot(U_k**3,B_q)
    return np.column_stack((f_beta1,f_beta2,f_beta3,f_beta4))

# Compute mean squared error (MSE) and predicted output
def compute_mse(y_true,u_input,A_q,B_q,beta):
    y_0 = np.zeros(np.shape(A_q))
    #u_0 = np.zeros(np.shape(B_q))
    g_u_0 = np.zeros(np.shape(B_q))
    y_pred = []
    
    for u in u_input:
        #u_0 = np.insert(u_0,[0],u)[0:-1]
        #f_u_k = f_polynomial(u_0,B_q)
        #y_0[0] = np.dot(y_0[1:],A_q[1:])+np.dot(f_u_k,beta) # y_k = -A(q)*[y(k-1),y(k-2),y(k-3)] + f_u_k*beta
        g_u_0 = np.insert(g_u_0,[0],np.dot([1,u,u**2,u**3],beta))[0:-1]        
        y_0[0] = np.dot(y_0[1:],A_q[1:])+np.dot(g_u_0,B_q) 
        y_pred.append(y_0[0])
        y_0 = np.roll(y_0,1) # rotate by a 1 to the right; will work as recalculating y_0[0] in every loop 
    
    residuals = y_true - y_pred
    mse_value = np.mean(sum(residuals**2))
    return y_pred, mse_value
    

#%% Iterative linear square optimization
A_q = np.array([1, 0.0,0.65])
na = range(1,len(A_q)) #na = [a1,a2,a3]; 1 - separate
B_q = np.array([0.5,0])
nb = range(1,len(B_q)) #nb = [b1,b2]; b0 - separate
beta = np.array([0,1,0.5,0.1])
training_length = 10000

time_int = samp_time[0:training_length]
Y_k = force_output[0:training_length]
for i in na:
    Y_k = np.column_stack((Y_k,np.insert(force_output[0:training_length],[0]*i,np.zeros((1,i)))[0:-i]))
        
U_k = sim_stim_input[0:training_length]
for i in nb:
    U_k = np.column_stack((U_k,np.insert(sim_stim_input[0:training_length],[0]*i,np.zeros((1,i)))[0:-i]))

# Evolution of parameters
A_q_all = A_q
B_q_all = B_q
beta_all = beta
mse_all = []

# Compute mean squared error (MSE) with initial values
y_pred, mse_initial = compute_mse(Y_k[:,0],U_k[:,0],A_q,B_q,beta)  
mse_all.append(mse_initial)

print("Initializing optimization loop: MSE = " + str(mse_initial))
plt.figure(), plt.hold('ON')
plt.plot(time_int,Y_k[:,0],'-b',linewidth = 2)
plt.plot(time_int,y_pred,'-k',linewidth = 1)

for j in range(1,101):
    #Step 1: Compute optimal beta
    F_U_k = f_polynomial(U_k,B_q) # Matrix with dim = Nx4
    nonlinear_model = sm.OLS(np.dot(Y_k,A_q),F_U_k)
    beta = nonlinear_model.fit().params 
    beta_all = np.row_stack((beta_all,beta))
    
    #Step 2: Optimize A_q, B_q
    g_u_k = np.dot(np.column_stack((np.ones(training_length),U_k[:,0],U_k[:,0]**2,U_k[:,0]**3)),beta)
    G_U_k = g_u_k
    for i in nb:
        G_U_k = np.column_stack((G_U_k,np.insert(g_u_k,[0]*i,np.zeros((1,i)))[0:-i]))
    
    Regressor_matrix = np.column_stack((Y_k[:,1:],G_U_k))
    linear_model = sm.OLS(Y_k[:,0],Regressor_matrix)
    linear_model_parameters = linear_model.fit().params
    A_q[1:] = linear_model_parameters[0:1]  #change
    A_q_all = np.row_stack((A_q_all,A_q))
    B_q = linear_model_parameters[2:]   #change
    B_q_all = np.row_stack((B_q_all,B_q))
    y_pred, mse_loop = compute_mse(Y_k[:,0],U_k[:,0],A_q,B_q,beta)  
    mse_all.append(mse_loop)
    
    print("Loop: " + str(j) + ", MSE = " + str(mse_all[-1]))
    #plt.plot(time_int,y_pred,'-k',linewidth = 1)
