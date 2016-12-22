# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell> Import the necessary project files

import csv
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
#matplotlib gtk
from scipy.signal import resample
import statsmodels.api as sm
import control as cntrl
#from sklearn.preprocessing import normalize
import fnmatch
import os
import scipy.optimize as sp_optimize
   
from impulse_recruitment_curve_functions import *

#func.clear_all()
# <codecell>    
# Function to read data from sim_input and impulse files and return as arrays
def read_csv_file_convert_data(input_filename, response_filename, data_dir = ''):
      
    if len(input_filename) != len(response_filename):
        raise TypeError("Unequal number of input and response files")
    
    input_data = []
    response_data = []
    
    # If multiple files referred then append data
    for file_name in input_filename:
        with open(data_dir + file_name,'r') as input_file:
            reader = csv.reader(input_file)
            reader.next() # Skip 1st row, which is header
            reader.next() # Skip 2nd row, which is header
            #input_data = [itr_row for itr_row in reader]
            for itr_row in reader:
                input_data.append(itr_row)
                
    for file_name in response_filename:
        with open(data_dir + file_name,'r') as response_file:
            reader = csv.reader(response_file)
            reader.next() # Skip 1st row, which is header
            reader.next() # Skip 2nd row, which is header
            #input_data = [itr_row for itr_row in reader]
            for itr_row in reader:
                response_data.append(itr_row)
    
    # convert list to array
    # Convert ADC counts to integers
    # Force sensor equation = 20.21 x + 0.02775
    response_data = np.array(response_data,float)
    response_data = (response_data*5/float(1023))*20.21 + 0.02775 # Unit: Newtons
    
    # convert list to array
    # Convert ADC counts to integers
    # Stim input equation = 15x (Voltage divider ratio = 1/15)
    input_data = np.array(input_data,float)
    input_data = (input_data*5/float(1023))*15 # Unit: Volts
    #stim_input_value = stim_input[:,1]
    #plt.plot(stim_input[0:32,1],'-or')
    #plt.hold("ON")
    #plt.plot(stim_input[32:,1],'-ob')
    
    
    print('Files read, input size = ' + str(len(input_data)) + ', response size = ' + str(len(response_data))) 
                
    return input_data, response_data
    
# <codecell>
# Subject information and global variables
Subject_name = 'NJBT'
muscle_name = 'FD'
impulse_block_nos = [1,2] # Blocks 1 and 2 for data collected on 10-12-2016
cv_block_nos = [1]
use_normalized_data_for_model_training = False
use_continuous_time_delay = True
data_files_dir = '../../BMI_FES_Project/Python_files/'

Fs_force = 100 # Sampling frequency
Fs_stim = 4000  # Pulse_width = 500 usec, Fs = 4 KHz;
No_of_stim_pulses = 12  #  In 0.5 sec cue interval
cv_stim_freq = 20

impulse_response_filename = []  #force response during impulse measurement
impulse_input_filename = []     #stimulation input during impulse measurement
cv_response_filename = []       #force response during cross-validation
cv_input_filename = []          #stimulation input during cross-validation

# Match filenames with files present in the directory
for file in os.listdir(data_files_dir):
    # impulse_start_response; impulse_start_input
    for i,val in enumerate(impulse_block_nos):
        if fnmatch.fnmatch(file, Subject_name + '_' + muscle_name + '_impulse_response_block' + 
                           str(impulse_block_nos[i]) + '*.txt'):
            impulse_response_filename.append(file)
        if fnmatch.fnmatch(file, Subject_name + '_' + muscle_name + '_impulse_input_block' + 
                           str(impulse_block_nos[i]) + '*.txt'):
            impulse_input_filename.append(file)       
                  
    # cv_response, cv_input
    for i,val in enumerate(cv_block_nos):
        if fnmatch.fnmatch(file, Subject_name + '_' + muscle_name + '_cv_response_block' + 
                           str(cv_block_nos[i]) + '*.txt'):
            cv_response_filename.append(file)
        if fnmatch.fnmatch(file, Subject_name + '_' + muscle_name + '_cv_input_block' + 
                           str(cv_block_nos[i]) + '*.txt'):
            cv_input_filename.append(file)        

# Read impulse response data, convert units 
impulse_input, impulse_response = read_csv_file_convert_data(impulse_input_filename, impulse_response_filename, data_files_dir)
cv_stim_input, cv_force_response = read_csv_file_convert_data(cv_input_filename, cv_response_filename, data_files_dir)

cv_t_force = np.arange(0, cv_force_response.shape[1]/float(Fs_force), 1/float(Fs_force))
cv_t_stim = np.arange(0, cv_stim_input.shape[1]/float(Fs_stim), 1/float(Fs_stim))

# Normalize measured force response
cv_force_response_peak, cv_force_response_norm, cv_force_response_baseline_correct = peak_baseline_normalize_force(cv_force_response)

# For observed traces, find force traces with peak >= 3 N - Why?
cv_force_response = np.delete(cv_force_response,np.where(cv_force_response_peak < 3),axis=0)
cv_stim_input = np.delete(cv_stim_input,np.where(cv_force_response_peak < 3),axis=0)

# Normalize stim input and force response from observed data
cv_force_response_peak, cv_force_response_norm, cv_force_response_baseline_correct = func.peak_baseline_normalize_force(cv_force_response)
#obs_force_response_peak, obs_force_response_norm, obs_force_response_baseline_correct = peak_baseline_normalize_force(obs_force_response)
#obs_stim_input_norm = np.zeros_like(obs_stim_input)
#for n_t,u_t in enumerate(obs_stim_input):
#    obs_stim_input_norm[n_t,:] = obs_stim_input[n_t,:]/float(abs(obs_stim_input[n_t,:]).max())

# Upsample force trace and then reduce force trace and stim input to 1 sec
if use_normalized_data_for_model_training == True: 
    force_interp_func = sp.interpolate.interp1d(obs_t_force,obs_force_response_norm)        
    obs_t_reqd = np.arange(0.0, 1.24, 1/float(Fs_stim))
    obs_force_response_resamp = force_interp_func(obs_t_reqd)
    obs_stim_input_resamp = np.column_stack((obs_stim_input_norm, np.zeros((obs_stim_input.shape[0],1460)))) # 2500 + 1550 = 4000 samples ~ 1 sec 
else:
    force_interp_func = sp.interpolate.interp1d(obs_t_force,obs_force_response)        
    obs_t_reqd = np.arange(0.0, 1.24, 1/float(Fs_stim))
    obs_force_response_resamp = force_interp_func(obs_t_reqd)
    obs_stim_input_resamp = np.column_stack((obs_stim_input, np.zeros((obs_stim_input.shape[0],1460)))) # 2500 + 1550 = 4000 samples ~ 1 sec 


# <codecell>
# Transfer functions
# Pade's approximation of continuous time delay
(delay_num,delay_den) = cntrl.pade(50E-3,2) # 2nd order approximation
H_s_delay_22 = cntrl.tf(delay_num,delay_den)
H_zoh_delay = cntrl.matlab.c2d(H_s_delay_22,1/float(Fs_stim),method='zoh') # Compute discrete time equivalent
#obs_stim_input_resamp_with_delay = np.zeros_like(obs_stim_input_resamp)

w_n = 60 #100 #1/float(5E-3)
H_s_damp2 = cntrl.tf(1,[1,2*w_n,w_n**2 + (2*np.pi*0)**2])
H_z_damp2 = cntrl.matlab.c2d(H_s_damp2,1/float(4000))


# <codecell>

#(c1,c2,c3) = popt_start
#(c1,c2) = popt_start
for j,s_t in enumerate(obs_stim_input_resamp):
    #f_s_t = f_IRC_2(s_t,c1,c2)
    sim_time,y_val,x0 = cntrl.forced_response(H_s_delay_22*H_s_damp2,obs_t_reqd, U=s_t, X0 = 0.0)
    obs_stim_input_resamp_with_delay[j,:] = y_val.T


# Concatenate observed force and stim input trials
obs_force_all = obs_force_response_resamp.flatten()
if use_continuous_time_delay == True:
    obs_stim_input_all = obs_stim_input_resamp_with_delay.flatten()
else:
    obs_stim_input_all = obs_stim_input_resamp.flatten()
obs_t_all = np.arange(0,len(obs_stim_input_all)/float(Fs_stim),1/float(Fs_stim))

# Perform baseline correction, compute peak force and normalize
impulse_response_peak, impulse_response_norm, impulse_response_baseline_correct = peak_baseline_normalize_force(impulse_response)
impulse_block_mid_peak, impulse_block_mid_norm, impulse_block_mid_baseline_correct = peak_baseline_normalize_force(impulse_block_mid)
impulse_block_end_peak, impulse_block_end_norm, impulse_block_end_baseline_correct = peak_baseline_normalize_force(impulse_block_end)


plt.figure()
# Plot the Recrutiment curve
plt.subplot(1,2,1)
plt.plot(impulse_input[:,1],impulse_response_peak,'ob')
(c1,c2,c3) = popt_start
u_sim = np.arange(0,71,1)
plt.plot(u_sim,f_IRC_3(u_sim, c1,c2,c3),'-b')
plt.xlabel('Stimulus input (V)')
plt.ylabel('Peak impulse response (N)')
plt.title('Isometric Recrutiment Curve')
plt.ylim([0,4.0])

# Plot normalized average impulse response 
#plt.subplot(1,2,2)
#plt.plot(np.arange(0,1,1/float(Fs_force)),np.mean(impulse_response_norm,axis=0),'-b')
#plt.grid('on')
#plt.xlabel('Time (s)')
#plt.ylabel('Norm. avg. impluse response (N)')
#plt.title('Linear dynamical system')

# Plot baseline corrected impulse reponse 
plt.subplot(1,2,2)
plt.plot(np.arange(0,1,1/float(Fs_force)),impulse_response_baseline_correct.T,'-b')
plt.grid('on')
plt.xlabel('Time (s)')
plt.ylabel('Baseline corr. impluse responses (N)')

#plt.figure()
#plt.subplot(1,2,1)
#plt.plot(cv_t_force,cv_force_response.T)
#plt.grid('on')
#plt.xlabel('Time (s)')
#plt.ylabel('Force response (N)')

#plt.subplot(1,2,2)
#plt.plot(cv_t_stim,cv_stim_input.T)
#plt.xlim([0,2])
#plt.grid('on')
#plt.xlabel('Time (s)')
#plt.ylabel('Stim input (V)')
plt.figure()
plt.plot(obs_stim_input_all)
plt.plot(obs_force_all,'-r')

# <codecell>

# Optimization of IRC parameters
lb = np.array([0,-np.inf])
ub = np.array([np.inf,0])

# Lavenberg-Marquardt is used for unconstrained optimization
(popt_start,pcov_start) = sp.optimize.curve_fit(f_IRC_3, impulse_input[:,1], impulse_response_peak,method='lm')
#(popt_mid,pcov_mid) = sp.optimize.curve_fit(f_IRC_3, stim_input_block_mid[:,1], impulse_block_mid_peak,method='lm')
(popt_end,pcov_end) = sp.optimize.curve_fit(f_IRC_3, stim_input_block_end[:,1], impulse_block_end_peak,method='lm')

# Use trf method for constrained optimization
#(popt_start,pcov_start) = sp.optimize.curve_fit(f_IRC_2, impulse_input[:,1], impulse_response_peak, bounds= (lb,ub),method='trf')
#(popt_mid,pcov_mid) = sp.optimize.curve_fit(f_IRC_2, stim_input_block_mid[:,1], impulse_block_mid_peak, bounds= (lb,ub),method='trf')
#(popt_end,pcov_end) = sp.optimize.curve_fit(f_IRC_2, stim_input_block_end[:,1], impulse_block_end_peak, bounds= (lb,ub),method='trf')

print 'Start parameters'
print popt_start, pcov_start

#print 'Mid parameters'
#print popt_mid, pcov_mid

print 'End parameters'
print popt_end, pcov_end

# <codecell>

plt.figure();
plt.hold("ON")
(c1,c2,c3) = popt_start
plt.plot(u_sim,f_IRC_3(u_sim, c1,c2,c3),'-b')
#(c1,c2,c3) = popt_mid
#plt.plot(u_sim,f_IRC_3(u_sim, c1,c2,c2),'-b')
(c1,c2,c3) = popt_end
plt.plot(u_sim,f_IRC_3(u_sim, c1,c2,c3),'-r')
#plt.ylim([0,5])
#plt.plot(impulse_input[:,1],impulse_response_peak,'ob')
plt.plot(impulse_input[:,1],impulse_response_peak,'ob')
#plt.plot(stim_input_block_mid[:,1],impulse_block_mid_peak,'ob')
plt.plot(stim_input_block_end[:,1],impulse_block_end_peak,'or')
u_sim = np.arange(0,70,1)

plt.xlabel('Stimulus input (V)')
plt.ylabel('Peak impulse response (N)')
plt.title('Isometric Recruitment Curve')
plt.ylim([0,4.0])
plt.legend({'Before stimulation','After stimulation'},loc = 2)
#(c1,c2) = popt_start
#plt.plot(u_sim,f_IRC_2(u_sim, c1,c2),'-k')
#(c1,c2) = popt_start
#plt.plot(u_sim,f_IRC_2(u_sim, c1,c2),'-b')
#(c1,c2) = popt_end
#plt.plot(u_sim,f_IRC_2(u_sim, c1,c2),'-r')

# <codecell>

# Pade's approximation to continuous time delay
tau = 50E-3
(delay_num,delay_den) = cntrl.pade(tau,2)
#delay_num, delay_den
H_s_delay_22 = cntrl.tf(delay_num,delay_den)
H_s_delay_12 = cntrl.tf([-2/float(tau), 6/float(tau**2)],[1,4/float(tau),6/float(tau**2)])
H_s_delay_23 = cntrl.tf([3/float(tau),-24/float(tau**2), 60/float(tau**3)],
                        [1,9/float(tau),36/float(tau**2),60/float(tau**3)])

u_t = np.column_stack((np.zeros((1,50)),np.ones((1,100))))
t_t = np.arange(0.0, 1.5, 0.01)
(t_t,yout_22,x0) = cntrl.forced_response(H_s_delay_22, t_t, U=u_t,X0=0.0)
(t_t,yout_12,x0) = cntrl.forced_response(H_s_delay_12, t_t, U=u_t,X0=0.0)
(t_t,yout_23,x0) = cntrl.forced_response(H_s_delay_23, t_t, U=u_t,X0=0.0)
plt.figure()
plt.plot(t_t,yout_22,'-b',t_t,u_t.T,'-k',t_t,yout_12,'-r',t_t,yout_23,'-m')

# <codecell>

tau = 50E-3
(delay_num,delay_den) = cntrl.pade(50E-3,2) # 2nd order approximation
H_s_delay_22 = cntrl.tf(delay_num,delay_den)
H_s_delay_12 = cntrl.tf([-2/float(tau), 6/float(tau**2)],[1,4/float(tau),6/float(tau**2)])

w_n = 60 #100 #1/float(5E-3)
H_s_damp2 = cntrl.tf(1,[1,2*w_n,w_n**2 + (2*np.pi*0)**2])
p1 = 7
#p2 = 50
#H_s_damp2 = cntrl.tf([1],[1,50+50,50*50])
H_s_damp3 = cntrl.tf(1,[1,p1])
impulse_t = np.arange(0,1,1/float(Fs_force))
(sim_t,sim_impulse_response) = cntrl.impulse_response(H_s_delay_12*H_s_damp2*H_s_damp3, T = impulse_t) 

avg_sig = np.mean(impulse_response_norm,axis=0)
pred_impulse_response = sim_impulse_response/float(sim_impulse_response.max())

#plt.figure()
#plt.plot(sim_t,np.hstack((np.zeros(2),pred_impulse_response[:-2])))
plt.plot(impulse_t,impulse_response_norm[12:19,:].T,'-c')
plt.plot(sim_t,pred_impulse_response,'-r')
plt.plot(impulse_t,avg_sig,'-b')

(lags,corr_val) = xcorr(avg_sig,sim_impulse_response, 10)
#plt.plot(lags,corr_val,'-ob')
lags[corr_val.argmax()]

# <codecell>

plt.figure()
plt.subplot(1,2,1)
plt.plot(cv_t_force,cv_force_response.T)
plt.grid('on')
plt.xlabel('Time (s)')
plt.ylabel('Force response (N)')

plt.subplot(1,2,2)
plt.plot(cv_t_stim,cv_stim_input.T)
plt.xlim([0,2])
plt.grid('on')
plt.xlabel('Time (s)')
plt.ylabel('Stim input (V)')

#plt.figure()
#plt.plot(1E3*obs_stim_input_all)
#plt.plot(obs_force_all,'-r')

# <codecell>

# Perform least squares optimization
#  y_k = b1*u_k-1 + b2*u_k-2 - a1*y_k-1 - a2*y_k-2
# No. of samples n = 4000, no. of parameters m = 4
# A = n*m; X = m*1; b = y_k = n*1

#force_signal = cv_force_response_norm[10,:]
#force_interp_func = sp.interpolate.interp1d(cv_t_force,force_signal)
#resampled_time = np.arange(0.0, 1.0, 1/float(Fs_stim))
#y_k = force_interp_func(resampled_time)

# Create A matrix
#U_k = np.concatenate((cv_stim_input[10,:], np.zeros(1500)))  # Make sure input is a tuple
U_k = obs_stim_input_all
U_k_1 = np.concatenate((np.zeros(1),U_k[:-1]))
U_k_2 = np.concatenate((np.zeros(2),U_k[:-2]))
y_k = obs_force_all
y_k_1 = np.concatenate((np.zeros(1),y_k[:-1]))
y_k_2 = np.concatenate((np.zeros(2),y_k[:-2]))

#A_mat = np.column_stack((U_k_1,U_k_2,-y_k_1,-y_k_2))
A_mat = np.column_stack((U_k_1,-y_k_1))

# Specify parameter bounds
#lb = np.array([0, -np.inf,-2,-1])
#ub = np.array([np.inf,0,2,1])

lb = np.array([0,-np.inf,-np.inf])
ub = np.array([np.inf,])

lsq_res = sp_optimize.lsq_linear(A_mat,y_k, bounds=(lb,ub), method='trf', verbose=1)  
#lsq_res = sp_optimize.lsq_linear(A_mat,y_k-U_k, bounds=(lb,ub), method='trf', verbose=1)  

#plt.figure
#plt.plot(resampled_time, y_k)
#plt.hold("ON")
#plt.plot(resampled_time,y_k,'-k')
#plt.plot(resampled_time,y_k_1,'-r')
#plt.plot(resampled_time,y_k_2,'-b')

lsq_res

# <codecell>

# Perform discrete transfer function transient analysis
#z1 = 2044.54 #c
#p1 = 350.96  #b
#p2 = 0    #a
#Ts = 1/float(Fs_stim)
#signal_interval = 1
#H_s = cntrl.tf([1,z1],[1,(p1+p2),p1*p2])

# Compute discrete equivalent
#-------------------------------Zeros
#b1 = ( np.exp(-p2*Ts)*(p2*p1 - z1*p1) + np.exp(-p1*Ts)*(z1*p2 - p1*p2) + z1*(p1 - p2))/float(p1*p2*(p1-p2))
#b2 = ( np.exp(-1*(p1+p2)*Ts)*z1*(p1-p2) + np.exp(-p2*Ts)*p2*(z1-p1) - np.exp(-p1*Ts)*p1*(z1-p2))/float(p1*p2*(p1-p2))
#-------------------------------Poles
#a1 = -(np.exp(-p1*Ts) + np.exp(-p2*Ts))
#a2 = (np.exp(-(p1+p2)*Ts))
#H_zoh = cntrl.tf([b1,b2],[1,a1,a2],Ts)

#(b1,b2,a1,a2) = lsq_res.x
#H_zoh = cntrl.tf([b1,b2],[1,a1,a2],1/float(Fs_stim))

#(b1,a1) = lsq_res.x
#H_zoh = cntrl.tf([b1],[1,a1],1/float(Fs_stim))

#H_s_damp1 = cntrl.tf([1],[1,10])
#H_z_damp = cntrl.matlab.c2d(H_s_damp,1/float(4000))
#H_z_damp1 = cntrl.matlab.c2d(H_s_damp1,1/float(4000))
#H_z_damp1 = 1

#(delay_num,delay_den) = cntrl.pade(70E-3,2) # @nd order approximation
#H_s_delay = cntrl.tf(delay_num,delay_den)

#w_n = 1/float(20E-3)
#H_s_damp2 = cntrl.tf([1],[1,2*w_n,w_n**2 + (2*np.pi*0)**2])
#H_z_damp2 = cntrl.matlab.c2d(H_s_damp2,1/float(4000))

signal_interval = 1
# Apply stimulation input to model
cv_sim_stim_input = np.column_stack((cv_stim_input, np.zeros((cv_stim_input.shape[0],1500)))) # 2500 + 1550 = 4000 samples ~ 1 sec
#cv_sim_stim_input = np.column_stack((cv_stim_input,cv_stim_input)) # 2500 + 1550 = 4000 samples ~ 1 sec
cv_sim_t_stim = np.arange(0, cv_sim_stim_input.shape[1]/float(Fs_stim), 1/float(Fs_stim))
cv_sim_force_response = np.zeros_like(cv_sim_stim_input)
cv_sim_force_response_downsampled = np.zeros((len(cv_force_response),signal_interval*Fs_force))
gof_force_response = np.zeros((len(cv_force_response),1))

(c1,c2,c3) = popt_start
for j,u_t in enumerate(cv_sim_stim_input):
    f_u_t = f_IRC_3(u_t,c1, c2, c3)
    #sim_time,y_val,x0 = cntrl.forced_response(H_zoh_delay*H_z_damp2, cv_sim_t_stim, U=u_t, X0 = 0.0)
    sim_time,y_val,x0 = cntrl.forced_response(H_s_delay_22*H_s_damp3*H_s_damp2, cv_sim_t_stim, U=f_u_t, X0 = 0.0)
    cv_sim_force_response[j,:] = y_val.T
    # Resample cv_sim_force_response
    (cv_sim_force_response_downsampled[j,:],downsampled_time) = resample(cv_sim_force_response[j,:],int(signal_interval*Fs_force),sim_time)      # Output is a tuple
    #SS_TOT = cv_force_response[j,:] - cv_force_response[j,:].mean()

# Normalize estimated/predicted force response    
cv_sim_force_response_peak, cv_sim_force_response_norm, cv_sim_force_response_baseline_correct = peak_baseline_normalize_force(cv_sim_force_response_downsampled)

#cv_sim_force_response_norm = 4*4.5*1E7*cv_sim_force_response_downsampled
#cv_force_response_norm = cv_force_response

# Caluclate goodness of fit
for i,val in enumerate(gof_force_response):
    den_term =  np.linalg.norm((cv_force_response_norm[i,0:int(signal_interval*Fs_force)] - np.mean(cv_force_response_norm[i,0:int(signal_interval*Fs_force)])), ord=2)
    num_term =  np.linalg.norm((cv_force_response_norm[i,0:int(signal_interval*Fs_force)] - cv_sim_force_response_norm[i,0:100]), ord=2)
    gof_force_response[i] = (1 - (num_term/den_term))*100

plt.figure()
plt.subplot(1,2,1)
plt.plot(cv_t_force, cv_force_response_norm.T,'-b')
#plt.xlim([0,1])
plt.subplot(1,2,1)
#plt.plot(cv_sim_t_stim,100*cv_sim_force_response.T,'-r')
plt.plot(downsampled_time,cv_sim_force_response_norm.T,'-r')
#plt.plot(samp_time,25*y_t,'-k')
plt.xlim([0,1])
plt.subplot(1,2,2)
plt.boxplot(gof_force_response)
plt.title('Goodness of fit, Avg = '+ str(np.mean(gof_force_response)))
plt.ylim([0,100])

# <codecell>

plt.figure()
plt.subplot(1,2,1)
plt.plot(cv_t_force, cv_force_response.T,'-b')
#plt.xlim([0,1])
plt.subplot(1,2,1)
#plt.plot(cv_sim_t_stim,100*cv_sim_force_response.T,'-r')
plt.plot(downsampled_time,slope*cv_sim_force_response_downsampled.T,'-r')
#plt.plot(samp_time,25*y_t,'-k')
plt.xlim([0,1])
plt.subplot(1,2,2)
plt.boxplot(gof_force_response)
plt.title('Goodness of fit, Avg = '+ str(np.mean(gof_force_response)))
plt.ylim([0,100])

# <codecell>

#%matplotlib gtk
val_to_plot = 49
plt.figure()
plt.subplot(1,2,1)
plt.plot(cv_sim_t_stim,cv_sim_stim_input[val_to_plot,:],'-k')
plt.xlabel('Time (s)')
plt.ylabel('Stimulation input (V)')
plt.xlim([-0.1,1])

plt.subplot(1,2,2)
plt.plot(cv_t_force[0:100],cv_force_response_norm[val_to_plot,0:100],'-b')
plt.plot(cv_t_force[0:100],cv_sim_force_response_norm[val_to_plot,0:100],'-r')
plt.xlabel('Time (s)')
plt.ylabel('Normalized force response')
plt.legend({'Observed','Predicted'},loc = 4)

# <codecell>

plt.figure()
plt.plot(gof_force_response,'-ob')
#plt.plot(u_t)
#plt.plot(f_IRC_2(u_t,c1,c2),'-r')
#plt.plot(f_IRC_3(u_t,c1,c2,c3),'-r')

#popt_end

# <codecell>

#slope, intercept, r_value, p_value, std_err = sp.stats.linregress(cv_sim_force_response_peak,cv_force_response_peak)
plt.figure()
plt.plot(cv_sim_force_response_peak,cv_force_response_peak,'or')
plt.plot(cv_sim_force_response_peak, 5*(0.9**np.arange(1,61,1))*slope*cv_sim_force_response_peak + intercept,'ob')

#plt.figure()
#plt.plot(cv_force_response_peak,'-ob')
#plt.figure()
#plt.plot(cv_sim_force_response_peak*(0.95**np.arange(1,61,1))*slope,'-or')

# <codecell>

0.95**np.arange(1,60,1)

# <codecell>

# Perform transient analysis
tp = 220E-3 # peak time in msec
w_n = 1/float(tp) # pole location
#w_n = 4.5
w_n1 = 1/float(300E-3)
w_n2 = 1/float(40E-3)
signal_interval = 1

# For critically damped system
#H_s = cntrl.tf([2*w_n**2],[1, 3*w_n, 3*w_n**2, w_n**3]) # 3rd order system - didn't improve much
#H_s = cntrl.tf([w_n**2],[1, 2*w_n, w_n**2])
#H_s = cntrl.tf([w_n],[1, w_n])
H_s = cntrl.tf([1,12000],[1, (w_n1+w_n2), w_n1*w_n2])
#cntrl.bode_plot(H_s)
cv_sim_stim_input = np.column_stack((cv_stim_input, np.zeros((80,5500)))) # 2500 + 1550 = 4000 samples ~ 1 sec
cv_sim_force_response = np.zeros_like(cv_sim_stim_input)
cv_sim_force_response_downsampled = np.zeros((len(cv_force_response),signal_interval*Fs_force))
gof_force_response = np.zeros((len(cv_force_response),1))

cv_sim_t_stim = np.arange(0, cv_sim_stim_input.shape[1]/float(Fs_stim), 1/float(Fs_stim))
cv_force_response_peak, cv_force_response_norm, cv_force_response_baseline_correct = peak_baseline_normalize_force(cv_force_response)

avg_force_peak = np.mean(np.reshape(cv_force_response_peak,(16,5)),axis=1)
avg_stim_peak = np.mean(np.reshape(cv_stim_input[:,1],(16,5)),axis=1)
avg_IRC_stim = np.mean(np.reshape(avg_stim_peak,(4,4)),axis=0)
avg_IRC_force = np.mean(np.reshape(avg_force_peak,(4,4)),axis=0)

#plt.figure()
#plt.plot(avg_stim_peak,avg_force_peak,'ob')
#plt.plot(avg_stim_peak,'or')


for j,u_t in enumerate(cv_sim_stim_input):
    #f_u = impulse_response_peak[find_nearest(impulse_input[:,1],u_t)]
    f_u = avg_IRC_force[find_nearest(avg_IRC_stim,u_t)]
    f_u = f_u - np.mean(f_u[5:10])
    sim_time,cv_sim_force_response[j,:],x0 = cntrl.forced_response(H_s, cv_sim_t_stim, U=f_u, X0 = 0.0)
    # Resample cv_sim_force_response
    (cv_sim_force_response_downsampled[j,:],downsampled_time) = resample(cv_sim_force_response[j,:],int(signal_interval*Fs_force),sim_time)      # Output is a tuple
    #SS_TOT = cv_force_response[j,:] - cv_force_response[j,:].mean()
    


#cv_sim_force_response_peak, cv_sim_force_response_norm, cv_sim_force_response_baseline_correct = peak_baseline_normalize_force(cv_sim_force_response)

plt.figure()
plt.subplot(1,2,1)
plt.plot(cv_t_force, cv_force_response.T,'-b')
plt.xlim([0,2])
plt.subplot(1,2,2)
#plt.plot(cv_sim_t_stim,100*cv_sim_force_response.T,'-r')
plt.plot(downsampled_time,cv_sim_force_response_downsampled.T,'-r')
#plt.plot(samp_time,25*y_t,'-k')
plt.xlim([0,2])

# <codecell>

# Calculate goodness of fit for each trial
# gof = 1 - sqrt(SS_RES/SS_TO
H_s_damp2

# <codecell>

#impulse_block_mid_filename, stim_input_block_mid_filename
#impulse_block_end_filename, stim_input_block_end_filename
#read_csv_file(impulse_input_filename, impulse_response_filename) 
#stiminput,response = read_csv_file(impulse_block_mid_filename,stim_input_block_mid_filename,data_files_dir)
#plt.plot(testdata.T)

#y, norm_y, zeromean_y = peak_baseline_normalize_force(impulimpulse_responsese_block_start)

#plt.plot(zeromean_y.T,'-b')
#plt.plot(impulse_response.T)
#plt.figure()
#plt.plot(force_response.T,'-b')
#plt.hold('on')
#plt.plot(impulse_response.T,'-r')
#cv_sim_force_response = np.zeros_like(cv_stim_input)
#plt.plot(cv_sim_force_response.T)

#ind_array = find_nearest(np.sort(impulse_input[:,1]),np.array([17.1]))
#ind_array = find_nearest(impulse_input[:,1],np.array([17.1]))
#impulse_input[ind_array,1]
#np.sort(impulse_input[:,1])
#impulse_input[:,1]
#ind_array

#ind = 36
#plt.figure()
#plt.plot(cv_stim_input[ind,:],'-b')
#plt.hold('on')
#plt.plot(impulse_response_peak[find_nearest(impulse_input[:,1],10*cv_stim_input[ind,:])],'-r')

#plt.plot(cv_t_stim,cv_sim_force_response.T)
#impulse_response_peak[find_nearest(impulse_input[:,1],np.array([55]))]
plt.figure()
plt.hold("ON")
plt.plot(cv_t_force, cv_force_response.T,'-b')
plt.plot(cv_t_stim,cv_sim_force_response.T,'-r')
#plt.plot(samp_time,25*y_t,'-k')
plt.xlim([0,2])
plt.hold("OFF")

# <codecell>

with open('../../BMI_FES_Project/Python_files/NJBT_FD_impulse_response_block2_03-15-2016_14_26_17.txt','r') as response_file:
    #has_header = csv.Sniffer().has_header(f.read(30))# Returns either true of false 
    #f.seek(0)
    reader = csv.reader(response_file)
    reader.next() # Skip 1st row, which is header
    reader.next() # Skip 2nd row, which is header
    force_adc_measurements = [itr_row for itr_row in reader] # import measurements into a list
    
    # convert list to array
    # Convert ADC counts to integers
    # Force sensor equation = 20.21 x + 0.02775
    force_response = np.array(force_adc_measurements,float)
    force_response = (force_response*5/float(1023))*20.21 + 0.02775 # Unit: Newtons
    
with open('../../BMI_FES_Project/Python_files/NJBT_FD_impulse_input_block2_03-15-2016_14_26_17.txt','r') as input_file:
    #has_header = csv.Sniffer().has_header(f.read(30))# Returns either true of false 
    #f.seek(0)
    reader = csv.reader(input_file)
    reader.next() # Skip 1st row, which is header
    reader.next() # Skip 2nd row, which is header
    input_adc_measurements = [itr_row for itr_row in reader] # import measurements into a list
    
    # convert list to array
    # Convert ADC counts to integers
    # Stim input equation = 15x (Voltage divider ratio = 1/15)
    stim_input = np.array(input_adc_measurements,float)
    stim_input = (stim_input*5/float(1023))*15 # Unit: Volts
    stim_input_value = stim_input[:,1]
    #plt.plot(stim_input[0:32,1],'-or')
    #plt.hold("ON")
    #plt.plot(stim_input[32:,1],'-ob')
    

Fs_force = 100 # Sampling frequency
Fs_stim = 4000  # Pulse_width = 500 usec, Fs = 4 KHz;
No_of_stim_pulses = 1  #In 0.1 sec cue interval
stim_freq = 8

# Low pass filter force response at 10Hz - Do not filter as it introduces distortion - Instead use SavitzkyGolay filter
#b_lpf, a_lpf = sp.signal.butter(8, 5/float(Fs_force/2), 'low', analog=False)
#force_response_filt = sp.signal.filtfilt(b_lpf,a_lpf,force_response, axis = 1)

# Baseline correction and Normalize force_response to maximum for each trial 
# Taking first sample as baseline voltage which is to be subtracted from entire signal
force_response_baseline_corrected = np.zeros_like(force_response);
force_response_normalized = np.zeros_like(force_response)
moving_avg_filter_order = 3

for trial_no,trial_force in enumerate(force_response):
    force_response_baseline_corrected[trial_no,:] = trial_force - trial_force[0]
    #force_response_baseline_corrected[trial_no,:] = savitzky_golay(force_response_baseline_corrected[trial_no,:], 9, 3)
    #force_response_baseline_corrected[trial_no,moving_avg_filter_order-1:] = moving_average(force_response_baseline_corrected[trial_no,:],moving_avg_filter_order)
    force_response_normalized[trial_no,:] = force_response_baseline_corrected[trial_no,:]/float(abs(force_response_baseline_corrected[trial_no,:]).max())

nan_removed_force_response_normalized = force_response_normalized[~np.isnan(force_response_normalized).any(axis=1)]
peak_force_response = force_response_baseline_corrected.max(axis=1)

plt.figure()
# Plot the Recrutiment curve
plt.subplot(1,2,1)
plt.plot(stim_input_value,peak_force_response,'ob')
plt.xlabel('Stimulus input (V)')
plt.ylabel('Force response (N)')
plt.title('Isometric Recrutiment Curve')

# Plot average impulse response 
plt.subplot(1,2,2)
plt.plot(np.arange(0,1,0.01),np.mean(nan_removed_force_response_normalized,axis=0),'-b')
plt.grid('on')
plt.xlabel('Time (s)')
plt.ylabel('Norm. avg. impluse response (N)')
plt.title('Linear dynamical system')
