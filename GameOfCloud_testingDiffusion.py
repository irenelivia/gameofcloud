#!/usr/bin/env python
# coding: utf-8

# Import universal packages:
import os
import numpy as np
import xarray as xr
import pandas as pd
import time
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import time



# # The parameters

def parameters():
    # parameters:
    global tau, D_h, cp_cooling, avg_Ts, amp_Ts, lifetime_rain, interaction, persistence_boost
    global pixels, event_heating, trop_cooling
    global delta_t, delta_x, alpha
    global kernel_diff, kernel
    global N_steps_day
    
    
    delta_t = 30*60 # unit: s, minutes in a time step * seconds in a minute
    delta_x = 10*1e3 # units: m, theoretical size of grid box in m(10 km)
    
    
    N_steps_day = 48
    persistence_boost = 0.06 # scale the rich-gets-richer anomaly boost
    tau = 3.6e4 # unit: s (36000s = 10 hours), coefficient that determines how the T_ns is modulated by the T_s
    D_h = 1e2 #1e2 # unit: m**2/s, coefficient that determines how quickly the field is 'diffused'
    cp_cooling = 2/20  # unit: 20 K
    avg_Ts = 0.5 # unit: 20 K
    amp_Ts = 0.5 # unit: 20 K
    lifetime_rain = 2 # unit: time steps that rain lasts and can interact
    interaction = 0.7 #0.7 scales the horizontal interactions
    pixels = 50 # number of pixels on one side of the domain, assuming one is 10x10 km**2 (total domain = pixels x pixels)
    event_heating = (4/20)/pixels**2 # unit: 20 K, tropospheric heating due to 1 single convective event
    trop_cooling = (3/20)/N_steps_day # unit: 20 K, free tropospheric cooling in one time step
    alpha = D_h*delta_t/(delta_x**2)   # unit: unitless, m**2/s  *   s/m**2
    kernel_diff = alpha * np.array([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]]) + (1-4*alpha)* np.array([[0, 0, 0],
                                                                         [0, 1, 0],
                                                                         [0, 0, 0]])
    kernel      = (1/8) * np.array([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]])


# # The model
def update(last_x, last_Tns, last_counter, step,  N_steps_day, DIU, cp_cooling):

    """Makes one step in the cellular automaton.
    Args:
        last_x (np.array): current state of the rain field (1: rain, 0: no rain)
        last_Tns (np.array): current probability matrix (physically near surface temperature T_ns, in units: 20 K)
        last_counter (np.array): keeping track of lifetimes of rain events 
        step : what step of the cellular automaton are we updating
        N_steps_day : how many steps in a day
        DIU : True, if oscillating Ts, False, if constant Ts
    Returns:
        x (np.array): updated state of the rain field
        Tns (np.array): updated state of the near surface temperature
    """
    parameters()
    
    new_counter            = last_counter
    new_counter            = np.where(np.logical_and(last_x==1,last_counter==lifetime_rain),-1,new_counter)
    deactivate_CP_regions  = np.where(new_counter<0,-cp_cooling,0)    
    new_counter            = np.where(np.logical_and(last_x==1,last_counter>=0,last_counter<lifetime_rain),                               new_counter+1,new_counter)
    
    # applying the surface heat flux and deactivating CP regions
    new_Tns = last_Tns + delta_t/tau*(Ts(step, N_steps_day,DIU)-last_Tns) + deactivate_CP_regions
    # applying the horizontal diffusion
    new_Tns = convolve2d(new_Tns, kernel_diff, 'same', boundary = 'wrap')
    eightneighboravg           = convolve2d(last_x, kernel, 'same', boundary = 'wrap')
    boost_Tns_inactive_regions = np.where(last_x==0,eightneighboravg,0)
    # prolonging active regions of rain
    prolong_active_regions     = np.where(np.logical_and(new_counter>0, new_counter<lifetime_rain),1,0)
    
    
    return new_Tns, prolong_active_regions, boost_Tns_inactive_regions, new_counter


def Ts(step, N_steps_day, DIU):
    parameters()
    if (DIU==True):
        Ts = avg_Ts - amp_Ts*np.cos(2*np.pi*np.mod(step,N_steps_day)/N_steps_day)
    else:
        Ts = avg_Ts
    return Ts


def gameofcloud(ndays, DIU, persistence, branching, Tns_branch):
    """ Game of Cloud, initiated from the night
    Args:
        pixels (int): number of cells in the row
        tot_steps (int): total number of steps to evolve the automaton
        lifetime_rain (int): number of steps that an active pixel persists
    Returns:    
        x: states of the automaton
        P: probability field of the automaton
    """
    
    parameters() #calling all parameters
    tot_steps = ndays*N_steps_day
    
    x = np.zeros((tot_steps, pixels,pixels))
    Tns = np.zeros((tot_steps, pixels,pixels))
    Tns[0,:,:] = np.ones((pixels,pixels)) * (avg_Ts - 1) +   .1 * (np.random.random((pixels,pixels)) - .5)
    
    if branching==True:
        Tns[0,:,:] = Tns_branch
           
    counter = np.zeros((tot_steps, pixels,pixels))
    trop_temp_saved=np.zeros(tot_steps)
    trop_temp_saved[0] = avg_Ts
    
    t00 = time.time()
    
    # with global constraint:
    for step in range(0,tot_steps-1):
        time0 = time.time()
        (Tns[step + 1, :], prolong_active_regions, boost_Tns_inactive_regions, counter[step+1,:]) = update(
            x[step, :], Tns[step, :], counter[step,:], step, N_steps_day, DIU, cp_cooling)

        trop_temp = trop_temp_saved[step]
        
        # persistence
        if persistence=='yes':
            # just computing the anomalies
            anomalies        = Tns[step + 1,:] - np.mean(Tns[step + 1,:])
            # applying a "rich-gets-richer" feedback with a "carrying capacity" on both ends
            boosted_Tns      = Tns[step + 1,:] + (1. + anomalies) * (1. - anomalies) * anomalies * persistence_boost
            # updating Tns accordingly
            Tns[step + 1, :] = boosted_Tns
            
        
        Tns[step + 1, :] = Tns[step + 1, :] + boost_Tns_inactive_regions * np.mean(Tns[step + 1, :]) * cp_cooling * interaction
        Tns_unstable = np.where(np.logical_and((Tns[step + 1, :] - trop_temp)>0, prolong_active_regions==0), Tns[step + 1, :], 0)

        
        x_pot  = np.zeros((pixels,pixels))
        x_fire = np.zeros((pixels,pixels))
        x_pot  = x_pot.flatten()

        sumTns = np.zeros(len(x_pot))

        if np.sum(Tns_unstable)>0:
            sumTns = np.cumsum(Tns_unstable/np.sum(Tns_unstable))

        # loop to select among positively buoyant pixels:
        # let a pixel fire, update free_trop_temp, continue until there are no more pixels

            for fire in range(len(sumTns)):
                random_number = np.random.rand()
                selection_index = np.where(sumTns>=random_number)[0][0] 
                if (trop_temp < Tns_unstable.flatten()[selection_index] and x_pot[selection_index] == 0):
                    x_pot[selection_index] = 1
                    trop_temp = trop_temp + event_heating 

        x_fire = x_pot.reshape((pixels,pixels))

        # applying radiative cooling of the free troposphere
        trop_temp = trop_temp - trop_cooling 
        
        # updating saved information on the free troposphere
        trop_temp_saved[step+1]=trop_temp
        x[step+1, :] = prolong_active_regions + x_fire



    return x,Tns,counter,trop_temp_saved

def calc_corr(x, ndays, N_steps_day):
    days=ndays
    steps = N_steps_day
    x_new = x.reshape((days,steps,pixels,pixels))
    x_mean = x_new.mean(1)
    corr=np.zeros(days)
    for it in range(1,days):
        x_mean_f0 = gaussian_filter(x_mean[it],   sigma=1)
        x_mean_f1 = gaussian_filter(x_mean[it-1], sigma=1)
        corr[it]= np.corrcoef(x_mean_f0.flatten(),x_mean_f1.flatten())[0, 1]
    return corr




