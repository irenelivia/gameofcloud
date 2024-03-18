#!/usr/bin/env python
# coding: utf-8

# Import universal packages:
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import time
from matplotlib import animation
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import time
import matplotlib.animation as animation
from scipy.optimize import curve_fit

import GameOfCloud
from GameOfCloud import Ts, parameters


def plot_gameofcloud(DIU, nsteps_day, ndays, Tns, x, trop_temp_saved):

    (tau, D_h, cp_cooling, avg_Ts, amp_Ts, lifetime_rain, interaction, persistence_boost, pixels, 
    event_heating, trop_cooling, delta_t, delta_x, alpha, kernel_diff, kernel, N_steps_day) = parameters()

    cps = Tns[:]*20 - np.mean(Tns*20,axis=(1,2))[:,None,None]

    fig = plt.figure(figsize=(20,16), dpi=200)
    ax1 = plt.subplot(411)

    ax1.set_title('domain rain')
    ax1.plot(np.arange(0,np.size(np.mean(x, axis=(1,2)))), np.mean(x, axis=(1,2)))
    xaxis=np.arange(0,len(x)+1,nsteps_day)
    ax1.set_xticks(np.arange(0,(ndays+1)*nsteps_day,7*nsteps_day))
    ax1.set_xticklabels(np.arange(0,ndays+1,7))
                   
    ax2 = plt.subplot(412)
    ax2.set_title('domain average T')
    ax2.plot(np.arange(0,np.size(np.mean(Tns, axis=(1,2)))), np.mean(Tns, axis=(1,2)), label='T_ns')
    if DIU==True:
        ax2.plot(np.arange(0,np.size(np.mean(Tns, axis=(1,2)))), Ts(np.arange(0,np.size(np.mean(Tns, axis=(1,2)))),nsteps_day,DIU), label='T_s')
    if DIU==False:
        ax2.plot(np.arange(0,np.size(np.mean(Tns, axis=(1,2)))), Ts(np.arange(0,np.size(np.mean(Tns, axis=(1,2)))),nsteps_day,DIU)*np.ones(len(Tns)), label='T_s')
    ax2.plot(trop_temp_saved, label='T_trop')
    ax2.legend()
    ax2.set_xticks(np.arange(0,(ndays+1)*nsteps_day,7*nsteps_day))
    ax2.set_xticklabels(np.arange(0,ndays+1,7))



    ax3 = plt.subplot(413)
    ax3.set_title('day2day corr rain')
    ax3.plot(np.arange(0,ndays),calc_corr(x, ndays, N_steps_day = nsteps_day))
    ax3.set_ylim(-1,1)
    ax3.set_xlim(0,ndays)
    ax3.set_xticks(np.arange(0,ndays+1,7))
    ax3.set_xticklabels(np.arange(0,ndays+1,7))

    ax4 = plt.subplot(414)
    ax4.set_title('variance Tns')
    ax4.plot(np.var(Tns,axis=(1,2)))
    ax4.set_xticks(np.arange(0,(ndays+1)*nsteps_day,7*nsteps_day))
    ax4.set_xticklabels(np.arange(0,ndays+1,7))
    ax4.set_xlabel('day')


    plt.show()


    fig = plt.figure(figsize=(20,5))

    d4 = 1
    d3 = int(ndays/3)
    d2 = int(ndays/2)
    d1 = ndays-1

    plotstep = 32
    day_minus4=(x[nsteps_day*d4 + plotstep])
    day_minus3=(x[nsteps_day*d3 + plotstep])
    day_minus2=(x[nsteps_day*d2 + plotstep])
    day_last  =(x[nsteps_day*d1 + plotstep])
    ax1 = plt.subplot(141)
    ax1.imshow(day_minus4, cmap='Greys',vmin=0, vmax=0.1)
    ax1.set_title('rain day ' + str(d4) + ', time:' + str(24*plotstep/48))
    ax2 = plt.subplot(142)
    ax2.imshow(day_minus3, cmap='Greys',vmin=0, vmax=0.1)
    ax2.set_title('rain day ' + str(d3)+ ', time:' + str(24*plotstep/48))
    ax3 = plt.subplot(143)
    ax3.imshow(day_minus2, cmap='Greys',vmin=0, vmax=0.1)
    ax3.set_title('rain day ' + str(d2)+ ', time:' + str(24*plotstep/48))
    ax4 = plt.subplot(144)
    ax4.imshow(day_last, cmap='Greys',vmin=0, vmax=0.1)
    ax4.set_title('rain day ' + str(d1)+ ', time:' + str(24*plotstep/48))

    plotstep = 36
    fig = plt.figure(figsize=(20,5))
    vmin=-5
    vmax=5
    day_minus4=(cps[nsteps_day*d4 + plotstep,:,:])
    day_minus3=(cps[nsteps_day*d3 + plotstep,:,:])
    day_minus2=(cps[nsteps_day*d1 -1 + plotstep,:,:])
    day_last  =(cps[nsteps_day*d1 + plotstep,:,:])
    ax1 = plt.subplot(141)
    im1 = ax1.imshow(day_minus4, cmap='coolwarm',vmin=vmin, vmax=vmax)
    ax1.set_title('Tns anomaly day ' + str(d4) + ', time:' + str(24*plotstep/48))
    ax2 = plt.subplot(142)
    im2 = ax2.imshow(day_minus3, cmap='coolwarm',vmin=vmin, vmax=vmax)
    ax2.set_title('Tns anomaly day ' + str(d3) + ', time:' + str(24*plotstep/48))
    ax3 = plt.subplot(143)
    im3 = ax3.imshow(day_minus2, cmap='coolwarm',vmin=vmin, vmax=vmax)
    ax3.set_title('Tns anomaly day ' + str(d2) + ', time:' + str(24*plotstep/48))
    ax4 = plt.subplot(144)
    im4 = ax4.imshow(day_last, cmap='coolwarm',vmin=vmin, vmax=vmax)
    ax4.set_title('Tns anomaly day ' + str(d1)+ ', time:' + str(24*plotstep/48))

    fig.colorbar(im1, ax=ax1, orientation='horizontal')
    fig.colorbar(im2, ax=ax2, orientation='horizontal')
    fig.colorbar(im3, ax=ax3, orientation='horizontal')
    fig.colorbar(im4, ax=ax4, orientation='horizontal')

    plt.show()
    return


def plot_diurnalcycle(x, Tns, trop_temp_saved, x_ocean, Tns_ocean, trop_temp_saved_ocean):
    fig, axs = plt.subplots(2,2, figsize=(13,8), dpi=100, sharex=True)

    Tns_timeseries=295+20*np.mean(Tns[7*48:], axis=(1,2))

    axs[0, 0].plot(295+20*Ts(np.arange(0,48),48,True), label='T$_s$', color='forestgreen')
    axs[0, 0].plot(np.mean(Tns_timeseries.reshape(-1, 48), axis=0), label='T$_{ns}$', color='lightgreen')
    axs[0, 0].plot(np.mean(295+20*trop_temp_saved[7*48:14*48].reshape(-1,48), axis=0), label='T$_{trop}$',color='grey')
    axs[0, 0].set_ylabel('Temperature [K]')


    axs[0, 0].set_yticks(np.linspace(295, 315, 3))
    axs[0, 0].legend(loc=2)


    rain_timeseries=np.mean(x, axis=(1,2))
    axs[1, 0].plot(np.mean(rain_timeseries[7*48:].reshape(-1, 48), axis=0), color='forestgreen')

    axs[1, 0].set_ylabel('Fraction of Pixels with Convection')
    axs[1, 0].set_xticks(np.linspace(0,48,4))
    axs[1, 0].set_xticklabels(['midnight', 'noon', '6pm', 'midnight'])
    axs[1, 0].set_ylim(-0.01,0.19)

    axs[1, 0].set_yticks(np.linspace(0, 0.16, 3))

    Tns_timeseries_ocean=295+20*np.mean(Tns_ocean[7*48:], axis=(1,2))
    axs[0, 1].plot(295+20*Ts(np.arange(0,48),48,False)*np.ones(48), label='T$_s$', color='dodgerblue')
    axs[0, 1].plot(np.mean(Tns_timeseries_ocean.reshape(-1, 48), axis=0), label='T$_{ns}$', color='lightblue')
    axs[0, 1].plot(np.mean(295+20*trop_temp_saved_ocean[7*48:14*48].reshape(-1,48), axis=0), label='T$_{trop}$',color='grey')
    #axs[0, 1].set_ylabel('Temperature [K]')
    axs[0, 1].set_ylim(303.9,308.1)
    axs[0, 1].set_yticks(np.linspace(304, 308, 3))
    #axs[0, 1].set_yticks(np.linspace(295, 315, 3))
    axs[0, 1].legend(loc=2)


    rain_timeseries=np.mean(x_ocean, axis=(1,2))
    axs[1, 1].plot(np.mean(rain_timeseries[7*48:].reshape(-1, 48), axis=0), color='dodgerblue')

    #axs[1, 1].set_ylabel('Fraction of Convective Pixels')
    axs[1, 1].set_xticks(np.linspace(0,48,4))
    axs[1, 1].set_xticklabels(['midnight', 'noon', '6pm', 'midnight'])
    axs[1, 1].set_ylim(-0.01,0.19)
    axs[1, 1].set_yticks(np.linspace(0, 0.16, 3))
    plt.show()
    return


def plot_one_pixel(Tns,trop_temp_saved,nsteps_day):
    plt.figure(figsize=(20,5))
    ax2 = plt.subplot(111)
    ax2.set_title('domain average T')
    ax2.plot(np.arange(0,np.size(np.mean(Tns, axis=(1,2)))), 295+20*(Tns[:,0,0]), label='T_ns pixel 0')
    ax2.plot(np.arange(0,np.size(np.mean(Tns, axis=(1,2)))), 295+20*Ts(np.arange(0,np.size(np.mean(Tns, axis=(1,2)))),nsteps_day,True), label='T_s', linewidth=0.2)
    ax2.plot(295+20*trop_temp_saved, label='T_trop')
    ax2.legend()
    ax2.set_xticks(np.arange(0,(ndays+1)*nsteps_day,7*nsteps_day))
    ax2.set_xticklabels(np.arange(0,ndays+1,7))
    return

def calc_corr(x, ndays, N_steps_day):
    (tau, D_h, cp_cooling, avg_Ts, amp_Ts, lifetime_rain, interaction, persistence_boost, pixels, 
    event_heating, trop_cooling, delta_t, delta_x, alpha, kernel_diff, kernel, N_steps_day) = parameters()
    days = ndays
    steps = N_steps_day
    x_new = x.reshape((days, steps, pixels, pixels))
    x_mean = x_new.mean(1)
    corr = np.zeros(days)
    for it in range(1, days):
        x_mean_f0 = gaussian_filter(x_mean[it], sigma=1)
        x_mean_f1 = gaussian_filter(x_mean[it - 1], sigma=1)
        corr[it] = np.corrcoef(x_mean_f0.flatten(), x_mean_f1.flatten())[0, 1]
    return corr


