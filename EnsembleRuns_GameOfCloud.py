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
import matplotlib.pyplot as plt


import GameOfCloud
from GameOfCloud import N_steps_day, gameofcloud
from Plotting_GameOfCloud import calc_corr

def ensemble_gameofcloud(ensemble_runs,days1,days2,daybranch1,daybranch2):
    ensemble_runs = ensemble_runs #10
    days1 = days1 #28
    days2 = days2 #42
    ndays              = days1 + days2
    ndays_ocean        = days1 + days2
    ndays_oceanBRANCH  = days2
    daybranch1         = daybranch1 #5
    daybranch2          = daybranch2 #28
    nsteps_day = N_steps_day

    corr_DIU_ALL=np.zeros((ensemble_runs,ndays))
    corr_OCEAN_ALL=np.zeros((ensemble_runs,ndays_ocean))
    corr_OCEANBRANCH1_ALL=np.zeros((ensemble_runs,ndays_oceanBRANCH))
    corr_OCEANBRANCH2_ALL=np.zeros((ensemble_runs,ndays_oceanBRANCH))

    var_DIU_ALL=np.zeros((ensemble_runs,ndays*nsteps_day))
    var_OCEAN_ALL=np.zeros((ensemble_runs,ndays_ocean*nsteps_day))
    var_OCEANBRANCH1_ALL=np.zeros((ensemble_runs,ndays_oceanBRANCH*nsteps_day))
    var_OCEANBRANCH2_ALL=np.zeros((ensemble_runs,ndays_oceanBRANCH*nsteps_day))


    for run in range(0,ensemble_runs):
        
        
        x, Tns, counter,trop_temp_saved = gameofcloud(
            ndays, DIU=True, persistence='yes', branching=False, Tns_branch=False)
        
        corr_DIU_ALL[run] = calc_corr(x, ndays, N_steps_day=nsteps_day)
        var_DIU_ALL[run] = np.var(Tns,axis=(1,2))/np.mean(Tns,axis=(1,2))
        
        x_oceanBRANCH1, Tns_oceanBRANCH1, counter_oceanBRANCH1, trop_temp_saved_oceanBRANCH1 = gameofcloud(
            ndays_oceanBRANCH, DIU=False, persistence='yes', branching=True, Tns_branch=Tns[daybranch1*nsteps_day])

        
        corr_OCEANBRANCH1_ALL[run] = calc_corr(x_oceanBRANCH1, ndays_oceanBRANCH, N_steps_day=nsteps_day)
        var_OCEANBRANCH1_ALL[run] = np.var(Tns_oceanBRANCH1,axis=(1,2))/np.mean(Tns_oceanBRANCH1,axis=(1,2))
       
        
        
        x_oceanBRANCH2, Tns_oceanBRANCH2, counter_oceanBRANCH2, trop_temp_saved_oceanBRANCH2 = gameofcloud(
            ndays_oceanBRANCH, DIU=False, persistence='yes', branching=True, Tns_branch=Tns[daybranch2*nsteps_day])

        
        corr_OCEANBRANCH2_ALL[run] = calc_corr(x_oceanBRANCH2, ndays_oceanBRANCH, N_steps_day=nsteps_day)
        var_OCEANBRANCH2_ALL[run] = np.var(Tns_oceanBRANCH2,axis=(1,2))/np.mean(Tns_oceanBRANCH2,axis=(1,2))
        
        x_ocean, Tns_ocean, counter_ocean, trop_temp_saved_ocean = gameofcloud(
            ndays_ocean, DIU=False, persistence='yes', branching=False, Tns_branch=False)
        
        corr_OCEAN_ALL[run] = calc_corr(x_ocean, ndays_ocean, N_steps_day=nsteps_day)
        var_OCEAN_ALL[run] = np.var(Tns_ocean,axis=(1,2))/np.mean(Tns_ocean,axis=(1,2))
        
    print('finished')
    
    diu = np.nanmean(corr_DIU_ALL, axis=0)
    diu_std = np.nanstd(corr_DIU_ALL, axis=0)
    ocean = np.nanmean(corr_OCEAN_ALL, axis=0)
    ocean_std = np.nanstd(corr_OCEAN_ALL, axis=0)
    branch1 = np.nanmean(corr_OCEANBRANCH1_ALL,axis=0)
    branch1_std = np.nanstd(corr_OCEANBRANCH1_ALL,axis=0)
    branch2 = np.nanmean(corr_OCEANBRANCH2_ALL,axis=0)
    branch2_std = np.nanstd(corr_OCEANBRANCH2_ALL,axis=0)


    diu_var = np.nanmean(var_DIU_ALL, axis=0)
    diu_var_std = np.nanstd(var_DIU_ALL, axis=0)
    ocean_var = np.nanmean(var_OCEAN_ALL, axis=0)
    ocean_var_std = np.nanstd(var_OCEAN_ALL, axis=0)
    branch1_var = np.nanmean(var_OCEANBRANCH1_ALL,axis=0)
    branch1_var_std = np.nanstd(var_OCEANBRANCH1_ALL,axis=0)
    branch2_var = np.nanmean(var_OCEANBRANCH2_ALL,axis=0)
    branch2_var_std = np.nanstd(var_OCEANBRANCH2_ALL,axis=0)
    
    return x, Tns, x_ocean, Tns_ocean, x_oceanBRANCH1, Tns_oceanBRANCH1, x_oceanBRANCH2, Tns_oceanBRANCH2, diu, diu_std, ocean, ocean_std, branch1, branch1_std, branch2, branch2_std, diu_var, diu_var_std, ocean_var, ocean_var_std, branch1_var, branch1_var_std, branch2_var, branch2_var_std


def plot_branches_all_spread(diu, diu_std, ocean, ocean_std, branch1, branch1_std, branch2, branch2_std, name,
    days1,days2,daybranch1, daybranch2, spread='no',log='no'):
    var = diu
    std = diu_std
    var_ocean = ocean
    std_ocean = ocean_std
    daybranch1= daybranch1
    daybranch2= daybranch2
    var_ocA1 = branch1
    std_ocA1 = branch1_std
    var_ocA2 = branch2
    std_ocA2 = branch2_std
    ylabel = str(name)
    
    fig, ax= plt.subplots(figsize=(10,3), dpi=200)
    var_ocA2[0]=var[daybranch2]
    p1, = plt.plot(np.linspace(0,days1+days2,len(var)),var,color='forestgreen', label='DIU',linewidth=1, alpha=1)
    p0, = plt.plot(np.linspace(0,days1+days2,len(var_ocean)),var_ocean, color='navy', linewidth=1, alpha=0.3, label='OCEAN')
    #p2, = plt.plot(np.linspace(daybranch1,daybranch1+days2,len(var_ocA1)-1),var_ocA1[1:],color='dodgerblue', label='OCEAN branch 1',linewidth=0.5)
    p3, = plt.plot(np.linspace(daybranch2,daybranch2+days2,len(var_ocA2)-1),var_ocA2[1:],color='dodgerblue', label='OCEAN branch 2',linewidth=0.5)
    
    l1 = plt.legend([p1,p0], ['DIU','OCEAN'], loc=2)
    l2 = plt.legend([p3], ['DIU2OCEAN branches'], loc=4)
    if spread=='yes':
        p1 = plt.fill_between(np.linspace(0,days1+days2,len(var)), (var-std), (var+std), color='forestgreen', alpha=.1)
        p0 = plt.fill_between(np.linspace(0,days1+days2,len(var_ocean)), (var_ocean-std_ocean), (var_ocean+std_ocean), color='navy', alpha=.1)
     #   p2 = plt.fill_between(np.linspace(daybranch1,daybranch1+days2,len(var_ocA1)-1), (var_ocA1-std_ocA1)[1:], (var_ocA1+std_ocA1)[1:], color='dodgerblue', alpha=.1)
        p3 = plt.fill_between(np.linspace(daybranch2,daybranch2+days2,len(var_ocA2)-1), (var_ocA2-std_ocA2)[1:], (var_ocA2+std_ocA2)[1:], color='dodgerblue', alpha=.1)
        
    plt.title('')
    plt.xlabel('time [day of sim]')
    plt.ylabel(ylabel)
    if log=='yes':
        plt.yticks(np.linspace(-4,0,3))
        plt.ylim((1e-4,1e0))
        plt.yscale('log')
    plt.xticks(np.arange(0,70,14))
    plt.xlim(0,70)

    plt.grid()
    plt.gca().add_artist(l1)
    plt.show()
    return

def plot_snapshots(x,Tns,x_ocean,Tns_ocean,x_oceanBRANCH2,Tns_oceanBRANCH2,
    daybranch2, plotdayOCEAN, plotdayDIU, plotdayBRANCH2, plotdayDIU_2):
    fig = plt.figure(figsize=(20,3), dpi=200)
    vmin=0.5
    vmax=0.55
    plt.rcParams.update({'font.size': 15})

    ax1 = plt.subplot(141)
    ax1.set_title('time = %i'% plotdayOCEAN +'.6 days')
    ax2 = plt.subplot(142)
    ax2.set_title('time = %i'% plotdayDIU +'.6 days')

    ax3 = plt.subplot(143)
    ax3.set_title('time = %i'% plotdayDIU_2 +'.6 days')
    ax4 = plt.subplot(144)
    ax4.set_title('time = %i'% plotdayBRANCH2 +'.6 days')
    im1 = ax1.imshow(x_ocean[plotdayOCEAN*48+30], cmap='Greys')
    im2 = ax2.imshow(x[plotdayDIU*48+30], cmap='Greys')
    im3 = ax3.imshow(x[plotdayDIU_2*48+30], cmap='Greys')
    im4 = ax4.imshow(x_oceanBRANCH2[(plotdayBRANCH2-daybranch2)*48+30],cmap='Greys')
    fig.colorbar(im1, ax=ax1, orientation='vertical', label='OCEAN rain')
    fig.colorbar(im2, ax=ax2, orientation='vertical', label='DIU rain')
    fig.colorbar(im3, ax=ax3, orientation='vertical', label='DIU rain')
    fig.colorbar(im4, ax=ax4, orientation='vertical', label='OCEAN BRANCH rain')
    plt.show()


    cps = Tns[:] - np.mean(Tns,axis=(1,2))[:,None,None]
    cps_ocean = Tns_ocean[:] - np.mean(Tns_ocean,axis=(1,2))[:,None,None]
    cps_oceanBRANCH2 = Tns_oceanBRANCH2[:] - np.mean(Tns_oceanBRANCH2,axis=(1,2))[:,None,None]

    fig = plt.figure(figsize=(18,3), dpi=200)
    vmin=0.5
    vmax=0.55

    ax1 = plt.subplot(141)
    ax1.set_title('time = %i'% plotdayOCEAN +'.6 days')
    ax2 = plt.subplot(142)
    ax2.set_title('time = %i'% plotdayDIU +'.6 days')
    ax3 = plt.subplot(143)
    ax3.set_title('time = %i'% plotdayDIU_2 +'.6 days')
    ax4 = plt.subplot(144)
    ax4.set_title('time = %i'% plotdayBRANCH2 +'.6 days')


    im1 = ax1.imshow(20*cps_ocean[plotdayOCEAN*48+30], cmap='BrBG', vmin=-2, vmax=2)
    im2 = ax2.imshow(20*cps[plotdayDIU*48+30], cmap='BrBG',vmin=-5, vmax=5)
    im3 = ax3.imshow(20*cps[plotdayDIU_2*48+30],cmap='BrBG',vmin=-10, vmax=10)
    im4 = ax4.imshow(20*cps_oceanBRANCH2[(plotdayBRANCH2-daybranch2)*48+30],cmap='BrBG',vmin=-10, vmax=10)
    #im4 = ax4.imshow(20*cps[41*48+30], cmap='BrBG' ,vmin=-15, vmax=15)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    fig.colorbar(im1, ax=ax1, orientation='vertical', label='Tns anomaly',ticks=[-2, 0, 2])
    fig.colorbar(im2, ax=ax2, orientation='vertical', label='Tns anomaly',ticks=[-5, 0, 5])
    fig.colorbar(im3, ax=ax3, orientation='vertical', label='Tns anomaly',ticks=[-10, 0, 10])
    fig.colorbar(im4, ax=ax4, orientation='vertical', label='Tns anomaly',ticks=[-15, 0, 15])
    plt.show()
    return
