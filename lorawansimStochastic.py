#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:56:23 2022

@author: felipe
"""

import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
from scipy import signal
import scipy.io as sio
import pandas as pd
import math
import time
import sys
from sklearn.mixture import GaussianMixture
from scipy.misc import derivative
from scipy.integrate import quad
from scipy.integrate import cumtrapz
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn import preprocessing
from collections import deque

def loadMatlabFiles(matlab_filepath):
    matfile = sio.loadmat(matlab_filepath)

    result_cycle_total_mJ = matfile['result_cycle_energy_total_mJ']
    result_cycle_DL_RX1_idle_mJ = matfile['result_cycle_energy_DL_RX1_idle_mJ']
    result_cycle_DL_RX1_mJ = matfile['result_cycle_energy_DL_RX1_mJ']
    result_cycle_DL_RX2_idle_mJ = matfile['result_cycle_energy_DL_RX2_idle_mJ']
    result_cycle_DL_RX2_mJ = matfile['result_cycle_energy_DL_RX2_mJ']
    result_cycle_idle_mJ = matfile['result_cycle_energy_idle_mJ']
    result_cycle_sleep_mJ = matfile['result_cycle_energy_sleep_mJ']   
    result_cycle_UL_mJ = matfile['result_cycle_energy_UL_mJ']
    result_cycle_total_ms = matfile['result_cycle_time_total_ms']
    result_cycle_DL_RX1_idle_ms = matfile['result_cycle_time_DL_RX1_idle_ms']
    result_cycle_DL_RX1_ms = matfile['result_cycle_time_DL_RX1_ms']
    result_cycle_DL_RX2_idle_ms = matfile['result_cycle_time_DL_RX2_idle_ms']
    result_cycle_DL_RX2_ms = matfile['result_cycle_time_DL_RX2_ms']
    result_cycle_idle_ms = matfile['result_cycle_time_idle_ms']
    result_cycle_sleep_ms = matfile['result_cycle_time_sleep_ms']   
    result_cycle_UL_ms = matfile['result_cycle_time_UL_ms']
    result_cycle_uplink_rate = matfile['results_calc_all_iterations_UL_delivery_rate']
    
    result_cycle_total_mJ_array	=	result_cycle_total_mJ.flatten()
    result_cycle_DL_RX1_idle_mJ_array	=	result_cycle_DL_RX1_idle_mJ.flatten()
    result_cycle_DL_RX1_mJ_array	=	result_cycle_DL_RX1_mJ.flatten()
    result_cycle_DL_RX2_idle_mJ_array	=	result_cycle_DL_RX2_idle_mJ.flatten()
    result_cycle_DL_RX2_mJ_array	=	result_cycle_DL_RX2_mJ.flatten()
    result_cycle_idle_mJ_array	=	result_cycle_idle_mJ.flatten()
    result_cycle_sleep_mJ_array	=	result_cycle_sleep_mJ.flatten()
    result_cycle_UL_mJ_array	=	(result_cycle_UL_mJ.flatten())
    result_cycle_total_ms_array	=	result_cycle_total_ms.flatten()
    result_cycle_DL_RX1_idle_ms_array	=	result_cycle_DL_RX1_idle_ms.flatten()
    result_cycle_DL_RX1_ms_array	=	result_cycle_DL_RX1_ms.flatten()
    result_cycle_DL_RX2_idle_ms_array	=	result_cycle_DL_RX2_idle_ms.flatten()
    result_cycle_DL_RX2_ms_array	=	result_cycle_DL_RX2_ms.flatten()
    result_cycle_idle_ms_array	=	result_cycle_idle_ms.flatten()
    result_cycle_sleep_ms_array	=	result_cycle_sleep_ms.flatten()
    result_cycle_UL_ms_array	=	(result_cycle_UL_ms.flatten())
    result_cycle_uplink_rate_array	=	(result_cycle_uplink_rate.flatten())
    
    return (result_cycle_total_mJ_array, result_cycle_DL_RX1_idle_mJ_array, 
            result_cycle_DL_RX1_mJ_array, result_cycle_DL_RX2_idle_mJ_array, 
            result_cycle_DL_RX2_mJ_array, result_cycle_idle_mJ_array, 
            result_cycle_sleep_mJ_array, result_cycle_UL_mJ_array, 
            result_cycle_total_ms_array,
            result_cycle_DL_RX1_idle_ms_array, result_cycle_DL_RX1_ms_array, 
            result_cycle_DL_RX2_idle_ms_array, result_cycle_DL_RX2_ms_array, 
            result_cycle_idle_ms_array, result_cycle_sleep_ms_array, 
            result_cycle_UL_ms_array, result_cycle_uplink_rate_array)

def loadParameters(data):
    beta_1 = 1/(np.mean(data[1])+np.mean(data[2]))
    beta_2 = 1/(np.mean(data[3])+np.mean(data[4]))
    beta_3 = 1/np.mean(data[5])
    beta_4 = 1/np.mean(data[6])
    beta_5 = 1/np.mean(data[7])
    return (beta_1, beta_2, beta_3, beta_4, beta_5)

def loadParameters0Axis(data, e_c_mean, e_c_min):
    correction_1_b = ((np.mean(data[1])+np.mean(data[2]))/e_c_mean)*e_c_min
    correction_2_b = ((np.mean(data[3])+np.mean(data[4]))/e_c_mean)*e_c_min
    correction_3_b = ((np.mean(data[5]))/e_c_mean)*e_c_min
    correction_4_b = ((np.mean(data[6]))/e_c_mean)*e_c_min
    correction_5_b = ((np.mean(data[7]))/e_c_mean)*e_c_min
    beta_1 = 1/(np.mean(data[1])+np.mean(data[2])-correction_1_b)
    beta_2 = 1/(np.mean(data[3])+np.mean(data[4])-correction_2_b)
    beta_3 = 1/(np.mean(data[5])-correction_3_b)
    beta_4 = 1/(np.mean(data[6])-correction_4_b)
    beta_5 = 1/(np.mean(data[7])-correction_5_b)   
    return (beta_1, beta_2, beta_3, beta_4, beta_5)

def calculatePDFCDF0Axis(beta, M, lmax, step):
    e = np.arange(0,lmax,step)
    den_1 = np.ones(M)
    pdf = np.zeros(int(lmax/step))
    pdf_aux = np.zeros(int(lmax/step))
    
    #calculate pdf
    num_1 = np.prod(beta)   
    for i in range(M):  
        for j in range(M):
            if(i != j):
                den_1[i] = den_1[i]*(beta[j]-beta[i])
        
        pdf_aux = np.exp(-beta[i]*(e))/den_1[i]        
        pdf = pdf + pdf_aux
        
    pdf = pdf*num_1    
    
    #calculate E[E_c]
    e_e = np.trapz(e*pdf,dx=step)
    
    #calculate cdf numerically   
    cdf = cumtrapz(pdf, e, initial=0)

    return e, e_e, pdf, cdf

def calculateShiftedPDFCDF(pdf, e_min, lmax, step):
    e = np.arange(0,lmax,step)
    pdf_aux = deque(pdf)
    pdf_aux.rotate(int(e_min/step))
    pdf_aux_array = np.array(pdf_aux)

    pdf_aux_array[0:int(e_min/step)] = [0]*int(e_min/step)
    
    #calculate cdf numerically   
    cdf = cumtrapz(pdf, e, initial=0)

    return pdf_aux_array, cdf

def calculatePDFTruncated(e, pdf, cdf, e_z, e_w, lmax, step):         
    a = int(e_w/step)
    b = int(e_z/step)
        
    P_a = cdf[a]
    P_b = cdf[b]
    
    x = P_b - P_a
    
    k = int(lmax/step)
    pdf_trunc = np.zeros(k)
        
    pdf_trunc[a:b] = np.divide(pdf[a:b], x)
    
    return e, pdf_trunc

def calculateMaximumConsumption(N, pdf, cdf, lmax, step):
    z = np.arange(0,lmax,step)
    prod = np.ones(int(lmax/step))
    p_Z_z = np.zeros(int(lmax/step))
  
    i=0
    for i in range(N):
        prod = np.ones(int(lmax/step))
        for j in range(N):
            if(i != j):
                prod = prod*cdf
        p_Z_z = p_Z_z + pdf*prod
        
    e_z = np.trapz(z*p_Z_z,dx=step)
    return e_z

def calculateMinimumConsumption(N, pdf, cdf, lmax, step):
    w = np.arange(0,lmax,step)
    prod = np.ones(int(lmax/step))
    p_W_w = np.zeros(int(lmax/step))
    
    i=0
    for i in range(N):
        prod = np.ones(int(lmax/step))
        for j in range(N):
            if(i != j):
                prod = prod*(1-cdf)
        p_W_w = p_W_w + pdf*prod
        
    e_w = np.trapz(w*p_W_w,dx=step)
    return e_w

def plotHistogram(data, n_bins, x_axis_legend, y_axis_legend, data_model_legend, lmin, lmax, file_name, color, alpha):
        p_o, bins, patches = plt.hist(x=data, bins=n_bins, density=True, color=color, alpha=alpha, rwidth=0.85)
        plt.xlabel(x_axis_legend, labelpad=8,fontsize=20)
        plt.ylabel(y_axis_legend, labelpad=8,fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=14,pad=8)
        plt.legend(data_model_legend)
        plt.tight_layout()
        plt.grid()
        axes = plt.gca()
        axes.set_xlim([lmin, lmax])
        plt.savefig(file_name)
        return p_o, bins 

def plotTheoreticalCurve(x, y, x_axis_legend, y_axis_legend, function_model_legend, lmin, lmax, file_name, color):
        plt.plot(x, y, c=color)
        plt.xlabel(x_axis_legend,labelpad=8,fontsize=20)
        plt.ylabel(y_axis_legend,labelpad=8,fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=14,pad=8)
        plt.legend(function_model_legend)
        plt.tight_layout()
        plt.grid(b=True, which='major', color='k', linestyle='--', alpha=0.3)
        axes = plt.gca()
        axes.set_xlim([lmin, lmax])
        plt.savefig(file_name)

matlab_filepath_1 = "LoRaWANSim_GW_1_ED_200_R_4_T_300_CR_1_iter_5_n20.mat"
matlab_filepath_2 = "LoRaWANSim_GW_1_ED_200_R_4_T_30_CR_1_iter_5_n20.mat"
# matlab_filepath_1 = "LoRaWANSim_GW_1_ED_200_R_4_T_300_CR_1_iter_5_n40.mat"
# matlab_filepath_2 = "LoRaWANSim_GW_1_ED_200_R_4_T_30_CR_1_iter_5_n40.mat"
# matlab_filepath_1 = "LoRaWANSim_GW_15_ED_200_R_4_T_30_CR_1_iter_1_n40.mat"
# matlab_filepath_2 = "LoRaWANSim_GW_1_ED_200_R_4_T_30_CR_1_iter_5_n40.mat"
# matlab_filepath_1 = "LoRaWANSim_GW_1_ED_200_R_4_T_300_CR_1_iter_5_n20.mat"
# matlab_filepath_2 = "LoRaWANSim_GW_1_ED_200_R_4_T_300_CR_1_iter_5_n40.mat"


graph_file_name = "scenario01.pdf"
#graph_file_name = "scenario02.pdf"
#graph_file_name = "scenario03.pdf"
#graph_file_name = "scenario04.pdf"

txt_file_name = "results_GW_1_ED_200_R_4_T_X_CR_1_iter_5_n20.txt"
#txt_file_name = "results_GW_1_ED_200_R_4_T_X_CR_1_iter_5_n40.txt"
#txt_file_name = "results_GW_X_ED_200_R_4_T_30_CR_1_iter_5_n40.txt"
#txt_file_name = "results_GW_1_ED_200_R_4_T_300_CR_1_iter_5_nx.txt"

data_model_legend = ['PDF ($T=300$)', 'PDF ($T=30$)', 'Simulation Data ($T=300$)', 'Simulation Data ($T=30$)']
#data_model_legend = ['PDF ($GW=15$)', 'PDF ($GW=1$)', 'Simulation Data ($GW=15$)', 'Simulation Data ($GW=1$)']
#data_model_legend = ['PDF ($n=2.0$)', 'PDF ($n=4.0$)', 'Simulation Data ($n=2.0$)', 'Simulation Data ($n=4.0$)']


data_1 = loadMatlabFiles(matlab_filepath_1)
data_2 = loadMatlabFiles(matlab_filepath_2)

e_c_1 = data_1[0]
e_c_2 = data_2[0]
ul_rate_1 = data_1[16][0]
ul_rate_2 = data_2[16][0]

n_bins_1 = 10
n_bins_2 = 12
x_axis_legend = '$e$ [mJ]'
y_axis_legend = 'Density'
lmin_plot = 90
lmax_plot = 220
lmax = 300
step = 0.1
alpha_1=0.5
alpha_2=0.5
color_1='blue'
color_2='red'

max_e_c_1 = np.max(e_c_1)
min_e_c_1 = np.min(e_c_1)
mean_e_c_1 = np.mean(e_c_1)
max_e_c_2 = np.max(e_c_2)
min_e_c_2 = np.min(e_c_2)
mean_e_c_2 = np.mean(e_c_2)

parameters_1 = loadParameters(data_1)
beta_1 = parameters_1
parameters_2 = loadParameters(data_2)
beta_2 = parameters_2


parameters_0_axis_1 = loadParameters0Axis(data_1, mean_e_c_1, min_e_c_1)
beta_0_axis_1 = parameters_0_axis_1
parameters_0_axis_2 = loadParameters0Axis(data_2, mean_e_c_2, min_e_c_2)
beta_0_axis_2 = parameters_0_axis_2

#e_c_0_axis = [x - min_e_c for x in e_c]

plotHistogram(e_c_1, n_bins_1, x_axis_legend, y_axis_legend, data_model_legend, lmin_plot, lmax_plot, graph_file_name, color_1, alpha_1)
plotHistogram(e_c_2, n_bins_2, x_axis_legend, y_axis_legend, data_model_legend, lmin_plot, lmax_plot, graph_file_name, color_2, alpha_2)


M = 5
e, e_e_1, pdf_0axis_1, cdf_0axis_1 = calculatePDFCDF0Axis(beta_0_axis_1, M, lmax, step)
e, e_e_2, pdf_0axis_2, cdf_0axis_2 = calculatePDFCDF0Axis(beta_0_axis_2, M, lmax, step)

e_e_1 = e_e_1+min_e_c_1
e_e_2 = e_e_2+min_e_c_2

pdf_shifted_1, cdf_shifted_1 = calculateShiftedPDFCDF(pdf_0axis_1, min_e_c_1, lmax, step)
pdf_shifted_2, cdf_shifted_2 = calculateShiftedPDFCDF(pdf_0axis_2, min_e_c_2, lmax, step)


plotTheoreticalCurve(e, pdf_shifted_1, x_axis_legend, y_axis_legend, data_model_legend, lmin_plot, lmax_plot, graph_file_name, color_1)
plotTheoreticalCurve(e, pdf_shifted_2, x_axis_legend, y_axis_legend, data_model_legend, lmin_plot, lmax_plot, graph_file_name, color_2)

N = 200
e_w_1 = calculateMinimumConsumption(N, pdf_0axis_1, cdf_0axis_1, lmax, step)+min_e_c_1
e_z_1 = calculateMaximumConsumption(N, pdf_0axis_1, cdf_0axis_1, lmax, step)+min_e_c_1
e_w_2 = calculateMinimumConsumption(N, pdf_0axis_2, cdf_0axis_2, lmax, step)+min_e_c_1
e_z_2 = calculateMaximumConsumption(N, pdf_0axis_2, cdf_0axis_2, lmax, step)+min_e_c_1

with open(txt_file_name, 'w') as f:
    print("Uplink Delivery Rate: " + str(ul_rate_1) + ", " + str(ul_rate_2),file=f)
    print("Mean Consumption: " + str(mean_e_c_1) + ", " + str(mean_e_c_2),file=f)
    print("Expected Consumption: "+ str(e_e_1) + ", " + str(e_e_2),file=f)
    print("Max Consumption: "+ str(max_e_c_1) + ", " + str(max_e_c_2),file=f)
    print("Min Consumption:"+ str(min_e_c_1) + ", " + str(min_e_c_2),file=f)
    print("Expected Max:"+ str(e_z_1) + ", " + str(e_z_2),file=f)
    print("Expected Min:"+ str(e_w_1) + ", " + str(e_w_2),file=f)
