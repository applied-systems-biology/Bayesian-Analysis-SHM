# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:09:46 2017

@author: Carl-Magnus Svensson
@affiliation: Research Group Applied Systems Biology, Leibniz Institute for 
Natural Product Research and Infection Biology – Hans Knöll Institute (HKI),
Adolf-Reichwein-Straße 23, 07745 Jena, Germany.
@email: carl-magnus.svensson@leibniz-hki.de or cmgsvensson@gmail.com

This is a script calulating and testing differences in realative mutation 
frequancy between Control and brca mutatants. Full details of the 
study can be found in Hirth et al., "Regulation of the germinal center reaction
and somatic hypermutation dynamics by homologus recombination." If any part 
of the code is used for academic purposes or publications, please cite the 
above mentioned paper.

Copyright (c) 2017, 
Dr. Carl-Magnus Svensson

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology -
Hans Knöll Insitute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

Licence: BSD-3-Clause, see ./LICENSE or 
https://opensource.org/licenses/BSD-3-Clause for full details

"""
import numpy as np
import pylab as plt
from scipy.special import binom
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression as LR

rc={'image.cmap': 'YlGnBu', 'font.size': 20, 'axes.labelsize': 40, 'legend.fontsize': 30.0, 
    'axes.titlesize': 50, 'xtick.labelsize': 40, 'ytick.labelsize': 40, 'lines.linewidth': 7.0,
    'font.family' : 'sans-serif', 'font.sans-serif' : 'Arial'}
sns.set_context(rc=rc)

cgat = ['G','C','A','T']
cgatx = ['GC','GA','GT', 'CG','CA','CT', 'AG','AC','AT', 'TG','TC','TA' ]

def binomial_distribution(k, n, pi):
    ''' Returns the PDF value for a binomial distribution
    
        Parameters
        ----------------------------------------------------------------------
        k: int, float
            The observed number of events.
        n: int, float
            The maximum number of events possible.
        pi: float
            The probability that each of the *n* possibilites takes on a 
            positive value.
            
        
        Returns
        -----------------------------------------------------------------------
        PDF_value: float
            The value of the PDF of the Binomial distribution given k, n and pi.
    '''
    return binom(n,k)*(pi**k)*(1-pi)**(n-k)

def P_pa_larger_than_pb(pa,pb):
    ''' Compare two PDFs to see if distribution *pa* is larger than *pb*
        
        By *pa* is larger than *pb* means that the probabilty that a value 
        drawn from the PDF *pa* is larger than a  value drawn from *pb*. The 
        calculation is done by numerical ingration of the convolution pa*pb.
        
        Parameters
        ----------------------------------------------------------------------
        pa: array_like
            Numerical representation of the PDF of the first distribution.   
        pb: array_like
            Numerical representation of the PDF of the second distribution.
        
        Returns
        -----------------------------------------------------------------------
        J: float
            The probability that that a value drawn from the PDF *pa* is larger 
            than a  value drawn from *pb*.
    '''
    nb = pb.shape[0]
    J = 0.0

    for ii in range(nb):
        J += pb[ii]*(np.sum(pa[ii:]))
            
    return J/np.sum(np.outer(pa,pb))
 
def p_phi_G_k(df, phi, group = 'Control', mut = 'N mutations', K = 1):
    ''' Returns the posterior PDF of the relative mutation frequency.
    
        Parameters
        -----------------------------------------------------------------------
        df: pandas dataframe
            Dataframe containing the number of mutations of all types for all
            sequences.
        phi: float, array_like
            The values for which the posterior PDF should be calculated in the
            range [0,1].
        group: string, optional
            The group of animals that are to be considered. *group* have to be
            a value that exist in the column Group in *df*.
        mut: string, optional
            The type of muatation for which the posterior PDF of the relative 
            mutation frequency is calculated for. *mut* have to be one of the 
            columns in *df*.
            
        Returns
        -----------------------------------------------------------------------
        p_pi: array_like
           The posterior PDF of the relative mutation frequency 
    
    '''
    p_pi = np.ones(np.asarray(phi).shape[0])
    k = df[mut].loc[df['Group'] == group].values
    if len(np.asarray(k).shape) > 1:
        k = np.sum(k,1)
        
    for k0 in k:
        ii = 0
        for pp in range(1001):
            p_pi[pp] *= binomial_distribution(k0,K,phi[pp])
        ii += 1
        p_pi = p_pi/np.sum(p_pi*phi[1])
    
    return p_pi

def MM_linear_fit(p_phi_G_k, phi, N_iter = 1000, verbose = True):
    ''' Monte Carlo simulation of values of *phi* and linear fitting. 
    
        We draw one *phi* for each mutation number based on *p_phi_G_k* and fit 
        a linear function to this. The procedure is repeated *N_iter* times.
        
        Parameters
        -----------------------------------------------------------------------
        p_phi_G_k: array_like. 2D
            The posterior PDF where the first dimension is the number of mutations
            and the second is the values of *phi*.
        phi: float, array_like
            The values of posterior PDF that can be drawn during the Monte Carlo
            simulations.
        N_iter: int, optional
            Number of iterations of the Monte Carlo simulation
        verbose: boolean
            Decides whether some key aspects of the fits should be printed
            to the system.
            
        Returns
        -----------------------------------------------------------------------
        slope_list: array_like
            All the slopes from the *N_iter* fits.
        inter_list: array_like
            All the intercepts from the *N_iter* fits.
       phi_all: array_like
            All the drawn *phi* from the *N_iter* fits. 
    
    '''
    slope_list = np.zeros(N_iter)
    inter_list = np.zeros(N_iter)
    R_list = np.zeros(N_iter)
    interc = 0
    phi_all = np.zeros([N_iter, p_phi_G_k.shape[0]])
    for ii in range(N_iter):
        phi_list = np.zeros(p_phi_G_k.shape[0])
        for k in range(p_phi_G_k.shape[0]): 
            accepted = False
            while not accepted:
                idx = np.random.randint(0, 1000)
                if np.random.uniform(0, np.max(p_phi_G_k[k])) < p_phi_G_k[k,idx]:
                    accepted = True
                    phi_list[k] = phi[idx]
                    phi_all[ii,k] = phi[idx]
            

        mod = LR()
        mod.fit(np.atleast_2d(range(1,6)).T, phi_list)

        slope_list[ii] = mod.coef_[0]
        inter_list[ii] = mod.intercept_
        interc += mod.intercept_
        R_list[ii] = mod.score(np.atleast_2d(range(1,6)).T, phi_list)

    if verbose:
        print('Slope: %.5f  (%.5f, %.5f)'%(np.median(slope_list), np.percentile(slope_list, 2.5), np.percentile(slope_list, 97.5)))
        print( 'Intercept: %.5f  (%.5f, %.5f)'%(np.median(inter_list), np.percentile(inter_list, 2.5), np.percentile(inter_list, 97.5)))
        print( 'R^2: %.5f  (%.5f, %.5f)'%(np.median(R_list), np.percentile(R_list, 2.5), np.percentile(R_list, 97.5)))
    
    return slope_list, inter_list, phi_all
 
# Read the dataset with the mutations and their types for all sequences
df = pd.read_csv('./mutation_log_vs2.csv', sep = ';')
# We consider the range between 1 and 5 mutations per sequence
k_array = np.array(range(1,6))

# We numerically evaluate 1001 values of the relative mutation frequency between
# 0 and 0.5.
phi = np.linspace(0, 0.5, 1001)

cmap = sns.cubehelix_palette(light=1, as_cmap=True)
subp = 2.0
fig1, axes1 = plt.subplots(nrows=4, ncols=4)
# Loop over all possible mutations
for m in cgatx:
    print(m)
    p_pi_c_N = np.zeros([5,1001])
    p_pi_b_N = np.zeros([5,1001])
    # Calculate the posterior PDF for Control and brca
    for ii in range(5):
        p_pi_c = p_phi_G_k(df.loc[df['N mutations'] == ii+1], phi, mut = m, K = ii+1)
        p_pi_b = p_phi_G_k(df.loc[df['N mutations'] == ii+1], phi, group = 'brca', mut = m, K = ii+1)
        p_pi_c_N[ii] = p_pi_c[:]
        p_pi_b_N[ii] = p_pi_b[:]
    # Perform trend analysis by Monte Carlo simulation and linear fits.
    print('control:')
    k_c,  i_c, phi_c = MM_linear_fit(p_pi_c_N, phi)
    print('brca:')
    k_b,  i_b, phi_b = MM_linear_fit(p_pi_b_N, phi)
    print('-'*40)

    # Check if the relative muatation frequency for Control is higher than brca
    # for different number of mutations per sequence. The probability that the 
    # elative muatation frequency for brca is higher than Control is 1-J.
    for ii in range(5):
        J = P_pa_larger_than_pb(p_pi_c_N[ii], p_pi_b_N[ii])   
        print(r'$P(\phi_{control}>\phi_{control}|K=%s,M=%s)$=%.4f$'%(ii,m,J))

    # Plot posterior PDFs for Control
    plt.subplot(4,4,subp)
    im1 = plt.pcolor(range(1,7), phi, p_pi_c_N.T, cmap = sns.light_palette("blue", n_colors = 24, as_cmap=True, input='xkcd'), vmax = 15)
    plt.plot(k_array+0.5, np.median(phi_c,axis=0), 'o', color = [0.4, 0.4, 0.4])   
    plt.xlabel('k')
    plt.ylabel(r'$\phi$')
    plt.title(r'$%s>%s$'%(m[0],m[1]))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)

    # Print the differences in slope and intercept between Control and brca 
    # with 95% CI for the differnces.
    print('Slope difference: %.5f  (%.5f, %.5f)'%(np.median(k_c-k_b), np.percentile(k_c-k_b, 2.5), np.percentile(k_c-k_b, 97.5)))
    print('Intercept difference: %.5f  (%.5f, %.5f)'%(np.median(i_c-i_b), np.percentile(i_c-i_b, 2.5), np.percentile(i_c-i_b, 97.5)))
        
    print('-'*40)
    subp += 1.0
    if np.mod(subp-1,5) == 0:
        subp += 1

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)

fig1.subplots_adjust(right=0.8)
cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig1.colorbar(im1, cax=cbar_ax)
cbar.ax.set_ylabel(r'$p(\phi|k,S=Ctrl,M)$')

# Perform the posterior PDF coalculation for each mutation for brca. 
# This is done again is for visualization reasons. See loop above for details.
fig2, axes2 = plt.subplots(nrows=4, ncols=4)
subp = 2.0
for m in cgatx:
    p_pi_b_N = np.zeros([5,1001])
    for ii in range(5):
        p_pi_b = p_phi_G_k(df.loc[df['N mutations'] == ii+1], phi, group = 'brca', mut = m, K = ii+1)
        p_pi_b_N[ii] = p_pi_b[:]

    k_b, i_b, phi_b = MM_linear_fit(p_pi_b_N, phi, verbose = False)

    plt.subplot(4,4,subp)
    im2 = plt.pcolor(range(1,7), phi, p_pi_b_N.T, cmap = sns.light_palette("green", n_colors = 24, as_cmap=True, input='xkcd'), vmax = 15)

    plt.plot(k_array+0.5, np.median(phi_b,axis=0), 'ko')      
    plt.xlabel('k')
    plt.ylabel(r'$\phi$')
    plt.title(r'$%s>%s$'%(m[0],m[1]))

    subp += 1.0
    if np.mod(subp-1,5) == 0:
        subp += 1

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)

fig2.subplots_adjust(right=0.8)
cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig2.colorbar(im2, cax=cbar_ax)
cbar.ax.set_ylabel(r'$p(\phi|k,S=Brca2KO,M)$')
sns.set_style("white")


phi = np.linspace(0, 0.5, 1001)
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
subp = 1.0
fig, axes = plt.subplots(nrows=2, ncols=4)
# Calculate the posterior PDFs of the focus muatations and visualize slightly 
# differently
for m in ['GC', 'CG', 'AG', 'TC']:
    print(m)
    p_pi_c_N = np.zeros([5,1001])
    p_pi_b_N = np.zeros([5,1001])
    for ii in range(5):
        p_pi_c = p_phi_G_k(df.loc[df['N mutations'] == ii+1], phi, mut = m, K = ii+1)
        p_pi_b = p_phi_G_k(df.loc[df['N mutations'] == ii+1], phi, group = 'brca', mut = m, K = ii+1)
        p_pi_c_N[ii] = p_pi_c[:]
        p_pi_b_N[ii] = p_pi_b[:]
    print('control:')
    k_c, i_c, phi_c = MM_linear_fit(p_pi_c_N, phi)
    print('brca:')
    k_b, i_b, phi_b = MM_linear_fit(p_pi_b_N, phi)
    print('-'*40)
    for ii in range(5):
        J = P_pa_larger_than_pb(p_pi_c_N[ii], p_pi_b_N[ii])   
        print(r'$P(\phi_{control}>\phi_{control}|K=%s,M=%s)$=%.4f$'%(ii,m,J))

    plt.subplot(2,4,subp)
    plt.pcolor(range(1,7), phi, p_pi_c_N.T, cmap = sns.light_palette("grey", as_cmap=True, input='xkcd'), vmax = 15)
    plt.plot(k_array+0.5, np.median(k_c)*(k_array)+np.median(i_c), color = [0.4, 0.4, 0.4])   
    #plt.xlabel('k')
    #plt.ylabel(r'$\phi$')
    plt.title(r'$%s>%s$'%(m[0],m[1]))
    plt.xticks([1.5, 2.5, 3.5, 4.5, 5.5], ['1', '2', '3', '4', '5'])

    plt.subplot(2,4,subp + 4)
    im = plt.pcolor(range(1,7), phi, p_pi_b_N.T, cmap = sns.light_palette("grey", as_cmap=True, input='xkcd'), vmax = 15)
    plt.plot(k_array+0.5, np.median(k_b)*(k_array)+np.median(i_b), 'k')    
    #plt.xlabel('k')
    #plt.ylabel(r'$\phi$')
    plt.xticks([1.5, 2.5, 3.5, 4.5, 5.5], ['1', '2', '3', '4', '5'])
    subp += 1.0
    print('Slope difference: %.5f  (%.5f, %.5f)'%(np.median(k_c-k_b), np.percentile(k_c-k_b, 2.5), np.percentile(k_c-k_b, 97.5)))
    print('-'*40  )
    
    if np.mod(subp-1,5) == 0:
        subp += 1

    plt.subplots_adjust(top=0.94, bottom=0.18, left=0.05, right=0.99,
                        hspace=0.2, wspace=0.3)
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.025])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.ax.set_xlabel(r'$p(\phi|k,S,M)$')

plt.show()

