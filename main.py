import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# Data Loading
data = np.loadtxt('hw2_fitting.dat')
nu   = data[:,0]
phi  = data[:,1]
err  = data[:,2]
# Functions
def lorentzian  (x, nu0, alpha_L): return (1./np.pi)*(alpha_L/((x - nu0)**2. + alpha_L**2.))
def gaussian    (x, nu0, alpha_D): return (1./alpha_D)*np.sqrt(np.log(2.)/np.pi)*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))
def Chi2        (f, dataf, datax, derr, a1, a2): return np.sum(((dataf - f(datax, a1, a2))/derr)**2.)
def levenMarq   (f, dfda1, dfda2, ddfda1da1, ddfda2da2, ddfda1da2, dataf, datax, derr, a1, a2):
    lamb = 0.001
    chisquared = Chi2(f, dataf, datax, derr, a1, a2)
    chi2d1 = -2.*np.sum(((dataf - f(datax, a1, a2))/(derr**2.))*dfda1(datax, a1, a2))
    chi2d2 = -2.*np.sum(((dataf - f(datax, a1, a2))/(derr**2.))*dfda2(datax, a1, a2))
    dchi2d1d1 = 2.*np.sum((1./derr**2.)*(dfda1(datax, a1, a2)**2. - (dataf - f(datax, a1, a2))*ddfda1da1(datax, a1, a2)))
    dchi2d2d2 = 2.*np.sum((1./derr**2.)*(dfda2(datax, a1, a2)**2. - (dataf - f(datax, a1, a2))*ddfda2da2(datax, a1, a2)))
    dchi2d1d2 = 2.*np.sum((1./derr**2.)*(dfda1(datax, a1, a2)*dfda2(datax, a1, a2) - (dataf - f(datax, a1, a2))*ddfda1da2(datax, a1, a2)))
    betaVector = -0.5*np.array([chi2d1, chi2d2])
    lmAlpha = 0.5*np.array([[dchi2d1d1*(1.+lamb), dchi2d1d2], [dchi2d1d2, dchi2d2d2*(1.+lamb)]])
    da1, da2 = np.linalg.solve(lmAlpha, betaVector)
    a1_new, a2_new = a1+da1, a2+da2
    chisquardNew = Chi2(f, dataf, datax, derr, a1_new, a2_new)
    while abs(chisquardNew - chisquared)/chisquared > 10.**(-6.):
        if chisquardNew >= chisquared:
            lamb = lamb*10.
            lmAlpha = 0.5*np.array([[dchi2d1d1*(1.+lamb), dchi2d1d2], [dchi2d1d2, dchi2d2d2*(1.+lamb)]])
            da1, da2 = np.linalg.solve(lmAlpha, betaVector)
            a1_new, a2_new = a1+da1, a2+da2
            chisquardNew = Chi2(f, dataf, datax, derr, a1_new, a2_new)
        elif chisquardNew < chisquared:
            lamb = lamb/10.
            a1, a2 = a1+da1, a2+da2
            chi2d1 = -2.*np.sum(((dataf - f(datax, a1, a2))/(derr**2.))*dfda1(datax, a1, a2))
            chi2d2 = -2.*np.sum(((dataf - f(datax, a1, a2))/(derr**2.))*dfda2(datax, a1, a2))
            dchi2d1d1 = 2.*np.sum((1./derr**2.)*(dfda1(datax, a1, a2)**2. - (dataf - f(datax, a1, a2))*ddfda1da1(datax, a1, a2)))
            dchi2d2d2 = 2.*np.sum((1./derr**2.)*(dfda2(datax, a1, a2)**2. - (dataf - f(datax, a1, a2))*ddfda2da2(datax, a1, a2)))
            dchi2d1d2 = 2.*np.sum((1./derr**2.)*(dfda1(datax, a1, a2)*dfda2(datax, a1, a2) - (dataf - f(datax, a1, a2))*ddfda1da2(datax, a1, a2)))
            betaVector = -0.5*np.array([chi2d1, chi2d2])
            lmAlpha = 0.5*np.array([[dchi2d1d1*(1.+lamb), dchi2d1d2], [dchi2d1d2, dchi2d2d2*(1.+lamb)]])
            da1, da2 = np.linalg.solve(lmAlpha, betaVector)
            a1_new, a2_new = a1+da1, a2+da2
            chisquared = chisquardNew
            chisquardNew = Chi2(f, dataf, datax, derr, a1_new, a2_new)
    lmAlpha = 0.5*np.array([[dchi2d1d1, dchi2d1d2], [dchi2d1d2, dchi2d2d2]])
    Cov = np.linalg.inv(lmAlpha)
    return (np.array([a1, a2]), Cov)
def dld1    (x, nu0, alpha_L): return (2.*(x - nu0)*alpha_L)/(np.pi*((x - nu0)**2. + alpha_L**2.)**2.)
def dld2    (x, nu0, alpha_L): return (1./np.pi)*(((x - nu0)**2. - alpha_L**2.)/(((x - nu0)**2. + alpha_L**2.)**2.))
def dld1d1  (x, nu0, alpha_L): return ((2.*alpha_L)/np.pi)*((3.*(x - nu0)**2. - alpha_L**2.)/(((x - nu0)**2. + alpha_L**2.)**3.))
def dld2d2  (x, nu0, alpha_L): return ((2.*alpha_L)/np.pi)*((alpha_L**2. - 3.*(x - nu0)**2.)/(((x - nu0)**2. + alpha_L**2.)**3.))
def dld1d2  (x, nu0, alpha_L): return ((2.*(x - nu0))/np.pi)*(((x - nu0)**2. - 3.*alpha_L**2.)/(((x - nu0)**2. + alpha_L**2.)**3.))
def dgd1    (x, nu0, alpha_D): return ((2.*(np.log(2.)**(3./2.))*(x - nu0))/((alpha_D**3.)*np.sqrt(np.pi)))*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))
def dgd2    (x, nu0, alpha_D): return (1./(alpha_D**4.))*np.sqrt(np.log(2.)/np.pi)*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))*(2.*np.log(2.)*(x - nu0)**2. - alpha_D**2.)
def dgd1d1  (x, nu0, alpha_D): return ((2.*(np.log(2.)**(3./2.)))/((alpha_D**4.)*np.sqrt(np.pi)))*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))*(2.*np.log(2.)*(x - nu0)**2. - alpha_D**2.)
def dgd2d2  (x, nu0, alpha_D): return (2./(alpha_D**7.))*np.sqrt(np.log(2.)/np.pi)*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))*(alpha_D**4. - 5.*np.log(2.)*((x - nu0)**2.)*(alpha_D**2.) + 2.*(np.log(2.)**2.)*(x - nu0)**4.)
def dgd1d2  (x, nu0, alpha_D): return ((2.*(np.log(2.)**(3./2.))*(x - nu0))/(alpha_D**6.))*np.exp((-np.log(2.)*(x - nu0)**2.)/(alpha_D**2.))*(2.*np.log(2.)*(x - nu0)**2. - 3.*alpha_D**2.)
# Scipy Function
lorentzianFit = opt.curve_fit(lorentzian, nu, phi, sigma = err, method='lm')
gaussianFit = opt.curve_fit(gaussian, nu, phi, sigma = err, method='lm')
lorentzian_nu0, lorentzian_alpha = lorentzianFit[0]
gaussian_nu0, gaussian_alpha = gaussianFit[0]
lorentzian_nu0_error, lorentzian_alpha_error = np.sqrt(np.diag(lorentzianFit[1]))
gaussian_nu0_error, gaussian_alpha_error = np.sqrt(np.diag(gaussianFit[1]))
# Leven Marq Function
nu0Estimate, LalphaEstimate, DalphaEstimate = 40., 10., 10.
lorentzianFit_own = levenMarq(lorentzian, dld1, dld2, dld1d1, dld2d2, dld1d2, phi, nu, err, nu0Estimate, LalphaEstimate)
gaussianFit_own = levenMarq(gaussian, dgd1, dgd2, dgd1d1, dgd2d2, dgd1d2, phi, nu, err, nu0Estimate, DalphaEstimate)
lorentzian_nu0, lorentzian_alpha = lorentzianFit_own[0]
gaussian_nu0, gaussian_alpha = gaussianFit_own[0]
lorentzian_nu0_error, lorentzian_alpha_error = np.sqrt(np.diag(lorentzianFit_own[1]))
gaussian_nu0_error, gaussian_alpha_error = np.sqrt(np.diag(gaussianFit_own[1]))
print('Lorentzian fit using the scipy function: \n nu_0 = %s +/- %s$, \n alpha_L = %s +/- %s' % (lorentzian_nu0, lorentzian_nu0_error, lorentzian_alpha, lorentzian_alpha_error))
print('Gaussian fit using the scipy function: \n nu_0 = %s +/- %s$, \n alpha_D = %s +/- %s' % (gaussian_nu0, gaussian_nu0_error, gaussian_alpha, gaussian_alpha_error))
print('Lorentzian fit using our levenMarq function: \n nu_0 = %s +/- %s, \n alpha_L = %s +/- %s' % (lorentzian_nu0, lorentzian_nu0_error, lorentzian_alpha, lorentzian_alpha_error))
print('Gaussian fit using our levenMarq function: \n nu_0 = %s +/- %s, \n alpha_D = %s +/- %s' % (gaussian_nu0, gaussian_nu0_error, gaussian_alpha, gaussian_alpha_error))
## Plotting using Matplot
# Lorenztian fit plot
fig = plt.figure(figsize=(10,5))
plot = GridSpec(1,1,left=0.125,bottom=0.15,right=0.95,top=0.925,wspace=0.1,hspace=0)
plt.subplot(plot[:,:])
plt.errorbar(nu, phi, yerr=err, fmt='.')
plt.plot(nu, lorentzian(nu, lorentzian_nu0, lorentzian_alpha), 'g-')
plt.suptitle("Lorentzian fit plot")
plt.xlabel(r'$\nu$', fontsize=20)
plt.ylabel(r'$\phi(\nu)$', fontsize=20)
plt.show()
plt.savefig('lorentz1.png')
plt.close()
# Gaussian fit plot
fit = plt.figure(figsize=(10,5))
plot = GridSpec(1,1,left=0.125,bottom=0.15,right=0.95,top=0.925,wspace=0.1,hspace=0)
plt.subplot(plot[:,:])
plt.errorbar(nu, phi, yerr=err, fmt='.')
plt.plot(nu, gaussian(nu, gaussian_nu0, gaussian_alpha), 'g-')
plt.suptitle("Gaussian fit plot")
plt.xlabel(r'$\nu$', fontsize=20)
plt.ylabel(r'$\phi(\nu)$', fontsize=20)
plt.show()
plt.savefig('gauss1.png')
plt.close()

