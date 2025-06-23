###AmBe plots ##
import os          
import numpy as np
import uproot
from tqdm import trange
from scipy.stats import norm
import re
import AmBeNeutronEff as ane
import ProfileLikelihoodBuilder as plb
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as scm
import scipy.optimize as scp
import seaborn as sns

import urllib.request
import types


expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B
mypoisson = lambda x,mu: (mu**x)*np.exp(-mu)/scm.factorial(x)
mypoissons = lambda x,R1,mu1,R2,mu2: R1*(mu1**x)*np.exp(-mu2)/scm.factorial(x) + R2*(mu2**x)*np.exp(-mu2)/scm.factorial(x)

BKG_WINDOW_START = 2000
SIGNAL_WINDOW_START = 2000
NUMTHROWS = 1E6

EffRanges = {'Position 0': np.arange(0,1,0.05), 'Position 1': np.arange(0.46,0.66,0.001),
        'Position 2': np.arange(0.33, 0.43, 0.001), 'Position 3': np.arange(0.25,0.35, 0.001)}
BkgRanges = {'Position 0': np.arange(0,1,0.01), 'Position 1': np.arange(0.055,0.085,0.005),
        'Position 2': np.arange(0.055,0.085,0.005), 'Position 3': np.arange(0.055,0.085,0.005)}

POSITION_TO_ANALYZE = "Position 0"

##open the AmBe Neutron Candidates file csv
df = pd.read_csv('EventAmBeNeutronCandidates_4499.csv')       # Replace with actual file path or name
bdf = pd.read_csv('BackgroundAmBeNeutronCandidates_4496.csv') # Replace accordingly

def NiceBins(theax, bin_left,bin_right,value,color,llabel):
    #xkcd_colors = [color for x in range(len(value)*2)]
    #sns.set_palette(sns.xkcd_palette(xkcd_colors))
    for j,val in enumerate(value):
        if j == len(value)-1:
            theax.plot([bin_left[j],bin_right[j]],[val,val],linewidth=6,linestyle='-',label=llabel)
            theax.plot([bin_right[j],bin_right[j]],[val,0],linewidth=6,linestyle='-')
            break
        elif j == 0:
            #ax.plot([0,bin_left[j]],[0,0],linewidth=6,linestyle='-')
            theax.plot([bin_left[j],bin_left[j]],[0,val],linewidth=6,linestyle='-')
            theax.plot([bin_left[j],bin_right[j]],[val,val],linewidth=6,linestyle='-')
            theax.plot([bin_right[j],bin_right[j]],[val,value[j+1]],linewidth=6,linestyle='-')
        else:
            theax.plot([bin_left[j],bin_right[j]],[val,val],linewidth=6,linestyle='-')
            theax.plot([bin_right[j],bin_right[j]],[val,value[j+1]],linewidth=6,linestyle='-')
    return theax


plt.hist(df['clusterTime'], bins=30, range=(0,70000), alpha=0.75, histtype='stepfilled', linewidth=6, color='blue')
plt.title("Cluster Time distribution")
plt.xlabel("Cluster Time")
plt.ylabel("Number of clusters")
plt.show()

##Background AmBe Neutron Candidates
plt.hist(bdf['clusterTime'], bins=30, range=(0,70000), alpha=0.75, histtype='stepfilled', linewidth=6, color='blue')
plt.title("Cluster Time distribution")
plt.xlabel("Cluster Time")
plt.ylabel("Number of clusters")
plt.show()

##Count how many times each eventID appears
event_counts = df['eventID'].value_counts()
Background_counts = bdf['eventID'].value_counts()

##Count how many eventIDs have the same multiplicity
#multiplicity_counts = event_counts.value_counts().sort_index()
#Background_multiplicity_counts = Background_counts.value_counts().sort_index()

plt.hist(event_counts, bins=range(0, 5), log=True, edgecolor='black', align='left')
plt.xlabel('Neutron multiplicity')
plt.ylabel('No. of AmBe Neutron Candidates in Event')
plt.title('Neutron multiplicity distribution')
plt.grid(True, linestyle='--', alpha=0.5)
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.show()

plt.hist(Background_counts, bins=range(0, 5), log=True, edgecolor='black', align='left')
plt.xlabel('Background Neutron multiplicity')
plt.ylabel('No. of AmBe Neutron Candidates in Background')
plt.title('Neutron multiplicity distribution for run')
plt.grid(True, linestyle='--', alpha=0.5)
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.show()

Bbins,Bbin_edges = np.histogram(Background_counts,range=(0,5),bins=5)
print("BINS AND EDGES")
print(Bbins)
print(Bbin_edges)
Bbins_lefts = Bbin_edges[0:len(Bbin_edges)-1] #Combine clusters of 19 and 20 at end... negligible effect
Bbins_normed = Bbins/float(np.sum(Bbins))
Bbins_normed_unc = np.sqrt(Bbins)/float(np.sum(Bbins))
zero_bins = np.where(Bbins_normed_unc==0)[0]
Bbins_normed_unc[zero_bins] = 1.15/float(np.sum(Bbins))
print("BBins_normed: " + str(Bbins_normed))
init_params = [1]
popt, pcov = scp.curve_fit(mypoisson, Bbins_lefts,Bbins_normed,p0=init_params, maxfev=6000,sigma=Bbins_normed_unc)
#init_params = [5000,0.04,100,1]
#popt, pcov = scp.curve_fit(mypoissons, Bbins_lefts,Bbins_normed,p0=init_params, maxfev=6000,sigma=Bbins_normed_unc)
print('BEST FIT POPTS: ' + str(popt))
myy = mypoisson(Bbins_lefts,popt[0])
myy_upper = mypoisson(Bbins_lefts,popt[0]+np.sqrt(pcov[0][0]))
#myy = mypoissons(Bbins_lefts,popt[0],popt[1],popt[2],popt[3])
plt.errorbar(x=Bbins_lefts,y=Bbins_normed,yerr=Bbins_normed_unc,linestyle='None',marker='o',label='No source ($t > 20\\,\\mu\\mathrm{s}$)')
plt.plot(Bbins_lefts,myy,marker='None',linewidth=6,label=r'Best poiss. fit $\mu= %s \pm %s$'%(str(np.round(popt[0],2)),str(np.round(np.sqrt(pcov[0]),2))),color='black')
plt.plot(Bbins_lefts,myy_upper,marker='None',linewidth=6,label=r'Best poiss. fit upper bound',color='gray')
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.show()

Sbins,Sbin_edges = np.histogram(event_counts,range=(0,5),bins=5)
print("SIGNAL BINS AND EDGES")
print(Sbins)
print(Sbin_edges)
Sbins_lefts = Sbin_edges[0:len(Bbin_edges)-1] #Combine clusters of 19 and 20 at end... negligible effect
Sbins_normed = Sbins/float(np.sum(Sbins))
Sbins_normed_unc = np.sqrt(Sbins)/float(np.sum(Sbins))
zero_bins = np.where(Sbins_normed_unc==0)[0]
Sbins_normed_unc[zero_bins] = 1.15/float(np.sum(Sbins))

plt.errorbar(x=Sbins_lefts,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='Source ($t>2 \\, \\mu\\mathrm{s}$)',markersize=12)
plt.errorbar(x=Bbins_lefts,y=Bbins_normed,yerr=Bbins_normed_unc,linestyle='None',marker='o',label='No source ($t>20 \\, \\mu\\mathrm{s}$)',markersize=12)
plt.plot(Bbins_lefts,myy,marker='None',linewidth=6,label=r'Best poiss. fit $\mu= %s \pm %s$'%(str(np.round(popt[0],2)),str(np.round(np.sqrt(pcov[0]),2))),color='black')
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.show()

PLBuilder = plb.ProfileLikelihoodBuilder()
BkgScaleFactor = (67000-SIGNAL_WINDOW_START)/(67000-BKG_WINDOW_START)  #Scale mean neutrons per window up by this
PLBuilder.SetBkgMean(BkgScaleFactor*popt[0])
PLBuilder.SetBkgMeanUnc(np.sqrt(pcov[0][0]))
NeutronProbProfile =EffRanges[POSITION_TO_ANALYZE] 
#TODO: Also return the multiplicity array to make a histogram
ChiSquare,lowestChiSqProfile = PLBuilder.BuildLikelihoodProfile(NeutronProbProfile,Sbins_normed,Sbins_normed_unc,NUMTHROWS,Bbins_normed,Bbins_normed_unc)
print("MINIMUM CHI SQUARE: " + str(np.min(ChiSquare)))
ChiSquare_normed = ChiSquare/np.min(ChiSquare)
plt.errorbar(x=Sbins_lefts+0.5,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='Source ($t>2 \\, \\mu\\mathrm{s}$)',markersize=12)
#TODO: replace this with the multiplicity histogram returned above
plt.plot(Sbins_lefts+0.5, lowestChiSqProfile,linestyle='None',marker='o',label='Data-driven best fit',markersize=12,color='blue')
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.title("Best fit model profiles to central AmBe source data")
plt.xlabel("Delayed cluster multiplicity")
plt.show()

plt.plot(NeutronProbProfile,ChiSquare_normed,marker='None',linewidth=6,label='Data-driven model',color='red')
plt.title("Normalized Chi-square test parameter \n as a function of neutron detection efficiency")
plt.xlabel("Neutron detection efficiency $\\epsilon_{n}$")
plt.ylabel("$\\chi^{2}$/$\\chi^{2}_{min}$")
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.show()

plt.plot(NeutronProbProfile,ChiSquare-np.min(ChiSquare),marker='None',linewidth=6,label='Data-driven model',color='red')
plt.title("Chi-square test parameter \n as a function of neutron detection efficiency")
plt.xlabel("Neutron detection efficiency $\\epsilon_{n}$")
plt.ylabel("$\\chi^{2} - \\chi^{2}_{min}$")
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.show()


PLBuilder2D = plb.ProfileLikelihoodBuilder2D()
neutron_efficiencies = EffRanges[POSITION_TO_ANALYZE]
background_mean = BkgRanges[POSITION_TO_ANALYZE]
PLBuilder2D.SetEffProfile(neutron_efficiencies)
PLBuilder2D.SetBkgMeanProfile(background_mean)
x_var, y_var, ChiSquare,lowestChiSqProfileUncorr = PLBuilder2D.BuildLikelihoodProfile(Sbins_normed,Sbins_normed_unc,NUMTHROWS)
print("MINIMUM CHI SQUARE: " + str(np.min(ChiSquare)))
plt.errorbar(x=Sbins_lefts,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='Source ($t>2 \\, \\mu\\mathrm{s}$)',markersize=12)
plt.plot(Sbins_lefts, lowestChiSqProfileUncorr,linestyle='None',marker='o',label='Best fit model profile',markersize=12)
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.title("Best fit MC profile relative to central source data")
plt.xlabel("Delayed cluster multiplicity")
plt.show()

#Look at 2D chi-squared map
chisq_map = pd.DataFrame({"Neutron detection efficiency":np.round(x_var,3), "Background rate [candidates/trigger]":np.round(y_var,3),"ChiSq":ChiSquare/np.min(ChiSquare)})
cmap = chisq_map.pivot(index="Neutron detection efficiency",columns="Background rate [candidates/trigger]",values="ChiSq")
ax = sns.heatmap(cmap,vmin=1,vmax=10)
plt.title("$\\chi^{2}$/$\\chi_{min}^{2}$ for profile likelihood parameters")
plt.show()

chisq_map = pd.DataFrame({"Neutron detection efficiency":np.round(x_var,3), "Background rate [candidates/trigger]":np.round(y_var,3),"ChiSq":ChiSquare - np.min(ChiSquare)})
cmap = chisq_map.pivot(index="Neutron detection efficiency",columns="Background rate [candidates/trigger]",values="ChiSq")
ax = sns.heatmap(cmap,vmin=0,vmax=80)
plt.title("$\\chi^{2} - \\chi_{min}^{2}$ for profile likelihood parameters")
plt.show()

LowestInd = np.where(ChiSquare==np.min(ChiSquare))[0]
best_eff = x_var[LowestInd]
best_mean = y_var[LowestInd]
best_eff_chisquareinds = np.where(x_var==best_eff)[0]
best_eff_chisquares = ChiSquare[best_eff_chisquareinds]
best_eff_bkgmeans = y_var[best_eff_chisquareinds]
plt.plot(best_eff_bkgmeans,best_eff_chisquares/np.min(ChiSquare),marker='None',linewidth=6, label = 'Uncorr. bkg. model',color='blue')
plt.title("Normalized Chi-square test parameter as $\\lambda_{n}$ varies \n (best-fit detection efficiency $\\epsilon_{n}$ fixed)")
plt.xlabel("Background mean $\\lambda_{n}$ [clusters/trigger]")
plt.ylabel("$\\chi^{2}$/$\\chi^{2}_{min}$")
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.show()
plt.plot(best_eff_bkgmeans,best_eff_chisquares-np.min(ChiSquare),marker='None',linewidth=6, label = 'Uncorr. bkg. model',color='blue')
plt.title("Normalized Chi-square test parameter as $\\lambda_{n}$ varies \n (best-fit detection efficiency $\\epsilon_{n}$ fixed)")
plt.xlabel("Background mean $\\lambda_{n}$ [clusters/trigger]")
plt.ylabel("$\\chi^{2} - \\chi^{2}_{min}$")
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.show()


best_mean_chisquareinds = np.where(y_var==best_mean)[0]
best_mean_chisquares = ChiSquare[best_mean_chisquareinds]
best_mean_efficiencypro = x_var[best_mean_chisquareinds]
plt.plot(best_mean_efficiencypro,best_mean_chisquares/np.min(ChiSquare),marker='None',linewidth=6, label = 'Uncorr. bkg. model',color='blue')
plt.title("Normalized Chi-square test parameter as $\\epsilon_{n}$ varies \n (best-fit background rate $\\lambda_{n}$ fixed)")
plt.xlabel("Neutron detection efficiency $\\epsilon_{n}$")
plt.ylabel("$\\chi^{2}$/$\\chi^{2}_{min}$")
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.show()

plt.plot(best_mean_efficiencypro,best_mean_chisquares - np.min(ChiSquare),marker='None',linewidth=6, label = 'Uncorr. bkg. model',color='blue')
plt.title("Normalized Chi-square test parameter as $\\epsilon_{n}$ varies \n (best-fit background rate $\\lambda_{n}$ fixed)")
plt.xlabel("Neutron detection efficiency $\\epsilon_{n}$")
plt.ylabel("$\\chi^{2} - \\chi^{2}_{min}$")
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.show()


#Compare the two model fits to the signal data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
bin_rights = Sbins_lefts + (Sbins_lefts[1] - Sbins_lefts[0])
ax = NiceBins(ax,Sbins_lefts,bin_rights,lowestChiSqProfileUncorr,'dark blue',"Uncorr. bkg. best fit")
ax = NiceBins(ax,Sbins_lefts,bin_rights,lowestChiSqProfile,'dark red',"Data-driven bkg. best fit")
ax.errorbar(x=Sbins_lefts+0.5,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='AmBe data',markersize=12,color='black')
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.title("Best fit multiplicity distributions to central source data")
plt.xlabel("Neutron candidate multiplicity")
plt.show()

