from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import pandas as pd

from astropy.coordinates import SkyCoord
from astropy import units as u

import sys
sys.path.append("../")
import colormap

import efficiency_estimate_juli3_2019 as eff

import matplotlib as mpl
inline_rc = dict(mpl.rcParams)
mpl.rcParams.update(inline_rc)

#This code shows hos the efficiency estimate was done for Wenzl et al 2021. IT IS NOT PLUG AND PLAY
#to apply these to new data you need to carefully review the code and make sure you include all your survey restrictions


cand = pd.read_csv("data/checked_candidates_cut_by_hand_final.csv")
print("all objects that I looked at:")
cand = cand.query("regression_highz>4.8")
print("ADDITIONAL RESTRITCTION APPLIED")

cand = cand.query("regression_highz > 5.6 or highz >= 0.8") ##option to increase efficiency!

print(cand["vis_id"].value_counts())
cand = cand.query("vis_id == 'good' or vis_id == 'good2'")
print("overall for good or good2 candidates we have {} total of which {} are known quasars".format(cand.shape[0], cand.query("Z_VI > 4").shape[0]))



controll_quasars = pd.read_csv("../../data/training/quasars_dr14_and_highz.csv")
uniform = eff.prepare_uniform_sample()
c_mstars, mstars = eff.prepare_mstar_sample()


c_cand = SkyCoord(ra=cand["ps_ra"].values*u.degree, dec=cand["ps_dec"].values*u.degree, frame='icrs')
print("DO I NEED TO REMOVE PREPARE CANDIDATES???")
c_cand,candidates = eff.prepare_candidates(cand, c_cand)
print("Number of candidates: {}".format(candidates.shape[0]))
median, q84, q16, upper, lower = eff.efficiency(c_cand, c_mstars, uniform)
print(median, "q84 ", q84, "q16 ", q16)

binning = 0.1
step =0.1
effs = []
upper_limits = []
lower_limits = []

min_efficiency = []

for i in np.arange(4.8, 6.1, step):
    cand_bin = cand.query("regression_highz >= {} and regression_highz < {}".format(i, (i+binning)))
    c_cand = SkyCoord(ra=cand_bin["ps_ra"].values*u.degree,
                     dec=cand_bin["ps_dec"].values*u.degree, frame='icrs')
    median, q84, q16, upper,lower = eff.efficiency(c_cand, c_mstars, uniform)
    effs.append(median)
    upper_limits.append(q84-median)
    lower_limits.append(-q16+median)
    min_efficiency.append(cand_bin.query("Z_VI >= 4").shape[0] /cand_bin.shape[0])

print(effs)


plt.figure(figsize=(6,3))
ax1 = plt.gca()

x= np.arange(4.8, 6.1, step)+binning/2
plt.errorbar(x, effs, yerr= [lower_limits, upper_limits], xerr=binning/2, fmt= "o", ecolor="grey", color=colormap.cmap[1])

def redraw_histogram(H, bins, args={}):
        #H = np.concatenate(([0], H, [0]))
        bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
        #bincentres = np.concatenate(([2*bincentres[0]-bincentres[1]], bincentres, [2*bincentres[-1]-bincentres[-2]]))
        plt.step(bincentres,H,where='mid', **args )
##plt.plot(x, min_efficiency)
x= np.arange(4.8, 6.1+step, step)
redraw_histogram(min_efficiency, x , {"color":colormap.cmap[0]})
#[[upper_limits[i], lower_limits[i]] for i in range(len(upper_limits))]
ax1.grid(linestyle='dotted')
ax1.tick_params(direction='in')
ax1.set_xlabel("Pred redshift (high redshift regression)", size="16")
ax1.set_ylabel("Efficiency", size="16")
ax1.tick_params(axis="x", labelsize=14)
ax1.tick_params(axis="y", labelsize=14)

plt.tight_layout()
plt.savefig("efficiency_highz_selection_RAW.pdf")
plt.show()
