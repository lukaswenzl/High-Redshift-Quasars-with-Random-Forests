from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy import units as u

import efficiency_estimate as eff #################!!! THIS IS SPECIFIC FOR THE SDSS DATA!!

#This code shows hos the efficiency estimate was done for Wenzl et al 2021. IT IS NOT PLUG AND PLAY
#to apply these to new data you need to carefully review the code and make sure you include all your survey restrictions

RECALC_EVERYTHING = False #set True to recalc

if(RECALC_EVERYTHING):
    uniform = eff.prepare_uniform_sample(restrictions="SDSS_test")
    c_mstars, mstars = eff.prepare_mstar_sample(restrictions="SDSS_test")

    sdss = pd.read_csv("data/SDSS_QSO_HIZ_original_candidates.csv")
    sdss["indentified_as"] = sdss["class"]
    sdss = sdss.query(" (indentified_as == 'QSO' and z>0.5) or (indentified_as == 'STAR')")
    c_sdss = SkyCoord(ra=sdss["ra"].values*u.degree, dec=sdss["dec"].values*u.degree, frame='icrs')
    c_sdss,sdss = eff.prepare_candidates(sdss, c_sdss, restrictions="SDSS_test")

    print(sdss["class"].value_counts())

    n_qso = sdss["class"].value_counts()["QSO"]
    n_stars = sdss["class"].value_counts()["STAR"]


true_effs = []
pred_effs = []
up_errors =[]
down_errors =[]


if(RECALC_EVERYTHING):
    for i in [ 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]:
        sdss['randNumCol'] = np.random.randint(1, 100, sdss.shape[0])
        #star_cutoff = np.random.randint(1, 100, 1)
        n = (n_qso-i*n_qso)/i
        cutoff = 100.0*n/n_stars
        #print("cutoff: {}".format(cutoff))
        local = sdss.query(" (indentified_as == 'QSO' and z>0.5) or (indentified_as == 'STAR' and randNumCol <{})".format(cutoff))
        c_local = SkyCoord(ra=local["ra"].values*u.degree, dec=local["dec"].values*u.degree, frame='icrs')
        median, q84, q16, info = eff.efficiency(c_local, c_mstars, uniform)
        pred_effs.append(median)
        up_errors.append(q84-median)
        down_errors.append(median-q16)
        try:
            loc_eff = local["class"].value_counts()["QSO"] /(local["class"].value_counts()["QSO"]+local["class"].value_counts()["STAR"])
        except:
            loc_eff=1

        true_effs.append(loc_eff)
        print(loc_eff)


    for i in [ 0.8, 0.9]:
        sdss['randNumCol'] = np.random.randint(1, 100, sdss.shape[0])
        #star_cutoff = np.random.randint(1, 100, 1)
        n = (n_stars-i*n_stars)/i
        cutoff = 100.0*n/n_qso
        #print("cutoff: {}".format(cutoff))
        local = sdss.query(" (indentified_as == 'QSO' and z>0.5 and randNumCol <{}) or (indentified_as == 'STAR')".format(cutoff))
        c_local = SkyCoord(ra=local["ra"].values*u.degree, dec=local["dec"].values*u.degree, frame='icrs')
        median, q84, q16, info = eff.efficiency(c_local, c_mstars, uniform)
        pred_effs.append(median)
        up_errors.append(q84-median)
        down_errors.append(median-q16)
        try:
            loc_eff = local["class"].value_counts()["QSO"] /(local["class"].value_counts()["QSO"]+local["class"].value_counts()["STAR"])
        except:
            loc_eff=1

        true_effs.append(loc_eff)
        print(loc_eff)
    np.savetxt('efficiencies_v2.out', (true_effs,pred_effs, down_errors, up_errors))
else:
    (true_effs,pred_effs, down_errors, up_errors) = np.loadtxt("efficiencies_v2.out")




plt.figure(figsize=(4,4))
x = np.arange(0,1.01, 0.1)
plt.ylim(0,1.01)
plt.xlim(0,1.01)

plt.plot(x,x, color="black")

#plt.plot(true_effs_2, np.array(pred_effs_2) ,'.', label="using SDSS stars")
#plt.plot(true_effs_3, np.array(pred_effs_3) ,'.', label="using SDSS qso")

print(len(down_errors))
print(len(up_errors))
plt.errorbar(true_effs, np.array(pred_effs), [down_errors, up_errors], fmt=".", color="#E69F00" )
#plt.legend(loc="best")
plt.xlabel("True efficiency", size=14)
plt.ylabel("Predicted efficiency", size=14)
plt.tight_layout()
plt.savefig("data/efficiency_estimate_SDSS_SCRIPT_REDONE.pdf")
#DIFFERENCIATE BY REDSHIFT!!!
