from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

from astropy.coordinates import SkyCoord
from astropy import units as u

import efficiency_estimate_SDSS as eff #################!!! THIS IS SPECIFIC FOR THE SDSS DATA!!

#This code shows hos the efficiency estimate was done for Wenzl et al 2021. IT IS NOT PLUG AND PLAY
#to apply these to new data you need to carefully review the code and make sure you include all your survey restrictions



uniform = eff.prepare_uniform_sample(restrictions="SDSS_test")
c_mstars, mstars = eff.prepare_mstar_sample(restrictions="SDSS_test")
sdss_all = pd.read_csv("data/SDSS_QSO_HIZ_original_candidates.csv")
sdss_all["indentified_as"] = sdss_all["class"]
sdss_all = sdss_all.query(" (indentified_as == 'QSO' and z>0.5) or (indentified_as == 'STAR')")
c_sdss_all = SkyCoord(ra=sdss_all["ra"].values*u.degree, dec=sdss_all["dec"].values*u.degree, frame='icrs')

c_sdss,sdss = eff.prepare_candidates(sdss_all, c_sdss_all,restrictions="SDSS_test" )
median, q84, q16, info = eff.efficiency(c_sdss, c_mstars, uniform)


from matplotlib import gridspec
def estimate_sdss(sdss, c_sdss, c_mstars, uniform, median):
    n = 20
    plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
    ax1 = plt.subplot(gs[0])
    hist_cand = plt.hist(c_sdss.galactic.b.value, bins=n, label="candidates", density=True, fill=True, facecolor ="grey",edgecolor = "none")
    bins= hist_cand[1]
    hist_cand = hist_cand[0]

    # hist_cat = plt.hist(c_mstars.galactic.b, bins=bins, label="stars", normed=True, alpha=0.5)
    # hist_cat = hist_cat[0]
    # hist_uni = plt.hist(uniform.galactic.b, bins=bins, label="uniform", normed=True, alpha=0.5);
    # hist_uni = hist_uni[0]

    H_stars, bins = np.histogram(c_mstars.galactic.b.value, bins=bins, density=True)
    H_stars = H_stars

    H_uniform, bins = np.histogram(uniform.galactic.b.value, bins=bins, density=True)
    H_uniform = H_uniform

    def redraw_histogram(H, bins, args={}):
        H = np.concatenate(([0], H, [0]))
        bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
        bincentres = np.concatenate(([2*bincentres[0]-bincentres[1]], bincentres, [2*bincentres[-1]-bincentres[-2]]))
        plt.step(bincentres,H,where='mid', **args)


    redraw_histogram(H_uniform*median+H_stars*(1-median), bins, args={"color":"#E69F00", "linewidth":2,"linestyle":"-", "label":"fit"})
    redraw_histogram(H_stars, bins, args={"color":"#56B4E9", "linewidth":2, "linestyle":":", "label":"stars"})
    redraw_histogram(H_uniform, bins, args={"color":"#CC79A7","linewidth":2, "linestyle":"--", "label":"uniform"})

    plt.xlim(20,90)
    plt.legend(loc="upper left")
    #plt.xlabel("Galactic latitude (b)", size="16")
    plt.ylabel("Probability density", size="16")
    plt.locator_params(axis='y', nbins=4)
    #plt.tick_params(axis="x", labelsize=14)
    for n, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        label.set_visible(False)
    plt.tick_params(axis="y", labelsize=14)
    plt.grid()

    a = plt.axes([.69, .65, .2, .2], projection="mollweide")#, facecolor='k'
    loc = sdss
    if(sdss.shape[0]>= 1000):
        loc = sdss.sample(1000)
    eff.skyplots_basic(loc["ra"].values, loc["dec"].values, color="grey", alpha=1)
    for n, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        label.set_visible(False)
    for n, label in enumerate(plt.gca().yaxis.get_ticklabels()):
        label.set_visible(False)

    plt.subplot(gs[1], sharex=ax1)

    redraw_histogram(H_uniform*median+H_stars*(1-median)-hist_cand, bins, args={"color":"#E69F00", "linewidth":2,"linestyle":"-", "label":"fit"})
    redraw_histogram(H_stars-hist_cand, bins, args={"color":"#56B4E9", "linewidth":2, "linestyle":":", "label":"stars"})
    redraw_histogram(H_uniform-hist_cand, bins, args={"color":"#CC79A7","linewidth":2, "linestyle":"--", "label":"uniform"})
    plt.xlim(20,90)
    plt.ylabel("Diff", size="16")
    plt.xlabel("Galactic latitude (b)", size="16")
    plt.locator_params(axis='y', nbins=4)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.grid()
    plt.gca().yaxis.get_ticklabels()[0].set_visible(False)
    plt.gca().yaxis.get_ticklabels()[-1].set_visible(False)

print((sdss.indentified_as == "QSO").sum() /sdss.shape[0])
estimate_sdss(sdss, c_sdss, c_mstars, uniform, median)


##plt.tight_layout()
plt.savefig("data/sdss_example_plot_final_v2RERUN.pdf", bbox_inches='tight')
plt.show()
