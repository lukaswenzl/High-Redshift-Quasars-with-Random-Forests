import pandas as pd

from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

import sfdmap


#This code shows hos the efficiency estimate was done for Wenzl et al 2021. IT IS NOT PLUG AND PLAY
#to apply these to new data you need to carefully review the code and make sure you include all your survey restrictions


def sample_spherical(npoints):
    vec = np.random.randn(3, npoints)
    #vec = vec-0.5
    vec /= np.linalg.norm(vec, axis=0)
    phi = np.arctan2(vec[1],vec[0])+np.pi
    phi = np.degrees(phi)
    theta = np.arctan(vec[2]/np.sqrt(vec[0]**2+vec[1]**2))
    theta = np.degrees(theta)
    return phi,theta


def skyplots(ra_in_deg, dec_in_deg, alpha=0.05):
    c_proj = SkyCoord(ra=ra_in_deg*u.degree, dec=dec_in_deg*u.degree, frame='icrs')
    plt.figure()
    plt.subplot(111, projection="mollweide")#aitoff")
    plt.title("galactic")
    plt.grid(True)
    ra = c_proj.galactic.l.deg/360*2*np.pi
    ra[ra> np.pi] =  ra[ra>np.pi] - 2*np.pi
    ra = -1*ra
    #plt.plot(ra, c_proj.galactic.b.deg/360*2*np.pi ,".", label=".", color="red", alpha=alpha, linestyle=".")
    plt.scatter(ra, c_proj.galactic.b.deg/360*2*np.pi , label=".", color="red", alpha=alpha)


    #plt.hist2d(ra, c.galactic.b.deg/360*2*np.pi, bins=100)

    plt.legend()
    plt.locator_params(nbins=6)
    plt.xticks([-2*np.pi/3, -1*np.pi/3, 0, 1*np.pi/3, 2*np.pi/3],[r"120$^\circ$", r"60$^\circ$", r"0$^\circ$", r"300$^\circ$", r"240$^\circ$"])

    plt.figure()
    plt.subplot(111, projection="mollweide")
    plt.title("icrs")
    plt.grid(True)
    ra = c_proj.icrs.ra.deg/360*2*np.pi
    ra[ra> np.pi] =  ra[ra>np.pi] - 2*np.pi
    ra = -1*ra
    #plt.plot(ra, c_proj.icrs.dec.deg/360*2*np.pi ,".", label=".", color="red", alpha=alpha)
    plt.scatter(ra, c_proj.icrs.dec.deg/360*2*np.pi , label=".", color="red", alpha=alpha)


    plt.legend()
    plt.locator_params(nbins=6)
    plt.xticks([-2*np.pi/3, -1*np.pi/3, 0, 1*np.pi/3, 2*np.pi/3],[r"120$^\circ$", r"60$^\circ$", r"0$^\circ$", r"300$^\circ$", r"240$^\circ$"])

def skyplots_basic(ra_in_deg, dec_in_deg, alpha=0.05, color="red"):
    c_proj = SkyCoord(ra=ra_in_deg*u.degree, dec=dec_in_deg*u.degree, frame='icrs')
    #plt.subplot(111, projection="mollweide")#aitoff")
    plt.title("galactic") #
    plt.grid(True)
    ra = c_proj.galactic.l.deg/360*2*np.pi
    ra[ra> np.pi] =  ra[ra>np.pi] - 2*np.pi
    ra = -1*ra
    #plt.plot(ra, c_proj.galactic.b.deg/360*2*np.pi ,".", label=".", color=color, alpha=alpha)
    plt.scatter(ra, c_proj.galactic.b.deg/360*2*np.pi , label=".", color=color, alpha=alpha)


    plt.locator_params(nbins=6)
    plt.xticks([-2*np.pi/3, -1*np.pi/3, 0, 1*np.pi/3, 2*np.pi/3],[r"120$^\circ$", r"60$^\circ$", r"0$^\circ$", r"300$^\circ$", r"240$^\circ$"])


# deprecated: flipped x axis now!
# def skyplots_old(ra_in_deg, dec_in_deg, alpha=0.05):
#     c_proj = SkyCoord(ra=ra_in_deg*u.degree, dec=dec_in_deg*u.degree, frame='icrs')
#     plt.figure()
#     plt.subplot(111, projection="mollweide")#aitoff")
#     plt.title("galacitic (usually ra increasing to left, not here)")
#     plt.grid(True)
#     ra = c_proj.galactic.l.deg/360*2*np.pi
#     ra[ra> np.pi] =  ra[ra>np.pi] - 2*np.pi
#     plt.plot(ra, c_proj.galactic.b.deg/360*2*np.pi ,".", label=".", color="red", alpha=alpha)
#     #plt.hist2d(ra, c.galactic.b.deg/360*2*np.pi, bins=100)
#
#     plt.legend()
#     plt.locator_params(nbins=6)
#
#     plt.figure()
#
#     plt.subplot(111, projection="mollweide")
#     plt.title("icrs  (usually ra increasing to left, not here)")
#     plt.grid(True)
#     ra = c_proj.icrs.ra.deg/360*2*np.pi
#     ra[ra> np.pi] =  ra[ra>np.pi] - 2*np.pi
#     plt.plot(ra, c_proj.icrs.dec.deg/360*2*np.pi ,".", label=".", color="red", alpha=alpha)
#
#     plt.legend()
#     plt.locator_params(nbins=6)
#     print("not usually projections are fliped: center ra is zero, but increases to the left")

# def apply_restrictions(positions, df=None):
#     """Applying the conditions on the catalog area + dust. I make this a method to insure its the same for stars, unform distribution and candidates """
#     andromeda = SkyCoord('0h42m44s', '+41d16m9s', frame='icrs')
#     galactic_center = SkyCoord('0h0m0s', '+0d0m0s', frame='galactic')
#
#     sep_andromeda = positions.separation(andromeda).deg
#     sep_galactic_center = positions.separation(galactic_center).deg
#
#     total_number = len(positions.galactic.b.deg)
#     area_res = (sep_andromeda>5) & (sep_galactic_center > 30) & ((positions.galactic.b.deg > 20)|(positions.galactic.b.deg < -20)) & (positions.icrs.dec.deg > -30)
#     positions = positions[area_res]
#     if(df is not None):
#         #print(df)
#         df = df[area_res]
#
#     m = sfdmap.SFDMap('../class_photoz/data/dustmap_sfddata/')
#
#     total_number2 = len(positions.galactic.b.deg)
#
#     dust_cutoff = 0.1
#
#     ebv = m.ebv(positions)
#     positions = positions[ebv<dust_cutoff]
#     if(df is not None):
#         #print(df)
#         df = df[ebv<dust_cutoff]
#
#     total_number3 = len(positions.galactic.b.deg)
#     print("Applied restrictions. {} objects removed because of area. {} objects removed because of dust".format(total_number-total_number2, total_number2-total_number3))
#
#     return positions, df

def restr_ml_quasar(sep_andromeda, sep_galactic_center, positions):
    area_res = (sep_andromeda>5) & (sep_galactic_center > 30) & ((positions.galactic.b.deg > 20)|(positions.galactic.b.deg < -20)) & (positions.icrs.dec.deg > -30)
    return area_res

def restr_sdss_test(sep_andromeda, sep_galactic_center,sep_SDSS_artifact, positions):
    area_res = (sep_andromeda>5) & (sep_galactic_center > 30) & (sep_SDSS_artifact > 5) & ((positions.galactic.b.deg > 20)|(positions.galactic.b.deg < -20)) & (positions.icrs.dec.deg > -30) & (positions.icrs.dec.deg > 0) & (positions.icrs.dec.deg < 60) & (positions.icrs.ra.deg > 140) & (positions.icrs.ra.deg < 240)
    return area_res

def restr_sdss_test2(sep_andromeda, sep_galactic_center,sep_SDSS_artifact, positions):
    print("Test2: ra cutoff at 120")
    area_res = (sep_andromeda>5) & (sep_galactic_center > 30) & (sep_SDSS_artifact > 5) & ((positions.galactic.b.deg > 20)|(positions.galactic.b.deg < -20)) & (positions.icrs.dec.deg > -30) & (positions.icrs.dec.deg > 0) & (positions.icrs.dec.deg < 60) & (positions.icrs.ra.deg > 120) & (positions.icrs.ra.deg < 240)
    return area_res

def restr_sdss_test3(sep_andromeda, sep_galactic_center,sep_SDSS_artifact, positions):
    print("Test3: ra cutoff at 180")
    area_res = (sep_andromeda>5) & (sep_galactic_center > 30) & (sep_SDSS_artifact > 5) & ((positions.galactic.b.deg > 20)|(positions.galactic.b.deg < -20)) & (positions.icrs.dec.deg > -30) & (positions.icrs.dec.deg > 0) & (positions.icrs.dec.deg < 60) & (positions.icrs.ra.deg > 180) & (positions.icrs.ra.deg < 240)
    return area_res

def restr_sdss_test4(sep_andromeda, sep_galactic_center,sep_SDSS_artifact, positions):
    print("Test4: ra cutoff at 120, but dec cutoff at 20 ")
    area_res = (sep_andromeda>5) & (sep_galactic_center > 30) & (sep_SDSS_artifact > 5) & ((positions.galactic.b.deg > 20)|(positions.galactic.b.deg < -20)) & (positions.icrs.dec.deg > -30) & (positions.icrs.dec.deg > 20) & (positions.icrs.dec.deg < 60) & (positions.icrs.ra.deg > 120) & (positions.icrs.ra.deg < 240)
    return area_res

def restr_sdss_test5(sep_andromeda, sep_galactic_center,sep_SDSS_artifact, positions):
    print("Test5: ra cutoff at 120, but dec cutoff < 40 ")
    area_res = (sep_andromeda>5) & (sep_galactic_center > 30) & (sep_SDSS_artifact > 5) & ((positions.galactic.b.deg > 20)|(positions.galactic.b.deg < -20)) & (positions.icrs.dec.deg > -30) & (positions.icrs.dec.deg > 0) & (positions.icrs.dec.deg < 40) & (positions.icrs.ra.deg > 120) & (positions.icrs.ra.deg < 240)
    return area_res

def restr_sdss_test6(sep_andromeda, sep_galactic_center,sep_SDSS_artifact, positions):
    print("Test6: only problematic region removed ")
    area_res = (sep_andromeda>5) & (sep_galactic_center > 30) & (sep_SDSS_artifact > 5) & ((positions.galactic.b.deg > 20)|(positions.galactic.b.deg < -20)) & (positions.icrs.dec.deg > -30) & (positions.icrs.dec.deg < 60) & (((positions.icrs.dec.deg > 20) & (positions.icrs.ra.deg < 140) & (positions.icrs.ra.deg > 120)) | ((positions.icrs.dec.deg > 0)&(positions.icrs.ra.deg > 140))) & (positions.icrs.ra.deg < 240)
    return area_res

def apply_restrictions(positions, df=None, restrictions="ml_quasar"):
    """Applying the conditions on the catalog area + dust. I make this a method to insure its the same for stars, unform distribution and candidates """
    andromeda = SkyCoord('0h42m44s', '+41d16m9s', frame='icrs')
    galactic_center = SkyCoord('0h0m0s', '+0d0m0s', frame='galactic')

    sep_andromeda = positions.separation(andromeda).deg
    sep_galactic_center = positions.separation(galactic_center).deg

    total_number = len(positions.galactic.b.deg)
    if(restrictions == "ml_quasar"):
        area_res = restr_ml_quasar(sep_andromeda, sep_galactic_center, positions)
    elif("SDSS_test" in restrictions):
        print("cut out an SDSS artifact. If it needs an explanation")
        SDSS_artifact = SkyCoord("09h0m49s", "+47d15m34s", frame="icrs")
        sep_SDSS_artifact = positions.separation(SDSS_artifact).deg
        if("2" in restrictions):
            area_res = restr_sdss_test2(sep_andromeda, sep_galactic_center, sep_SDSS_artifact, positions)
        elif("3" in restrictions):
            area_res = restr_sdss_test3(sep_andromeda, sep_galactic_center, sep_SDSS_artifact, positions)
        elif("4" in restrictions):
            area_res = restr_sdss_test4(sep_andromeda, sep_galactic_center, sep_SDSS_artifact, positions)
        elif("5" in restrictions):
            area_res = restr_sdss_test5(sep_andromeda, sep_galactic_center, sep_SDSS_artifact, positions)
        elif("6" in restrictions):
            area_res = restr_sdss_test6(sep_andromeda, sep_galactic_center, sep_SDSS_artifact, positions)
        else:
            area_res = restr_sdss_test(sep_andromeda, sep_galactic_center, sep_SDSS_artifact, positions)
    else:
        print("WARNING: no valid restrictions specified")
        return None

    positions = positions[area_res]
    if(df is not None):
        #print(df)
        df = df[area_res]

    m = sfdmap.SFDMap('../../data/dustmaps/')

    total_number2 = len(positions.galactic.b.deg)

    dust_cutoff = 0.1

    ebv = m.ebv(positions)
    positions = positions[ebv<dust_cutoff]
    if(df is not None):
        #print(df)
        df = df[ebv<dust_cutoff]

    total_number3 = len(positions.galactic.b.deg)
    print("Applied restrictions. {} objects removed because of area. {} objects removed because of dust".format(total_number-total_number2, total_number2-total_number3))

    return positions, df




def prepare_uniform_sample(n=500000, restrictions="ml_quasar"):
    """Uniform sample on sky for n objects"""
    print(">Uniform sample")
    ra, dec = sample_spherical(n)
    uniform = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    total_number_uni = len(uniform.galactic.b.deg)
    #AREA AND DUST RESTRICTIONS
    uniform, _ = apply_restrictions(uniform, restrictions=restrictions)
    print("Uniform sample prepared. Represents {} of the sky".format(len(uniform.galactic.b.deg)/total_number_uni))
    return uniform

def prepare_mstar_sample(restrictions="ml_quasar"):
    print(">Mstar sample (now using mstars not random sample)")# (STILL USING RANDOM OBJECTS NOT MSTAR CLASSIFIED OBKJECTS)")
    #catalog = pd.read_csv("data/full2_sample1mio.csv")
    catalog = pd.read_csv("data/mstars_1mio.csv")
    c_cat = SkyCoord(ra=catalog["ps_ra"].values*u.degree, dec=catalog["ps_dec"].values*u.degree, frame='icrs')
    total_number_cat = len(c_cat.galactic.b.deg)
    #AREA AND DUST RESTRICTIONS
    c_cat, catalog = apply_restrictions(c_cat, catalog, restrictions)
    print("Mstars prepared. {} of the objects remain.".format(len(c_cat.galactic.b.deg)/total_number_cat))
    return c_cat, catalog

def prepare_kstar_sample(restrictions="ml_quasar"):
    print(">Kstar sample (now using mstars not random sample)")# (STILL USING RANDOM OBJECTS NOT MSTAR CLASSIFIED OBKJECTS)")
    #catalog = pd.read_csv("data/full2_sample1mio.csv")
    catalog = pd.read_csv("data/kstars_1mio.csv")
    c_cat = SkyCoord(ra=catalog["ps_ra"].values*u.degree, dec=catalog["ps_dec"].values*u.degree, frame='icrs')
    total_number_cat = len(c_cat.galactic.b.deg)
    #AREA AND DUST RESTRICTIONS
    c_cat, catalog = apply_restrictions(c_cat, catalog, restrictions)
    print("Kstars prepared. {} of the objects remain.".format(len(c_cat.galactic.b.deg)/total_number_cat))
    return c_cat, catalog

def prepare_m_kstar_sample(restrictions="ml_quasar"):
    print(">M and Kstar sample (now using mstars not random sample)")# (STILL USING RANDOM OBJECTS NOT MSTAR CLASSIFIED OBKJECTS)")
    #catalog = pd.read_csv("data/full2_sample1mio.csv")
    catalog = pd.read_csv("data/kstars_1mio.csv")
    catalog2 = pd.read_csv("data/mstars_1mio.csv")
    catalog = pd.concat([catalog, catalog2])
    c_cat = SkyCoord(ra=catalog["ps_ra"].values*u.degree, dec=catalog["ps_dec"].values*u.degree, frame='icrs')
    total_number_cat = len(c_cat.galactic.b.deg)
    #AREA AND DUST RESTRICTIONS
    c_cat, catalog = apply_restrictions(c_cat, catalog, restrictions)
    print("M and Kstars prepared. {} of the objects remain.".format(len(c_cat.galactic.b.deg)/total_number_cat))
    return c_cat, catalog

def prepare_kstar_sample_sdss_sample(restrictions="ml_quasar"):
    print(">Kstar sample (simply the K stars from my traning set)")# (STILL USING RANDOM OBJECTS NOT MSTAR CLASSIFIED OBKJECTS)")
    #catalog = pd.read_csv("data/full2_sample1mio.csv")
    catalog = pd.read_csv("data/stars_sdss_and_dwarfs.csv")
    catalog = catalog.query("mult_class_true == 'K'")
    c_cat = SkyCoord(ra=catalog["ps_ra"].values*u.degree, dec=catalog["ps_dec"].values*u.degree, frame='icrs')
    total_number_cat = len(c_cat.galactic.b.deg)
    #AREA AND DUST RESTRICTIONS
    c_cat, catalog = apply_restrictions(c_cat, catalog, restrictions)
    print("Kstars prepared. {} of the objects remain.".format(len(c_cat.galactic.b.deg)/total_number_cat))
    return c_cat, catalog


def prepare_candidates(candidates, positions, restrictions="ml_quasar"):
    print(">candidates")
    positions, candidates = apply_restrictions(positions, candidates, restrictions)
    print("Prepared candidates set")
    return positions, candidates


from scipy.optimize import minimize
def compare(ratio, hist_cand, hist_mstars, hist_uni):
    if(ratio <0 or ratio > 1):
        return 9999
    return (np.abs(((1-ratio)*hist_mstars + ratio*hist_uni - hist_cand))).sum()/len(hist_cand)
def compare_sqrt(ratio, hist_cand, hist_mstars, hist_uni):
    if(ratio <0 or ratio > 1):
        return 9999
    return (np.sqrt(np.abs(((1-ratio)*hist_mstars + ratio*hist_uni - hist_cand)))).sum()/len(hist_cand)
# def compare_l(ratio, , hist_cand, hist_mstars, hist_uni):
#     if(ratio <0 or ratio > 1):
#         return 9999
#     return (np.abs(((1-ratio)*hist_mstars_l + ratio*hist_uni_l - hist_cand_l))).sum()/len(hist_cand)

def efficiency_oldWITHPYPLOT(c_cand, c_mstars, uniform):

    ratios=[]
    #range: 20 to 100
    steps = np.random.rand(20)*80+20

    for n in steps:#[20,14,39, 37,17,40,50,56]:
        n_int = int (n)
        hist_cand = plt.hist(c_cand.galactic.b, bins=n_int, label="candidates", normed=True)
        bins= hist_cand[1]
        hist_cand = hist_cand[0]
        hist_mstars = plt.hist(c_mstars.galactic.b, bins=bins, label="catalog", normed=True, alpha=0.5)
        hist_mstars = hist_mstars[0]
        hist_uni = plt.hist(uniform.galactic.b, bins=bins, label="uniform", normed=True, alpha=0.5);
        hist_uni = hist_uni[0]
        res =minimize(compare, [0.5], args=(hist_cand, hist_mstars, hist_uni))
        print(res.fun)
        print(res.x)
        if(res.fun != 9999 and res.x[0] >= 0 and res.x[0] <=1 ):
            ratios.append(res.x[0])
    #print(np.array(ratios).mean())
    #print(np.array(ratios).std())
    median = np.median(ratios)
    q75, q25 = np.percentile(ratios, [75 ,25])
    #print("max diff to med {} min diff to median {} quartile1 diff to median: {} quartile3 diff to median: {} median: {}".format(-median+np.array(ratios).max(),
    #                                -median+np.array(ratios).min(), -median+q75, -median+q25 , median ))

    print("efficiency = {:.4f} +{:.4f}{:.4f} (quartiles) +{:.4f}{:.4f} (min max)".format( median, -median+q75, -median+q25, -median+np.array(ratios).max(),
                                -median+np.array(ratios).min()  ))

    # print(-median+np.array(ratios).max())
    # print(-median+np.array(ratios).min())
    # print("quartiles")
    # print(-median+q75)
    # print(-median+q25)
    # print("median")
    # print(median)
    return median, q75, q25


def efficiency(c_cand, c_mstars, uniform):

    ratios=[]
    #range: 20 to 100
    steps = np.random.rand(20)*80+20
    ns = []

    for n in steps:#[20,14,39, 37,17,40,50,56]:
        n_int = int (n)
        hist_cand = np.histogram(c_cand.galactic.b, bins=n_int, density=True)
        bins= hist_cand[1]
        hist_cand = hist_cand[0].astype(float)
        hist_mstars = np.histogram(c_mstars.galactic.b, bins=bins, density=True)
        hist_mstars = hist_mstars[0].astype(float)
        hist_uni = np.histogram(uniform.galactic.b, bins=bins, density=True);
        hist_uni = hist_uni[0].astype(float)
        res =minimize(compare, [0.5], args=(hist_cand.value, hist_mstars.value, hist_uni.value))
        if(res.fun != 9999 and res.x[0] >= 0 and res.x[0] <=1 ):
            ratios.append(res.x[0])
            ns.append(n)
    print(ratios)
    median = np.median(ratios)
    print("CHANGED from quartiles to sigma!")
    q84, q16 = np.percentile(ratios, [84, 16])
    print("efficiency = {:.4f} +{:.4f}{:.4f} (16th and 84th percentile) +{:.4f}{:.4f} (min max)".format( median, -median+q84, -median+q16, -median+np.array(ratios).max(),
                                -median+np.array(ratios).min()  ))
    #return median, q84, q16, -median+np.array(ratios).max(), -median+np.array(ratios).min()
    info = {"median":median, "mean":np.mean(ratios), "q84":q84, "q16":q16, "ratios":ratios, "ns":ns}
    return median, q84, q16, info
