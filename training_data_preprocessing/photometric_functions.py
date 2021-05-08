import numpy as np
import pandas as pd

def VEGAtoAB(VEGAMAG_in,band_name):
    #VEGAMAG = float(VEGAMAG_in)
    #VEGAMAG = [float(i) for i in VEGAMAG_in]
    VEGAMAG = pd.to_numeric(VEGAMAG_in,errors='coerce')
    mag_corr_dict = {'SDSS_u':-0.04,
                    'SDSS_g':0.0,
                    'SDSS_r':0.0,
                    'SDSS_i':0.0,
                    'SDSS_z':0.02,
                    'TMASS_j':0.894,
                    'TMASS_h':1.374,
                    'TMASS_k':1.84,
                    'WISE_w1':2.699,  #note this site has slightly different constants http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html#WISEZMA
                    'WISE_w2':3.339,  #should not be a problem since that is well withing the systematic uncertainty in vega mags also discussed by the site
                    'WISE_w3':5.174,
                    'WISE_w4':6.62,
                    'UNWISE_w1':2.699,
                    'UNWISE_w2':3.339,
                    'PS_g': 0.0,
                    'PS_r': 0.0,
                    'PS_i': 0.0,
                    'PS_z': 0.0,
                    'PS_y': 0.0
                    }

    return VEGAMAG+mag_corr_dict[band_name]

# def VEGAtoAB_flux(band_name):
#     mag_corr_dict = {'SDSS_u':-0.04,
#                     'SDSS_g':0.0,
#                     'SDSS_r':0.0,
#                     'SDSS_i':0.0,
#                     'SDSS_z':0.02,
#                     'TMASS_j':0.894,
#                     'TMASS_h':1.374,
#                     'TMASS_k':1.84,
#                     'WISE_w1':2.699,
#                     'WISE_w2':3.339,
#                     'WISE_w3':5.174,
#                     'WISE_w4':6.62,
#                     'UNWISE_w1':2.699,
#                     'UNWISE_w2':3.339,
#                     'PS_g': 0.0,
#                     'PS_r': 0.0,
#                     'PS_i': 0.0,
#                     'PS_z': 0.0,
#                     'PS_y': 0.0
#                     }
#
#     return np.power(10,-0.4*mag_corr_dict[band_name])  #THere is a factor missing here. I do not think i am using this, will comment out


def deredden_flux(flux, flux_band, ext, ext_band):
    ext_deltamag_dict = \
        {'EBV': 1.0,
	 'A_V':3.1,
        'SDSS_u':4.239,
        'SDSS_g':3.303,
        'SDSS_r':2.285,
        'SDSS_i':1.698,
        'SDSS_z':1.263,
        'TMASS_j':0.709,
        'TMASS_h':0.449,
        'TMASS_k':0.302,
        'WISE_w1':0.189,
        'WISE_w2':0.146,
        'WISE_w3':0.0,
        'WISE_w4':0.0,
        'UNWISE_w1':0.189,
        'UNWISE_w2':0.146,
        'PS_g': 3.172 ,
        'PS_r': 2.271,
        'PS_i': 1.682,
        'PS_z': 1.322,
        'PS_y': 1.087
        }

    extinction = ext/ext_deltamag_dict[ext_band]*ext_deltamag_dict[flux_band]

    return flux / np.power(10,-0.4*extinction)

def deredden_mag(mag, mag_band, ext, ext_band):
    ext_deltamag_dict = \
        {'EBV': 1.0,
	 'A_V':3.1,
        'SDSS_u':4.239,
        'SDSS_g':3.303,
        'SDSS_r':2.285,
        'SDSS_i':1.698,
        'SDSS_z':1.263,
        'TMASS_j':0.709,
        'TMASS_h':0.449,
        'TMASS_k':0.302,
        'WISE_w1':0.189,
        'WISE_w2':0.146,
        'WISE_w3':0.0,
        'WISE_w4':0.0,
        'UNWISE_w1':0.189,
        'UNWISE_w2':0.146,
        'PS_g': 3.172 ,
        'PS_r': 2.271,
        'PS_i': 1.682,
        'PS_z': 1.322,
        'PS_y': 1.087
        }

    extinction = ext/ext_deltamag_dict[ext_band]*ext_deltamag_dict[mag_band]

    return mag - extinction
    # Schlafly&Finkbeiner 2011, Fitzpatrick 1999, IRSA
    # http://irsa.ipac.caltech.edu/applications/DUST




def FLUXtoABMAG(flux):
    return  -5./2.*np.log10(flux/3631.) #flux in Jy

def ABMAGtoFLUX(ab_mag):
    return np.power(10,-0.4*ab_mag)*3631. #Jy

def ABERRtoFLUXERR(ab_mag,ab_magerr):
    return abs(-0.4*np.log(10) * ab_magerr * np.power(10,-0.4*ab_mag) * 3631)

def ASINHMAGERRtoFLUXERR(mag,magerr,band):

    if band == 'SDSS_u':
        b = 1.4e-10
    elif band == 'SDSS_g':
        b = 0.9e-10
    elif band == 'SDSS_r':
        b = 1.2e-10
    elif band == 'SDSS_i':
        b = 1.8e-10
    elif band == 'SDSS_z':
        b = 7.4e-10
    else :
        print('Conversion from ASINHMAG to FLUX unsuccessful \n FILTER BAND NOT RECOGNIZED')
        return -1

    return abs(2*b*np.cosh( (-mag) / (2.5/np.log(10)) - np.log(b)) * ((-magerr) / (2.5/np.log(10)))*3631)

def ASINHMAGtoFLUX(mag,band):

    if band == 'SDSS_u':
        b = 1.4e-10
    elif band == 'SDSS_g':
        b = 0.9e-10
    elif band == 'SDSS_r':
        b = 1.2e-10
    elif band == 'SDSS_i':
        b = 1.8e-10
    elif band == 'SDSS_z':
        b = 7.4e-10
    else :
        print('Conversion from ASINHMAG to FLUX unsuccessful \n FILTER BAND NOT RECOGNIZED')
        return -1

    f_f_0 = np.sinh( (-mag) / (2.5/np.log(10)) - np.log(b))*2.*b

    return 3631 *f_f_0

#S = 3631 Jy * f/f0
# mag = -(2.5/ln(10))*[asinh((f/f0)/2b)+ln(b)]
#u 1.4 * 10-10	24.63	22.12
#g 0.9 * 10-10	25.11	22.60
#r 1.2 * 10-10	24.80	22.29
#i 1.8 * 10-10	24.36	21.85
#z 7.4 * 10-10	22.83	20.32
#error(mag) = 2.5 / ln(10) * error(counts)/exptime * 1/2b * 100.4*(aa + kk * airmass) / sqrt(1 + [(f/f0)/2b]2)
#http://classic.sdss.org/dr7/algorithms/fluxcal.html#asinh_table
#http://classic.sdss.org/dr7/algorithms/edr.tb21.html
