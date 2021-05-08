import pandas as pd

from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u
#from astroquery.ned import Ned #necessary for one method
#from astroquery.ned.core import RemoteServiceError #necessary for one method
#from dustmaps.sfd import SFDQuery
#not working under windows, will replace it with sfdmap
import sfdmap

import photometric_functions as phot

import numpy as np




def get_extinction_values(catalog, ra, dec):


    coords = SkyCoord(catalog[ra].tolist()*u.deg, catalog[dec].tolist()*u.deg)

    #sfd = SFDQuery()

    #ebv = sfd(coords)

    m = sfdmap.SFDMap('data/dustmaps/')

    ebv = m.ebv(coords)

    catalog['EBV'] = ebv


    return catalog




def build_PS_flux_model_catalog_from_cat(catalog, use_stackmag=False):

    # -------------------------------------------------------------------------
    # Building the flux catalog DataFrame from the catalog
    # -------------------------------------------------------------------------

    flux_catalog = catalog.copy(deep=True)

    #for n in flux_catalog.columns:
    #    print (n)

    # --------------------------------------------------------------------------
    #  Specify magnitudes from the quasar catalog to save in the new
    #  flux catalog
    # --------------------------------------------------------------------------

    # magnitudes in the PanSTARRS bands grizy in normal AB magnitudes
    ps_mag_names = ['gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag',
                    'zMeanPSFMag', 'yMeanPSFMag']

    if(use_stackmag):
        ps_mag_names = ['gPSFStackMag','rPSFStackMag','iPSFStackMag',
                        'zPSFStackMag','yPSFStackMag']
    # errors on those AB magnitudes
    ps_mag_err_names = ['gMeanPSFMagErr', 'rMeanPSFMagErr', 'iMeanPSFMagErr',
                        'zMeanPSFMagErr', 'yMeanPSFMagErr']

    ps_new_mag_names = ['PS_mag_g', 'PS_mag_r', 'PS_mag_i', 'PS_mag_z', 'PS_mag_y']

    # these are the column names for the PanSTAARS fluxes in the output
    # flux catalog
    ps_bandpass_names = ['PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y']


    extinction_name = ['EBV']
    # Galactic extinction values in magnitudes for the SDSS bandpasses

    vega_mag_names = [#'j_m_2mass', 'h_m_2mass', 'k_m_2mass',
                      'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro']
    # magnitudes of other survey bandpasses in VEGA magnitudes
    vega_mag_err_names = [#'j_msig_2mass', 'h_msig_2mass', 'k_msig_2mass',
                          'w1sigmpro', 'w2sigmpro', 'w3sigmpro', 'w4sigmpro']
    # 1-sigma error on magnitudes of other survey bandpasses in VEGA magnitudes

    # These are the column names for the other magnitudes fluxes in the output
    # flux catalog
    # These names have to be in the corresponding order to the vega_mag_names
    # above
    vega_bandpass_names = [#'TMASS_j','TMASS_h','TMASS_k',
                            'WISE_w1','WISE_w2','WISE_w3','WISE_w4']

    vega_new_mag_names = [#'TMASS_mag_j','TMASS_mag_h','TMASS_mag_k',
                            'WISE_mag_w1','WISE_mag_w2','WISE_mag_w3','WISE_mag_w4']

    # -------------------------------------------------------------------------
    # Convert all "empty" columns to np.NaNs
    # -------------------------------------------------------------------------

    for i in range(len(ps_bandpass_names)):

        # replace values that are 0 with np.NaN
        #flux_catalog['sigma_'+ps_bandpass_names[i]] = catalog[ps_mag_err_names[i]].replace(-999,np.NaN)

        flux_catalog[ps_bandpass_names[i]] = catalog[ps_mag_names[i]].replace(-999,np.NaN)

    for i in range(len(vega_bandpass_names)):

        # replace values that are 0 with np.NaN
        flux_catalog['sigma_'+vega_bandpass_names[i]] = catalog[vega_mag_err_names[i]].replace(0.0,np.NaN)

        flux_catalog[vega_bandpass_names[i]] = catalog[vega_mag_names[i]].replace(0.0,np.NaN)

    # -------------------------------------------------------------------------
    # Convert PS magnitudes using the correct AB magnitude zero point and
    # deredden them for the flux catalog
    # -------------------------------------------------------------------------

    # Conversion from AB magnitudes to Jansky and correction to correct zero point flux
    for i in range(len(ps_bandpass_names)):
        name = ps_bandpass_names[i]
        new_mag_name = ps_new_mag_names[i]

        # convert VEGA to AB magnitudes
        flux_catalog[name] = phot.VEGAtoAB(flux_catalog[name], name)
        # apply correct dereddening
        flux_catalog[name] = phot.deredden_mag(flux_catalog[name],
                                name, catalog.EBV, 'EBV')

        flux_catalog[new_mag_name] = flux_catalog[name]

        flux_catalog[name] = phot.ABMAGtoFLUX(flux_catalog[name])

    # ------------------------------------------------------------------------
    # Conversion of the Vega magnitudes from the other survey bandpasses to
    # fluxes, AB correction and reddening is applied before
    # ------------------------------------------------------------------------

    for i in range(len(vega_bandpass_names)):

        # convert VEGA to AB magnitudes
        flux_catalog[vega_bandpass_names[i]] = \
                    phot.VEGAtoAB(flux_catalog[vega_bandpass_names[i]],
                    vega_bandpass_names[i])

        # apply the correct dereddening
        flux_catalog[vega_bandpass_names[i]] = \
                    phot.deredden_mag(flux_catalog[vega_bandpass_names[i]],
                    vega_bandpass_names[i], catalog.EBV, 'EBV')

        flux_catalog[vega_new_mag_names[i]] = \
                                        flux_catalog[vega_bandpass_names[i]]

        # convert exinction corrected AB magnitudes to fluxes in Jy
        flux_catalog[vega_bandpass_names[i]] = \
                    phot.ABMAGtoFLUX(flux_catalog[vega_bandpass_names[i]])

    #print (flux_catalog.columns)


    return flux_catalog


def prepare_catalog(df, use_stackmag=False):

    #  1) Drop Duplicates on wise_designation
    ## df.drop_duplicates('wise_designation', keep='first', inplace=True)

    #  2) Get Schlafly, Finkbeiner, Davis extinction values
    ra='ps_ra'
    dec='ps_dec'
    df = get_extinction_values(df, ra,dec)
    # print df.EBV

    #  2) Calculate magnitudes + fluxes and add them to the catalog
    df = build_PS_flux_model_catalog_from_cat(df, use_stackmag)

    return df


def prepare_sim_catalog(df):
    #Here we already have the correct AB magnetudes, i just convert them to flux_ratio_err_names

    passbands = ['PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y', 'WISE_w1','WISE_w2']
    passbands_mag = ['PS_mag_g', 'PS_mag_r', 'PS_mag_i', 'PS_mag_z', 'PS_mag_y', 'WISE_mag_w1','WISE_mag_w2']

    for i in range(len(passbands)):
        df[passbands[i]] = phot.ABMAGtoFLUX(df[passbands_mag[i]])

    return df
