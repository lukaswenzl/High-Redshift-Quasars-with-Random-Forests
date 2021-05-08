#
#
#Author Lukas Wenzl
#written in python 3

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.externals import joblib ##load pkl (pickle) file

from astropy.io import fits

from datetime import datetime

import sys
sys.path.append("training_data_preprocessing/")

# Convert vega mags in ab mags and calculate the fluxes
import catalog_processing_LW as process

# Prepare the quasar sample by calculating flux ratios
import ml_quasar_sample as qs


# Load the random forest classification functions
import random_forest as rf

#for parsing the arguments for the file
import argparse

#to standardize scaling
from sklearn.preprocessing import RobustScaler

from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u

""" examples: python 3_apply_models.py full14 full14_v21_test -t 1
python 3_apply_models.py full14 full14_v21_highz
python 3_apply_models.py full14 full14_v21_full -w 1

"""
def loading_trained_models(prefix=""):
    """Load the models into memory.

    Parameters
    ----------
    Prefix : string
        Prefix used for crossvaldiation runs.

    Returns
    -------
    directory
        Contains all the loaded models.

    """
    print("prefix used: {}".format(prefix))
    print('loading classifier from pickle file (standard)')
    clf = joblib.load('data/'+prefix+'rf_classifier_PS_ALLWISE.pkl')

    # print('loading classifier from pickle file (with the non Quasars, less accurate but more realistic)')
    # clf2 = joblib.load('data/'+prefix+'rf_classifier_with_Not_quasars_PS_ALLWISE.pkl')

    # print('loading classifier from pickle file for simulated quasars and all star data')
    # clf3 = joblib.load('data/'+prefix+'rf_classifier_sim_QSO_PS_ALLWISE.pkl')

    print('loading regressors from pickle files')
    reg_full = joblib.load('data/'+prefix+'rf_regressor_full_PS_ALLWISE.pkl')
    reg_highz = joblib.load('data/'+prefix+'rf_regressor_highz_PS_ALLWISE.pkl')
    #reg_sim_full = joblib.load('data/'+prefix+'rf_regressor_sim_full_PS_ALLWISE.pkl')
    #reg_sim = joblib.load('data/'+prefix+'rf_regressor_sim_PS_ALLWISE.pkl')


    scaler_full = joblib.load('data/'+prefix+'rf_regressor_full_PS_ALLWISE_scaler.pkl')
    scaler_highz = joblib.load('data/'+prefix+'rf_regressor_highz_PS_ALLWISE_scaler.pkl')
    #scaler_sim_full = joblib.load('data/'+prefix+'rf_regressor_sim_full_PS_ALLWISE_scaler.pkl')
    #scaler_sim = joblib.load('data/'+prefix+'rf_regressor_sim_PS_ALLWISE_scaler.pkl')

    models = {"clas_standart":clf, #"clas_realistic":clf2, "clas_sim":clf3,
              "reg_full":reg_full, "reg_highz":reg_highz, #"reg_sim_full":reg_sim_full, "reg_sim":reg_sim,
              "scaler_full":scaler_full, "scaler_highz":scaler_highz
              #,"scaler_sim_full":scaler_sim_full, "scaler_sim":scaler_sim
              }
    return models




def classify(short_filename, output_filename, write, test):
    """Classify and regress the file given.

    File has to be in for_classification
    folder and will be copied to the classified folder

    Parameters
    ----------
    df : dataframe
        The data to classify and regress
    pathtodata : string
        Path to the data. Only for files to classify, not for the classifiers
        and regressors.
    write : int
        int wether to write the full result to disk (1) or Only
        the best candidates (0)

    """
    filename = "data/catalog/"+short_filename+'.fits'

    print("reading the file "+short_filename+".fits for classification")
    hdu = fits.open(filename)
    if(test==1):
        df = pd.DataFrame(hdu[1].data[0:10000])
    else:
        df = pd.DataFrame(hdu[1].data)

    #print('make column for extension of object')
    #
    #df['psf_ap'] = df['zMeanPSFMag'] - df['zMeanApMag']  #here i still have to use the mean mags
    #df['psf_kron'] = df['zMeanPSFMag'] - df['zMeanKronMag']

    #print('preparing catalog (calc fluxes) using Stacked magnitudes')
    df = process.prepare_catalog(df, use_stackmag=True) ##using stack mags now


    #print('replacing missing values in g, r and i band since highz quasars will be dark there')

    df["PS_g"].fillna(1e-10, inplace=True)
    df["PS_r"].fillna(1e-10, inplace=True)
    df["PS_i"].fillna(1e-10, inplace=True)

    #print('preparing flux ratios')
    passband_names = [ 'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y',\
                    'WISE_w1', \
                    'WISE_w2']
    df, features =  qs.prepare_flux_ratio_catalog(df, passband_names)


    print('classifying')
    features = ['PS_z','WISE_w1','gr','ri','iz','zy', 'yw1','w1w2']
    prefix = ""
    df = rf.rf_class_predict(models["clas_standart"], df, features, prefix)
    #----------------------------------------------------------------------------------------------------
    #same with the non quasars i got from jinyi
    #print('classifying with non quasars')
    #add prefix for the second classifier
    # prefix = "realistic_"
    # df = rf.rf_class_predict(models["clas_realistic"], df, features, prefix)
    #----------------------------------------------------------------------------------------------------
    #same with the simulated quasars
    #print('classifying simulated quasars')
    #add prefix for the second classifier
    # prefix = "sim_"
    # df = rf.rf_class_predict(models["clas_sim"], df, features, prefix)

    #----------------------------------------------------------------------------------------------------

    print('finished classifying for '+short_filename)

    print(df["pred_class"].value_counts())

    #-------------------------------------------------------------------------------------
    # do regression on most promissing candidates
    #------------------------------------------------------------------------------------
    if(write==0):
        print("only keeping pred_class == 'highz' or highz>0.15 candidates")
        df = df.query("pred_class == 'highz' or highz>0.15")# or sim_highz > 0.25")

    if(len(df.index) > 0):

        features = ['PS_z','WISE_w1','gr','ri','iz','zy', 'yw1','w1w2']


        df = rf.rf_reg_predict(models["reg_full"], models["scaler_full"], df.copy(), features, "regression_full")
        df = rf.rf_reg_predict(models["reg_highz"], models["scaler_highz"], df.copy(), features, "regression_highz")
        #df = rf.rf_reg_predict(models["reg_sim_full"], models["scaler_sim_full"], df, features, "regression_sim_full")
        #df = rf.rf_reg_predict(models["reg_sim"], models["scaler_sim"], df, features, "regression_sim")


        #print('here are the objects classified highz')
        #print(df[df.pred_class == 'highz'][['PS1_ObjID','ps_ra','ps_dec','psf_ap', 'psf_kron','PS_g','K','M','highz','qso_prob','pred_class', 'realistic_pred_class','sim_pred_class','regression_full','regression_highz', 'regression_sim', 'regression_sim_full']])

        if(write==1):
            df = df[["ps_ra", "ps_dec",  "gPSFStackMag", "rPSFStackMag", "iPSFStackMag", "zPSFStackMag", "yPSFStackMag",
            'g_sn', 'r_sn', 'i_sn', 'z_sn', 'y_sn', "wise_designation", 'coadd_id',
            'w1mpro', 'w1sigmpro', 'w2mpro', 'w2sigmpro', "EBV",
            'PS_mag_g', 'PS_mag_r', 'PS_mag_i', 'PS_mag_z', 'PS_mag_y', 'WISE_mag_w1', 'WISE_mag_w2', 'WISE_mag_w3', 'WISE_mag_w4',
            'A', 'F', 'G', 'K', 'L', 'M', 'T', 'highz', 'lowz', 'midz', 'vlowz', 'qso_prob', 'pred_class', 'regression_full', 'regression_highz']]
        print("saving the candidates for "+short_filename+" as "+output_filename+".csv")
        output_filename1 = 'data/results/'+output_filename+'.csv'
        df.to_csv(output_filename1)
        #df.to_hdf(output_filename1, key='data', mode='w') can't be read by vaex!! (but is a lot smaller...)

        # | df.highz>0.15


    print("number of objects classified highz: "+str(len(df[df.pred_class == 'highz'].index)))
    print("number of objects just saved to disk: "+str(len(df.index)))
    print("value_counts of pred_class")
    print(df["pred_class"].value_counts())

    positions = SkyCoord(ra=df["ps_ra"].values*u.degree, dec=df["ps_dec"].values*u.degree, frame='icrs')
    andromeda = SkyCoord('0h42m44s', '+41d16m9s', frame='icrs')
    galactic_center = SkyCoord('0h0m0s', '+0d0m0s', frame='galactic')
    sep_andromeda = positions.separation(andromeda).deg
    sep_galactic_center = positions.separation(galactic_center).deg
    area_res = (sep_andromeda>5) & (sep_galactic_center > 30)
    df = df[area_res]
    df = df.query("EBV < 0.1")
    print("sep andromeda 5deg, sep gal center 30deg, dust < 0.1. Then how many remain:")
    print(df.shape)
    print(df["pred_class"].value_counts())

    print("checking other area restrictions (otherwise my total area estimate isn't quite right): below -30dec:")
    check = df.query("ps_dec < -30")
    print(check["pred_class"].value_counts())

    return len(df[df.pred_class == 'highz'].index)



def parseArguments():
    """Parses the given Arguments when calling the file from the command line.

    Returns
    -------
    arg
        The result from parsing.

    """
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("inputname", help="name for the input file", type=str)
    parser.add_argument("outputname", help="name for the output file", type=str)


    # Optional arguments

    parser.add_argument("-t", "--test", help="Optionally test algorithm on subsample of the data. Set to 1 for testing", type=int, default=0)


    parser.add_argument("-w", "--write", help="Should all data be written. Default: 0 (only good candidates, for all set to 1) ", type=int, default=0)

    parser.add_argument("--prefix", help="Prefix for all files for crossvaldiation. Leave empty if runnin on whole data", type=str, default="")



    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 2.0') #
    #changelog
    #version 1.2 now needs highz_sim >25% to keep it. In an effort to make the resulting file smaller
    #version 1.3_server optimized for servers with a lot of ram, using stack mag now
    #version 1.4_server now using only one file for the data
    #version 1.5_server now highly parallel instead
    #version 1.6 added support for prefix to be able to run crossvaldiation on subset
    #version 2.0 rewrote program for signle large data file
    #version 2.1 made console output more informative

    # Parse arguments
    args = parser.parse_args()

    return args




if __name__ == '__main__':
    #filename without the _PS_WISE extension!
    args = parseArguments()
    print("program version: 2.1")
    StartTime = datetime.now()


    prefix_filenames=""#for now
    models = loading_trained_models(prefix_filenames)

    result = classify(args.inputname, args.outputname, args.write, args.test)
    #statistical analyis.....
    print("finished at ")
    print(datetime.now())
    print("overall time taken")
    print(datetime.now()-StartTime)
