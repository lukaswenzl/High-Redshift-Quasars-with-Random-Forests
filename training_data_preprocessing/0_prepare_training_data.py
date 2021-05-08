
import pandas as pd
import numpy as np

import ml_quasar_sample as qs
import catalog_processing_LW as process

print("best run: python prepare_training_data.py > ../data/training/training_data.log")
print("ran version 1.0")

datapath = "../data/"
print("loading catalogs")
print("replacing null in g, r and i band with 1e-10 flux")
df_stars = pd.read_csv(datapath+'raw_training/JT_stars_PS_WISE_stackflux.csv')


df_stars["PS_g"].fillna(1e-10, inplace=True)
df_stars["PS_r"].fillna(1e-10, inplace=True)
df_stars["PS_i"].fillna(1e-10, inplace=True)

df_quasars = pd.read_csv(datapath+'raw_training/JT_quasars_PS_WISE_stackflux.csv')

df_quasars["PS_g"].fillna(1e-10, inplace=True)
df_quasars["PS_r"].fillna(1e-10, inplace=True)
df_quasars["PS_i"].fillna(1e-10, inplace=True)
df_quasars['z'] = df_quasars['Z_VI']

df_browndwarfs = pd.read_csv(datapath+'raw_training/brown_dwarfs_PS_WISE_stackflux.csv')

#increase data by assuming that missing g and r band is because of dropout, so i put in limiting value +x?
df_browndwarfs["PS_g"].fillna(1e-10, inplace=True)
df_browndwarfs["PS_r"].fillna(1e-10, inplace=True)
df_browndwarfs["PS_i"].fillna(1e-10, inplace=True)

df_highzquasars = pd.read_csv(datapath+'raw_training/highz_qso_PS_WISE_stackflux.csv')

df_highzquasars["PS_g"].fillna(1e-10, inplace=True)
df_highzquasars["PS_r"].fillna(1e-10, inplace=True)
df_highzquasars["PS_i"].fillna(1e-10, inplace=True)

# df_not_quasars = pd.read_csv('../class_photoz/data/Not_QSOs_crossmatched2_PS_WISE.csv')
df_not_quasars = pd.read_csv(datapath+'raw_training/not_quasars_PS_WISE_stackflux.csv')

df_not_quasars["PS_g"].fillna(1e-10, inplace=True)
df_not_quasars["PS_r"].fillna(1e-10, inplace=True)
df_not_quasars["PS_i"].fillna(1e-10, inplace=True)


passband_names = [\
        #'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
        'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y',\
        # 'TMASS_j','TMASS_h','TMASS_k', \
        'WISE_w1','WISE_w2', \
        # 'WISE_w3' \
        ]

#df_sim = pd.read_hdf('../class_photoz/data/simqsos_z7_90000sqdeg.hdf5', 'data')
# df_sim = pd.read_hdf('../class_photoz/data/simqsos_3to7_10000sqdeg_dec_2018.hdf5', 'data')
# df_sim = process.prepare_sim_catalog(df_sim)
#
# df_sim = process.prepare_sim_catalog(df_sim)
#
# #df_sim_low = pd.read_hdf('../class_photoz/data/simqsos_lessz4_1000sqdeg.hdf5', 'data')
# df_sim_low = pd.read_hdf('../class_photoz/data/simqsos_0to3_500sqdeg_dec2018.hdf5', 'data')
# df_sim_low = process.prepare_sim_catalog(df_sim_low)


# df_sim_low = process.prepare_sim_catalog(df_sim_low)
# for name in passband_names:
#     #replace dark measuremnets with the correct flux i use
#     df_sim_low.loc[df_sim_low[name] <1e-10, name] = 1e-10
#     df_sim.loc[df_sim[name] <1e-10, name] = 1e-10 ##added 22. May 2018



print ("Stars input: ", df_stars.shape)
print ("Quasars input: ", df_quasars.shape)
print ("Browndwarfs input: ", df_browndwarfs.shape)
print ("High z Quasars input: ", df_highzquasars.shape)
print ("Not Quasars (previously missclassified) input: ", df_not_quasars.shape)
# print ("Sim Quasars low z input: ", df_sim_low.shape)
# print ("Sim Quasars high z input: ", df_sim.shape)

#to test ###########################################
# if (testmode==1):
#     df_stars = df_stars.sample(n=10000)
#     df_quasars = df_quasars.sample(n=10000)
#     df_sim_low = df_sim_low.sample(n=10000)
#     df_sim = df_sim.sample(n=3000)



passband_names = [ #'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z', \
                   # 'SDSS_u', \
                    'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y',\
                    # 'TMASS_j', \
                    # 'TMASS_h', \
                    # 'TMASS_k', \
                    'WISE_w1', \
                    'WISE_w2', \
                    # 'WISE_w3', \
                    # 'WISE_w4', \
                    ]

df_stars, features = \
    qs.prepare_flux_ratio_catalog(df_stars, passband_names)
df_quasars, features = \
    qs.prepare_flux_ratio_catalog(df_quasars, passband_names)
df_not_quasars, features = \
    qs.prepare_flux_ratio_catalog(df_not_quasars, passband_names)

df_browndwarfs, features = \
    qs.prepare_flux_ratio_catalog(df_browndwarfs, passband_names)
df_highzquasars, features = \
    qs.prepare_flux_ratio_catalog(df_highzquasars, passband_names)
# df_sim_low, features = \
#     qs.prepare_flux_ratio_catalog(df_sim_low, passband_names)
# df_sim, features = \
#     qs.prepare_flux_ratio_catalog(df_sim, passband_names)

print("numbers after calculating flux ratios and droping nan's")
print ("Stars: ", df_stars.shape)
print ("Quasars: ", df_quasars.shape)
print ("Browndwarfs: ", df_browndwarfs.shape)
print ("High z Quasars: ", df_highzquasars.shape)
print ("Not Quasars (previously missclassified): ", df_not_quasars.shape)
# print ("Sim Quasars low z: ", df_sim_low.shape)
# print ("Sim Quasars high z: ", df_sim.shape)


# Create detailed classes
df_quasars = qs.create_qso_labels(df_quasars, 'mult_class_true', 'Z_VI')
df_highzquasars = qs.create_qso_labels(df_highzquasars, 'mult_class_true', 'z')
df_highzquasars['Z_VI'] = df_highzquasars['z']

df_stars = qs.create_star_labels(df_stars, 'mult_class_true', 'star_class')
df_browndwarfs = qs.create_star_labels(df_browndwarfs, 'mult_class_true', 'star_class')

# df_sim = qs.create_qso_labels(df_sim, 'mult_class_true', 'z')
# df_sim_low = qs.create_qso_labels(df_sim_low, 'mult_class_true', 'z')


df_not_quasars['mult_class_true'] = 'M'
df_not_quasars['bin_class_true'] = 'STAR'

# Create binary classes
df_quasars['bin_class_true'] = 'QSO'
df_highzquasars['bin_class_true'] = 'QSO'

df_stars['bin_class_true'] = 'STAR'
df_browndwarfs['bin_class_true'] = 'STAR'

# df_sim['bin_class_true'] = 'QSO'
# df_sim_low['bin_class_true'] = 'QSO'

#record sources
df_not_quasars["source"]="not_quasars"
df_quasars["source"] = "sdss_dr14"
df_highzquasars["source"] = "highz"
df_browndwarfs["source"] = "dwarfs"
df_stars ["source"] = "sdss_stars"

print("catalogs loaded and prepared")
print("now combining them (first without non Quasars)")

#combine quasar dataframes
frames1 = [df_quasars, df_highzquasars]

df_all_quasars = pd.concat(frames1, sort=False)

before = df_all_quasars.shape[0]
df_all_quasars = df_all_quasars.drop_duplicates(subset="wise_designation", keep='first')
print("number of duplicate quasars removed: {} (can be quite different from 0 here)".format(df_all_quasars.shape[0]-before))

#combine star dataframes
frames2 = [df_stars, df_browndwarfs]#, df_not_quasars]

df_all_stars = pd.concat(frames2, sort=False)

#not quasars are ignored at the moment

# yes = {'yes','y', 'ye', ''}
# no = {'no','n'}
#
# print("Should the data be written to disk? [y/n]")
# choice = input().lower()
# if choice in yes:
df_all_quasars.to_csv("../data/training/quasars_dr14_and_highz.csv")
df_all_stars.to_csv("../data/training/stars_sdss_and_dwarfs.csv")
# else:
#    print("Please respond with 'yes' or 'no'")
print("finished")
###sample code for sim quasars
# print("replacing null in all bands with 1e-10 flux")
# #df_sim = pd.read_hdf('../class_photoz/data/simqsos_z7_90000sqdeg.hdf5', 'data')
# df_sim = pd.read_hdf('../class_photoz/data/simqsos_3to7_10000sqdeg_dec_2018.hdf5', 'data')
#
# df_sim = process.prepare_sim_catalog(df_sim)
# passband_names = [\
#         #'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
#         'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y',\
#         # 'TMASS_j','TMASS_h','TMASS_k', \
#         'WISE_w1','WISE_w2', \
#         # 'WISE_w3' \
#         ]
# for name in passband_names:
#     #replace dark measuremnets with the correct flux i use
#     df_sim.loc[df_sim[name] <1e-10, name] = 1e-10
#
#
# #df_sim_low = pd.read_hdf('../class_photoz/data/simqsos_lessz4_1000sqdeg.hdf5', 'data')
# df_sim_low = pd.read_hdf('../class_photoz/data/simqsos_0to3_500sqdeg_dec2018.hdf5', 'data')
# df_sim_low = process.prepare_sim_catalog(df_sim_low)
# for name in passband_names:
#     #replace dark measuremnets with the correct flux i use
#     df_sim_low.loc[df_sim_low[name] <1e-10, name] = 1e-10
#
# frames1 = [df_sim, df_sim_low]
# df_train = pd.concat(frames1)
#
# df_train.replace(np.inf, np.nan,inplace=True)
#
# print("calculating flux ratios")
# df_train, features = qs.prepare_flux_ratio_catalog(df_train,passband_names)
# df_sim, features = qs.prepare_flux_ratio_catalog(df_sim,passband_names)
