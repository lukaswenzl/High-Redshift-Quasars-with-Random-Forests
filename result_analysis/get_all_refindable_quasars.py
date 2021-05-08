import numpy as np
import pandas as pd
from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u

df = pd.read_csv("../data/results/full14.csv")

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



print("now we are just keeping all objects that are known quasars. ie these objects could be found by the algorithm")
controll_quasars = pd.read_csv("../data/training/quasars_dr14_and_highz.csv")
quasars = controll_quasars[controll_quasars["wise_designation"].isin(df["wise_designation"])]

# print("we also add a flag indicating if it is in the crossvalidation region or not")
# training_region = (quasars["ps_ra"]> 60) & (quasars["ps_ra"] < 300) & ((quasars["ps_dec"]> 1.26) | (quasars["ps_dec"]< -1.26)) ##!!! WRONG
# quasars["crossval"] = True
# quasars.loc[training_region, "crossval"] = training_region == False

# print("Quasars in training  region: {}, Quasars in crossval region: {}".format((quasars.query("crossval == False")).shape[0], (quasars.query("crossval == True")).shape[0]))

quasars.to_csv("../data/training/quasars_in_catalog_data.csv")

###################
controll_stars = pd.read_csv("../data/training/stars_sdss_and_dwarfs.csv")
stars = controll_stars[controll_stars["wise_designation"].isin(df["wise_designation"])]

# print("we also add a flag indicating if it is in the crossvalidation region or not")
# training_region = (stars["ps_ra"]> 60) & (stars["ps_ra"] < 300) & ((stars["ps_dec"]> 1.26) | (stars["ps_dec"]< -1.26)) ####!!!!!! WRONG
# stars["crossval"] = True
# stars.loc[training_region, "crossval"] = stars == False
#
# print("Stars in training  region: {}, Stars in crossval region: {}".format((stars.query("crossval == False")).shape[0], (stars.query("crossval == True")).shape[0]))

stars.to_csv("../data/training/stars_in_catalog_data.csv")
