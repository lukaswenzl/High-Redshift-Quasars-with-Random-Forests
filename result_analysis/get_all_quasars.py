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

print("checking other area restrictions (otherwise my total area estimate isn't quite right): below -30dec:")
check = df.query("ps_dec < -30")
print(check["pred_class"].value_counts())

print("now we are just keeping all objects that could be quasars.")
quasars = df.query("qso_prob > 0.15 and regression_full> 1.5")

quasars.to_csv("../data/results/full14_all_qso.csv")
