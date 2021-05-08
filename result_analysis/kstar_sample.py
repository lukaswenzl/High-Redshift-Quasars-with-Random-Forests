import numpy as np
import pandas as pd

data = pd.read_csv("../data/results/full14.csv")
kstars = data.query("pred_class == 'K'")

kstars = kstars.sample(n=1000000)
kstars.to_csv("../data/analysis/kstars_1mio.csv")
