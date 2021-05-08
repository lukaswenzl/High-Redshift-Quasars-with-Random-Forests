import numpy as np
import pandas as pd

data = pd.read_csv("../data/results/full14.csv")
mstars = data.query("pred_class == 'M'")

mstars = mstars.sample(n=1000000)
mstars.to_csv("../data/analysis/mstars_1mio.csv")
