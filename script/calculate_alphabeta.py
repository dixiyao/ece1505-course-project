import numpy as np
import pandas as pd

p = np.array(pd.read_csv("./results/fedavg_p.csv")["p"])
G = np.array(pd.read_csv("./results/fedavg_G.csv")["G"])

sump2G2 = np.sum(p * p * G * G)
sumpG2 = np.sum(p * G * G)
print(sump2G2, sumpG2)

R1 = 210
R2 = 134

result = (R1 / R2 - 1) / (100 * sump2G2 - R1 / R2 * sumpG2)
print(result)
