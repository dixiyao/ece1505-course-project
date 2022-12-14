import numpy as np
import pandas as pd

p = np.array(pd.read_csv("./results/fedavg_200_p.csv")["p"])
G = np.array(pd.read_csv("./results/fedavg_200_G.csv")["G"])

sump2G2 = np.sum(p * p * G * G)
sumpG2 = np.sum(p * G * G)
print(sump2G2, sumpG2)

R1 = 168
R2 = 102

result = (R1 / R2 - 1) / (100 * sump2G2 - R1 / R2 * sumpG2)
print(result)
