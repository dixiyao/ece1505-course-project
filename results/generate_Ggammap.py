import pandas as pd
import numpy as np

file = pd.read_csv("200_accuracy.csv")

Gs = np.zeros(200)
gammas = {}
ps = np.zeros(200)

for index, G in enumerate(file["gbound"]):
    Gs[file["client_id"][index] - 1] = max(Gs[file["client_id"][index] - 1], G)
    ps[file["client_id"][index] - 1] = file["samples"][index]
    if file["client_id"][index] not in gammas:
        gammas[file["client_id"][index]] = [file["round_time"][index]]
    else:
        gammas[file["client_id"][index]].append(file["round_time"][index])

G_file = open("fedavg_200_G.csv", "w")
G_file.write("id,G\n")
p_file = open("fedavg_200_p.csv", "w")
p_file.write("id,p\n")
gamma_file = open("fedavg_200_gamma.csv", "w")
gamma_file.write("id,gamma\n")

ps = ps / np.sum(ps)

for i in range(200):
    if Gs[i] == 0:
        Gs[i] = Gs[168]
    if ps[i] == 0:
        ps[i] = ps[168]

for i in range(200):
    G_file.write(str(i + 1) + "," + str(Gs[i]) + "\n")
    p_file.write(str(i + 1) + "," + str(ps[i]) + "\n")
    try:
        gamma = gammas[i + 1]
    except:
        gamma = gammas[169]
    gamma = np.mean(np.array(gamma))
    gamma_file.write(str(i + 1) + "," + str(gamma) + "\n")
