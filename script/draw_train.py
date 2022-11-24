import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

fedavg = pd.read_csv("./results/fedavg_acc_total.csv")
pi = pd.read_csv("./results/pi.csv")
convex = pd.read_csv("./results/convex.csv")
fedavg_ = pd.read_csv("./results/fedavg_client_info.csv")
pi_ = pd.read_csv("./results/pi_accuracy.csv")
convex_ = pd.read_csv("./results/convex_accuracy.csv")

times = []
for method in [fedavg_, pi_, convex_]:
    time_ = [0]
    for round_idx in range(min(method["round"]), max(method["round"]) + 1):
        rts = []
        for idx, rt in enumerate(method["round_time"]):
            if (method["round"][idx]) == round_idx:
                rts.append(rt)
        time_.append(time_[-1] + max(rts))
    times.append(time_)

print(times[0][-1], times[1][-1], times[2][-1])

plt.plot(times[0], [0] + np.array(fedavg["accuracy"]).tolist(), label="FedAvg")
plt.plot(times[1], [0] + np.array(pi["accuracy"]).tolist(), label="p=q")
plt.plot(times[2], [0] + np.array(convex["accuracy"]).tolist(), label="Ours")

plt.ylabel("Accuracy")
plt.xlabel("Wall Clock Time (s)")
plt.legend()
plt.savefig("./training.png")
