import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

fedavg = pd.read_csv("./results/fedavg_acc_total.csv")
pi = pd.read_csv("./results/pi.csv")
convex = pd.read_csv("./results/convex.csv")
fedavg_ = pd.read_csv("./results/fedavg_client_info.csv")
pi_ = pd.read_csv("./results/pi_accuracy.csv")
convex_ = pd.read_csv("./results/convex_accuracy.csv")
fedavg200 = pd.read_csv("./results/200.csv")
pi200 = pd.read_csv("./results/200pq.csv")
convex200 = pd.read_csv("./results/convex200.csv")
fedavg200_ = pd.read_csv("./results/200_accuracy.csv")
pi200_ = pd.read_csv("./results/200pq_accuracy.csv")
convex200_ = pd.read_csv("./results/convex200_accuracy.csv")

times = []
for method in [fedavg_, pi_, convex_, fedavg200_, pi200_, convex200_]:
    time_ = [0]
    for round_idx in range(min(method["round"]), max(method["round"]) + 1):
        rts = []
        for idx, rt in enumerate(method["round_time"]):
            if (method["round"][idx]) == round_idx:
                rts.append(rt)
        time_.append(time_[-1] + max(rts))
    times.append(time_)

print(
    times[0][-1], times[1][-1], times[2][-1], times[3][-1], times[4][-1], times[5][-1]
)

# plt.plot(
#     times[0],
#     [0] + np.array(fedavg["accuracy"]).tolist(),
#     label="FedAvg Client 100",
#     linewidth=1.5,
# )
# plt.plot(
#     times[1],
#     [0] + np.array(pi["accuracy"]).tolist(),
#     label="p=q, Client 100",
#     linewidth=1.5,
# )
# plt.plot(
#     times[2],
#     [0] + np.array(convex["accuracy"]).tolist(),
#     label="Ours, Client 100",
#     linewidth=1.5,
# )
plt.plot(
    times[3],
    [0] + np.array(fedavg200["accuracy"]).tolist(),
    label="FedAvg Client 200",
    linewidth=1.5,
)
plt.plot(
    times[4],
    [0] + np.array(pi200["accuracy"]).tolist(),
    label="p=q, Client 200",
    linewidth=1.5,
)
plt.plot(
    times[5],
    [0] + np.array(convex200["accuracy"]).tolist(),
    label="Ours, Client 200",
    linewidth=1.5,
)

font = {
    "family": "Times New Roman",
    "weight": "bold",
    "size": 16,
}

plt.ylabel("Accuracy", fontdict=font)
plt.xlabel("Wall Clock Time (s)", fontdict=font)
plt.legend(prop={"weight": "bold", "size": 16, "family": "Times New Roman"})
plt.savefig("./training_200.png")
