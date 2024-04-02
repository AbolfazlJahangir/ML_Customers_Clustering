import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 

df = pd.read_csv("Customers.csv")

df["Annual Income (k$)"] = df["Annual Income (k$)"] * 1000

df = df.drop(columns=["CustomerID"])

columns = {"Gender": "Gender", "Age": "Age", "Annual Income (k$)": "Income", "Spending Score (1-100)": "Score"}

df = df.rename(columns=columns)

df["Gender"] = df["Gender"].replace({"Male": 1, "Female": 0})

#df.boxplot()
#df.hist()

scaler = StandardScaler()
x = df.values
x = scaler.fit_transform(x)

for i in range(4, 10):
    i = float(i / 10)
    for j in range(8, 26):
        print(i, j)
        model = DBSCAN(eps=i, min_samples=5)
        model.fit(x)
        labels = model.labels_

        df["CustomerType"] = labels

        core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        unique_labels = set(labels)

        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        x1 = df[df["CustomerType"] == 0]["Age"].values
        y1 = df[df["CustomerType"] == 0]["Score"].values
        z1 = df[df["CustomerType"] == 0]["Gender"].values

        x2 = df[df["CustomerType"] == 1]["Age"].values
        y2 = df[df["CustomerType"] == 1]["Score"].values
        z2 = df[df["CustomerType"] == 1]["Gender"].values

        x3 = df[df["CustomerType"] == 2]["Age"].values
        y3 = df[df["CustomerType"] == 2]["Score"].values
        z3 = df[df["CustomerType"] == 2]["Gender"].values

        x4 = df[df["CustomerType"] == 3]["Age"].values
        y4 = df[df["CustomerType"] == 3]["Score"].values
        z4 = df[df["CustomerType"] == 3]["Gender"].values

        x5 = df[df["CustomerType"] == 4]["Age"].values
        y5 = df[df["CustomerType"] == 4]["Score"].values
        z5 = df[df["CustomerType"] == 4]["Gender"].values
        
        x6 = df[df["CustomerType"] == 5]["Age"].values
        y6 = df[df["CustomerType"] == 5]["Score"].values
        z6 = df[df["CustomerType"] == 5]["Gender"].values
        
        x7 = df[df["CustomerType"] == 6]["Age"].values
        y7 = df[df["CustomerType"] == 6]["Score"].values
        z7 = df[df["CustomerType"] == 6]["Gender"].values
        
        x8 = df[df["CustomerType"] == 7]["Age"].values
        y8 = df[df["CustomerType"] == 7]["Score"].values
        z8 = df[df["CustomerType"] == 7]["Gender"].values

        scatter1 = ax.scatter(x1, y1, z1, c="red", alpha=0.65, label="Type1")
        scatter2 = ax.scatter(x2, y2, z2, c="orange", alpha=0.65, label="Type2")
        scatter3 = ax.scatter(x3, y3, z3, c="blue", alpha=0.65, label="Type3")
        scatter4 = ax.scatter(x4, y4, z4, c="green", alpha=0.65, label="Type4")
        scatter5 = ax.scatter(x5, y5, z5, c="purple", alpha=0.65, label="Type5")
        scatter6 = ax.scatter(x6, y6, z6, c="pink", alpha=0.65, label="Type6")
        scatter7 = ax.scatter(x7, y7, z7, c="black", alpha=0.65, label="Type7")
        scatter8 = ax.scatter(x8, y8, z8, c="yellow", alpha=0.65, label="Type8")

        
        ax.legend(handles=[scatter1, scatter2, scatter3, scatter4, scatter5, scatter6, scatter7, scatter8], loc="upper left", title="Costumers")

        ax.set_xlabel("Age", fontsize=15)
        ax.set_ylabel("Score", fontsize=15)
        ax.set_zlabel("Gender", fontsize=15)

        plt.show()


'''
    x1 = df[df["CustomerType"] == 0]["Age"].values
            y1 = df[df["CustomerType"] == 0]["Score"].values
            z1 = df[df["CustomerType"] == 0]["Gender"].values

            x2 = df[df["CustomerType"] == 1]["Age"].values
            y2 = df[df["CustomerType"] == 1]["Score"].values
            z2 = df[df["CustomerType"] == 1]["Gender"].values

            x3 = df[df["CustomerType"] == 2]["Age"].values
            y3 = df[df["CustomerType"] == 2]["Score"].values
            z3 = df[df["CustomerType"] == 2]["Gender"].values

            x4 = df[df["CustomerType"] == 3]["Age"].values
            y4 = df[df["CustomerType"] == 3]["Score"].values
            z4 = df[df["CustomerType"] == 3]["Gender"].values

            x5 = df[df["CustomerType"] == 4]["Age"].values
            y5 = df[df["CustomerType"] == 4]["Score"].values
            z5 = df[df["CustomerType"] == 4]["Gender"].values
            
            x6 = df[df["CustomerType"] == 5]["Age"].values
            y6 = df[df["CustomerType"] == 5]["Score"].values
            z6 = df[df["CustomerType"] == 5]["Gender"].values
            
            x7 = df[df["CustomerType"] == 6]["Age"].values
            y7 = df[df["CustomerType"] == 6]["Score"].values
            z7 = df[df["CustomerType"] == 6]["Gender"].values
            
            x8 = df[df["CustomerType"] == 7]["Age"].values
            y8 = df[df["CustomerType"] == 7]["Score"].values
            z8 = df[df["CustomerType"] == 7]["Gender"].values

            scatter1 = ax.scatter(x1, y1, z1, c="red", alpha=0.65, label="Type1")
            scatter2 = ax.scatter(x2, y2, z2, c="orange", alpha=0.65, label="Type2")
            scatter3 = ax.scatter(x3, y3, z3, c="blue", alpha=0.65, label="Type3")
            scatter4 = ax.scatter(x4, y4, z4, c="green", alpha=0.65, label="Type4")
            scatter5 = ax.scatter(x5, y5, z5, c="purple", alpha=0.65, label="Type5")
            scatter6 = ax.scatter(x6, y6, z6, c="pink", alpha=0.65, label="Type6")
            scatter7 = ax.scatter(x7, y7, z7, c="brown", alpha=0.65, label="Type7")
            scatter8 = ax.scatter(x8, y8, z8, c="yellow", alpha=0.65, label="Type8")

            [scatter1, scatter2, scatter3, scatter4, scatter5, scatter6, scatter7, scatter8]
'''