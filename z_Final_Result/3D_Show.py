import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv("customers.csv")

df["Annual Income (k$)"] = df["Annual Income (k$)"] * 1000

df = df.drop(columns=["CustomerID"])

columns = {"Gender": "Gender", "Age": "Age", "Annual Income (k$)": "Income", "Spending Score (1-100)": "Score"}

df = df.rename(columns=columns)

df["Gender"] = df["Gender"].replace({"Male": 1, "Female": 0})

scaler = StandardScaler()
x = df.values
x = scaler.fit_transform(x)

model = KMeans(n_clusters=4, init="k-means++", n_init=30, random_state=4)
customerType = model.fit_predict(x)
df["CustomerType"] = customerType
labels = model.labels_

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

scatter1 = ax.scatter(x1, y1, z1, c="red", alpha=0.65, label="Type1")
scatter2 = ax.scatter(x2, y2, z2, c="orange", alpha=0.65, label="Type2")
scatter3 = ax.scatter(x3, y3, z3, c="blue", alpha=0.65, label="Type3")
scatter4 = ax.scatter(x4, y4, z4, c="green", alpha=0.65, label="Type4")

ax.legend(handles=[scatter1, scatter2, scatter3, scatter4], loc="upper left", title="Customers")

ax.set_xlabel("Age", fontsize=15)
ax.set_ylabel("Score", fontsize=15)
ax.set_zlabel("Gender", fontsize=15)

plt.show()
