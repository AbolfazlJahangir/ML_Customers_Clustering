import pylab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from scipy.cluster import hierarchy 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("customers.csv")

df["Annual Income (k$)"] = df["Annual Income (k$)"] * 1000

df = df.drop(columns=["CustomerID"])

columns = {"Gender": "Gender", "Age": "Age", "Annual Income (k$)": "Income", "Spending Score (1-100)": "Score"}

df = df.rename(columns=columns)

df["Gender"] = df["Gender"].replace({"Male": 1, "Female": 0})

df.boxplot()
df.hist()

scaler = StandardScaler()
x = df.values
x = scaler.fit_transform(x)

dist_matrix = euclidean_distances(x,x)

Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')

fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s %s]' % (int(float(df['Gender'][id])), df['Age'][id], df['Income'][id], df['Score'][id])
    
Dendrogram = hierarchy.dendrogram(Z_using_dist_matrix,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

model = AgglomerativeClustering(n_clusters=4, linkage='average')
model.fit(x)

clusters = model.labels_
df['CustomerType'] = model.labels_

n_clusters = max(model.labels_) + 1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = df[df.CustomerType == label]
    for i in subset.index:
            plt.text(subset.Age[i], subset.Income[i],str(subset['Gender'][i]), rotation=25) 
    plt.scatter(subset.Age, subset.Income, s= subset.Score*10, c=color, label='CustomerType'+str(label),alpha=0.5)

plt.legend()
plt.title('Clusters')
plt.xlabel('Age')
plt.ylabel('Income')
