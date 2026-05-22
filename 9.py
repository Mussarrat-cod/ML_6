from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = load_iris()
X = pd.DataFrame(data.data, columns=['SL','SW','PL','PW'])
Y = data.target

colors = np.array(['red', 'lime', 'black'])

plt.figure(figsize=(14,7))

# Real labels
plt.subplot(1,3,1)
plt.scatter(X.PL, X.PW, c=colors[Y])
plt.title('Real')

# KMeans
k = KMeans(n_clusters=3)
k.fit(X)
plt.subplot(1,3,2)
plt.scatter(X.PL, X.PW, c=colors[k.labels_])
plt.title('KMeans')

# GMM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

g = GaussianMixture(n_components=3)
g.fit(X_scaled)

plt.subplot(1,3,3)
plt.scatter(X.PL, X.PW, c=colors[g.predict(X_scaled)])
plt.title('GMM')

plt.show()