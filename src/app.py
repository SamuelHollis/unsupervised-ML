from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# clustering
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# metricas
from sklearn.metrics import silhouette_score

# reduccion de dimensionalidad
from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings("ignore")

data_1 = "C://Users//samue//OneDrive//Escritorio//4GeeksAcademy//25a clase-Aprendizaje_no_supervisado//unsupervised-ML//data//raw//data.csv"

df = pd.read_csv(data_1)

columns = ['Latitude','Longitude','MedInc']

X = df[columns].values

X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

model = KMeans(n_clusters = 6, random_state = 42)
model.fit(X_train)

train_labels = model.predict(X_train)
test_labels = model.predict(X_test)