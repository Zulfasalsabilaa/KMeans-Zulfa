import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
from plotly import graph_objects

df = pd.read_csv('CC GENERAL.csv')

st.header("isi dataset")
st.write(df)
features = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'CASH_ADVANCE', 'PAYMENTS', 'CASH_ADVANCE_FREQUENCY']
x = df[features]
y = df['TENURE']
print(x.shape, y.shape)

#Menampilkan panah Elbow
clusters = []

for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(x)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

# Panah elbow
ax.annotate('Possible Elbow Point', xy=(2, 1.55), xytext=(2, 1.00),xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
ax.annotate('Possible Elbow Point', xy=(5, 0.80), xytext=(5, 1.5),xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster :", 2,10,3,1)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(x)
    x['Labels'] = kmean.labels_
    plt.figure(figsize=(10,8))
    sns.scatterplot(x = x['BALANCE'], y = x['PAYMENTS'], hue=x['Labels'], markers=True, size=x['Labels'], palette=sns.color_palette('hls', n_clust))

    for label in x['Labels']:
        plt.annotate(label,
            (x[x['Labels']==label]['BALANCE'].mean(),
             x[x['Labels']==label]['PAYMENTS'].mean()),
             horizontalalignment = 'center',
             verticalalignment = 'center',
             size = 20, weight='bold',
             color = 'black')
    st.header('Cluster Plot')
    st.pyplot()
    st.write(x)

k_means(clust)
