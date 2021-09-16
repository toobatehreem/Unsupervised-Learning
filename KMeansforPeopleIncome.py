import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\Tooba Tehreem Sheikh\\PycharmProjects\\AI Course\\income.csv')
print(df.head())

plt.scatter(df['Age'], df['Income($)'])
plt.show()

#working on unscaled data
km = KMeans(n_clusters=3)
print(km)

y_predicted = km.fit_predict(df[['Age', 'Income($)']]) #divided into 3 clusters 0,1,2
print(y_predicted)

df['cluster'] = y_predicted
print(df.head())

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1['Income($)'], color='green')
plt.scatter(df2.Age, df2['Income($)'], color='red')
plt.scatter(df3.Age, df3['Income($)'], color='black')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='purple', marker='+', label='centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
plt.show()

#scaling the data
scaler = MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
print(df.head())

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
print(df.head())

#working on scaled data
km = KMeans(n_clusters=3)
print(km)

y_predicted = km.fit_predict(df[['Age', 'Income($)']]) #divided into 3 clusters 0,1,2
print(y_predicted)

df['cluster'] = y_predicted
print(df.head())

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1['Income($)'], color='green')
plt.scatter(df2.Age, df2['Income($)'], color='red')
plt.scatter(df3.Age, df3['Income($)'], color='black')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='purple', marker='+', label='centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
plt.show()