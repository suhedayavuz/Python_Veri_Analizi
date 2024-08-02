# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 23:13:14 2024
@author: suhed
"""

# Gerekli kütüphaneleri içe aktar
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# CSV dosyasından veriyi içe aktar
input_data = pd.read_csv("customer_shopping_data.csv")

# Giriş verisini incele
print(input_data.info())

# Gereksiz sütunları çıkar
filter_input = input_data.drop(['invoice_no', 'customer_id', 'invoice_date'], axis=1)

# Quantity sütununu kategorik olarak işaretle
filter_input.quantity = filter_input.quantity.astype(str)

# Veri türü 'object' olan sütunları incele
for col in filter_input.select_dtypes(include='object'):
    print(col)
    print(filter_input[col].unique())

# Veriyi ölçeklendir
ss = StandardScaler()
ss_input = filter_input.copy()
cols = filter_input.select_dtypes(exclude='object').columns
ss_input[cols] = ss.fit_transform(ss_input[cols])

# Kategorik sütunların pozisyonlarını belirle
catColumnsPos = [ss_input.columns.get_loc(col) for col in list(ss_input.select_dtypes('object').columns)]
print('Kategorik sütunlar           : {}'.format(list(ss_input.select_dtypes('object').columns)))
print('Kategorik sütunların pozisyonları  : {}'.format(catColumnsPos))

# Veriyi numpy matrisine dönüştür
dfMatrix = ss_input.to_numpy()

# Kümelere ayırma (n_clusters=3)
n_clusters = 3
kprototype = KPrototypes(n_jobs=-1, n_clusters=n_clusters, init='Huang', random_state=0)
clusters = kprototype.fit_predict(dfMatrix, categorical=catColumnsPos)

# Kümeleri isimlendirme
cluster_names = ['Cluster_' + str(i) for i in range(n_clusters)]
filter_input['cluster'] = [cluster_names[cluster] for cluster in clusters]

print("Cluster Centers:", kprototype.cluster_centroids_)
print("Cluster Labels:", clusters)

# Grafiklerin Oluşturulması
# Gender vs. Category (Bar Plot)
plt.figure(figsize=(10, 6))
sns.countplot(data=filter_input, x='gender', hue='category', palette='bright')
plt.title('Gender vs. Category (Bar Plot)')
plt.legend(title='Category', loc='upper right')
plt.show()

# Gender vs. Payment Method (Pie Chart)
plt.figure(figsize=(10, 6))
gender_payment_method_counts = filter_input.groupby('gender')['payment_method'].value_counts(normalize=True).unstack()
gender_payment_method_counts.plot.pie(subplots=True, figsize=(15, 6), autopct='%1.1f%%', legend=False)
plt.title('Gender vs. Payment Method (Pie Chart)')
plt.show()

# Gender vs. Shopping Mall (Horizontal Bar Plot)
plt.figure(figsize=(10, 6))
sns.countplot(data=filter_input, y='gender', hue='shopping_mall', palette='bright')
plt.title('Gender vs. Shopping Mall (Horizontal Bar Plot)')
plt.legend(title='Shopping Mall', loc='best')
plt.show()

# Payment Method vs. Shopping Mall (Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(data=filter_input, x='payment_method', hue='shopping_mall', multiple='stack', palette='bright')
plt.title('Payment Method vs. Shopping Mall (Histogram)')
plt.legend(title='Shopping Mall', loc='upper right')
plt.show()

# Gender-Category-Shopping Mall (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filter_input, x='category', y='shopping_mall', hue='gender', palette='bright')
plt.title('Gender-Category-Shopping Mall (Scatter Plot)')
plt.legend(title='Gender', loc='upper right')
plt.show()

# Gender vs. Quantity (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=filter_input, x='gender', y='quantity', palette='bright')
plt.title('Gender vs. Quantity (Box Plot)')
plt.show()

# Category vs. Quantity (Violin Plot)
plt.figure(figsize=(10, 6))
sns.violinplot(data=filter_input, x='category', y='quantity', palette='bright')
plt.title('Category vs. Quantity (Violin Plot)')
plt.show()

# Payment Method vs. Quantity (Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(data=filter_input, x='payment_method', y='quantity', hue='shopping_mall', multiple='stack', palette='bright')
plt.title('Payment Method vs. Quantity (Histogram)')
plt.legend(title='Shopping Mall', loc='upper right')
plt.show()

# Shopping Mall vs. Quantity (Bar Plot)
plt.figure(figsize=(10, 6))
sns.barplot(data=filter_input, x='shopping_mall', y='quantity', hue='gender', palette='bright')
plt.title('Shopping Mall vs. Quantity (Bar Plot)')
plt.legend(title='Gender', loc='upper right')
plt.show()

# Gender vs. Category (Pie Chart)
plt.figure(figsize=(10, 6))
gender_category_counts = filter_input.groupby('gender')['category'].value_counts(normalize=True).unstack()
gender_category_counts.plot.pie(subplots=True, figsize=(15, 6), autopct='%1.1f%%', legend=False)
plt.title('Gender vs. Category (Pie Chart)')
plt.show()

# Payment Method vs. Category (Horizontal Bar Plot)
plt.figure(figsize=(10, 6))
sns.countplot(data=filter_input, y='payment_method', hue='category', palette='bright')
plt.title('Payment Method vs. Category (Horizontal Bar Plot)')
plt.legend(title='Category', loc='upper right')
plt.show()

# Kategorik sütunların tüm verilerini histogram ile görselleştirme
categorical_columns = ['gender', 'category', 'quantity', 'payment_method', 'shopping_mall']
num_plots = len(categorical_columns)
num_cols = 3
num_rows = num_plots // num_cols + 1

plt.figure(figsize=(20, 15))

for i, col in enumerate(categorical_columns, 1):
    plt.figure()
    sns.histplot(data=filter_input, x=col, palette='bright')
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Scatter Plot - Harcama Tutarı (Price) vs Harcama Adedi (Quantity)
XX = np.array(filter_input.price)
YY = np.array(input_data.quantity)
ZZ = np.array(filter_input.age)
GG = np.array(filter_input.gender)

fig = plt.figure()
plt.scatter(XX, YY, c=kprototype.labels_)
plt.xlabel("Fiyat")
plt.ylabel("Adet")
plt.grid()
plt.show()

# 3D Scatter Plot - Harcama Tutarı (Price), Yaş (Age) ve Harcama Adedi (Quantity)
fig = plt.figure(figsize=(14, 25))
ax = fig.add_subplot(projection='3d')
ax.scatter(YY, ZZ, XX, c=kprototype.labels_)
ax.set_zlabel("Yaş", fontsize=18)
ax.set_xlabel("Harcama Tutarı", fontsize=18)
ax.set_ylabel("Harcama Adedi", fontsize=18)
ax.set_title('Kümeleme 3 Boyut Gösterim', fontsize=24)
ax.grid(True)
plt.show()

# Scatter Plot - Harcama Tutarı (Price) vs Harcama Adedi (Quantity)
fig = plt.figure()
plt.scatter(XX, YY, c=kprototype.labels_)
plt.xlabel("Harcama Tutarı")
plt.ylabel("Harcama Adedi")
plt.grid()
plt.show()

# Scatter Plot - Harcama Tutarı (Price) vs Yaş (Age)
fig = plt.figure()
plt.scatter(XX, ZZ, c=kprototype.labels_)
plt.xlabel("Harcama Tutarı")
plt.ylabel("Yaş")
plt.grid()
plt.show()

# Scatter Plot - Harcama Adedi (Quantity) vs Yaş (Age)
fig = plt.figure()
plt.scatter(YY, ZZ, c=kprototype.labels_)
plt.xlabel("Harcama Adedi")
plt.ylabel("Yaş")
plt.grid()
plt.show()

# 3D Scatter Plot - Harcama Tutarı (Price), Cinsiyet (Gender) ve Yaş (Age)
fig = plt.figure(figsize=(14, 25))
ax = fig.add_subplot(projection='3d')
ax.scatter(XX, GG, ZZ, c=kprototype.labels_)
ax.set_zlabel("Yaş", fontsize=18)
ax.set_xlabel("Harcama Tutarı", fontsize=18)
ax.set_ylabel("Cinsiyet", fontsize=18)
ax.set_title('Kümeleme 3 Boyut Gösterim', fontsize=24)
ax.grid(True)
plt.show()
