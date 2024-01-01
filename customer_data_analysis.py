import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("online_shoppers_intention.csv")

print(df.head())
print(df.describe())
print(df.shape)
print(df.isna().sum())

plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), annot=True, cmap="viridis", linewidths=.5)
plt.show()

df.drop(columns=["Administrative", "Informational", "ProductRelated_Duration"], inplace=True)
print(df.head())
print(df["ProductRelated"].value_counts())
print(df["Month"].value_counts())
print(df["OperatingSystems"].value_counts())
print(df["VisitorType"].value_counts())
print(df["Browser"].value_counts())
print(df["Region"].value_counts())
print(df.info())

visitor = pd.get_dummies(df["VisitorType"])
df = pd.concat([df, visitor], axis=1)
print(df.head())
df.drop(columns=["VisitorType", "Month"], inplace=True)
print(df.head())

encoder = OrdinalEncoder()
df["Weekend"] = encoder.fit_transform(df[["Weekend"]])
df["Revenue"] = encoder.fit_transform(df[["Revenue"]])

print(df.info())
print(df.shape)

X = df.drop(columns="Revenue").values
y = df["Revenue"].values


model = Sequential()

model.add(Dense(32, input_shape=(15, ), activation="relu"))
model.add(Dense(18, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, y, validation_split=0.2, epochs=120, batch_size=10)

loss, accuracy = model.evaluate(X, y)
print(loss, accuracy)

y_pred = (model.predict(X) > 0.5).astype(int)

model_json = model.to_json()

with open("crpmodel.json", "w")as file:
    file.write(model_json)

model.save_weights("model.h5")
