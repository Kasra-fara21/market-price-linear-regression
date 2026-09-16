# importing:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression 

# make the data:
suply = np.array([
    203, 567, 354, 735,
    809, 45, 902, 293,
    464, 90, 469, 1001,
    1004, 341, 842, 65,
    82, 733, 123, 789,
    1346, 1992, 234, 267,
    667, 764, 456, 1346,
    48, 672, 333, 331,
    546, 143, 1943, 345,
    145, 563, 772, 98,
    13, 112, 155, 2842,
    3567, 321, 56, 59,
    671, 999, 23, 404,
    616, 989, 718, 790,
    236, 934, 2331, 786,
    982, 144, 56, 15,
    1821, 711, 456, 659,
    135, 566, 98, 14,
    56, 1344, 414, 784,
    6789, 2556, 0, -1,
    1547, -14, 15, 168,
    145, 167, 1689, 234,
    0, 2567, 3455, 1354,
    1321, -112, 1311, 2346,
    516, 918, 102, -82
])

demand = np.array([
    1036, 2344, 872, 679,
    1355, 2311, 2223, 149,
    198, 2345, 1345, 356,
    98, 64, 1345, 1361,
    631, 1181, 1397, 792,
    137, 219, 456, 146,
    145, 1667, 1151, 7234,
    1346, 247, 321, 245,
    494, 45, 1245, 1459,
    4351, 1111, 231, 1441,
    4114, 2113, 2451, 4921,
    11, 1452, 113, 13,
    534, 411, 414, 981,
    45, 761, 115, 141,
    117, 1838, 42, 455,
    554, 1452, 3112, 3457,
    143, 452, 987, 987,
    1000, 1344, 134, 872,
    1354, 5432, 134, 24,
    72, 23, 25, 1084,
    -83, 0, 100, 11,
    -145, 1452, 2562, -1,
    0, 1494, 552, 156,
    293, 1442, 1415, 5432,
    134, 11, 11, 241
])

# make dataset with arrays:
Data = {"suply": suply, "demand": demand}
df = pd.DataFrame(Data)

df = df[df["suply"]> 0]   
df = df[df["demand"]> 0]

np.random.seed(42)
Price = ((df["demand"] / df["suply"]) * 10000) + np.random.randint(-10, 10, len(df))
 
df["Price"] = Price


# ploting and make regression chart:
plt.style.use("ggplot")
plt.figure(figsize=(10, 6))

sns.scatterplot(x = "suply", y = "Price", color = 'red', data=df)
sns.regplot(x = "suply", y = "Price", color = 'blue', data=df)
plt.title("Relationship between suply and price")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x = "demand", y = "Price", color ="red", data=df)
sns.regplot(x = "demand", y = "Price", color = "blue", data=df)
plt.title("Relationship between demand and price")
plt.show()


# build regression model an make preditions:
reg = LinearRegression()

X = df.drop(columns=["Price"]) 
y = df[["Price"]]

reg.fit(X, y)
predict_data = pd.DataFrame({
    "suply": [500, 500, 500, 1000, 1000],
    "demand": [300, 600, 900, 600, 900]
})
predicts = reg.predict(predict_data)

print(f"the model predictions: \n{predicts}")

m = reg.coef_ 
b = reg.intercept_

formula = m[0][0] * predict_data["suply"] + m[0][1] * predict_data["demand"] + b

# Results:
print(f"\n get model prediction with Formula: \n{formula}") 
print(f"the coef: {m} \n the intercept{b}") 
print(f"\nthe Formula: m1 × first feature + m2 × second feature + b \n for our model the Formula is (m1 * suply + m2 * demand + b)")
