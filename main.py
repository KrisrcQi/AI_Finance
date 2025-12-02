import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# load dataset
df = pd.read_csv("/Users/krisqi/ACCFIN5230_lab_workspace/lab_1/loan.csv")

# dataset inspection
print(df.head())
print(df.info())
print(df.describe())

# plot the histogram
plt.figure(figsize=(8,4))
plt.hist(df['fico'], bins = 11, edgecolor = "black")
plt.title('Histogram of fico')
plt.xlabel('Credit score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# plot the scatter with trend line and correlation
x = df['dti']
y = df['fico']

s, i = np.polyfit(x, y, 1)
line = np.linspace(x.min(), x.max(), 100)
r = np.corrcoef(x, y)[0, 1]


fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(x, y, s=24, alpha=0.8)
ax.plot(line, s*line + i, color = 'red', label = 'Trend line')
ax.set_title(f'dti vs fico (r={r:.2f})')
ax.set_xlabel('Debt to income ratio')
ax.set_ylabel('Credit score')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.legend()
plt.show()


# normalize dataset
def nomalize (x):
    """
    nom_x = (x-min)/(max-min)
    
    """
    nomalized_data = ((x-min(x))/(max(x) - min(x)))
    return nomalized_data

std_df = df.apply(nomalize)

# splitting trainning data and test data
X = std_df[['int.rate', 'installment', 'log.annual.inc', 'dti']]
Y = std_df['fico']

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=11,
)

x_train = X_train
x_test  = X_test
y_train = Y_train
y_test  = Y_test

# design a MLP and fitting data to MLP
nn = MLPRegressor(
    hidden_layer_sizes = (2,), # one hidden layer with 2 units
    activation = 'relu',
    max_iter = 1000,
    random_state = 11
)

nn.fit(x_train, y_train)

# out-of-sample prediction:
y_pred = nn.predict(x_test)

# combining y_pred and y_test as dateframe
results_df = pd.DataFrame({
    "Groundtruth":y_test.values,
    "Prediction":y_pred
})
print(results_df.head())

y_min = y_test.min()
y_max = y_test.max()
y_range = y_max - y_min

print("min:", y_min)
print("max:", y_max)
print("range:", y_range)

ori_pred = results_df['Prediction'] * y_range + y_min
ori_act = results_df['Groundtruth'] * y_range + y_min

originalScaledPrediction = pd.DataFrame({
    "Prediction": ori_pred,
    "Groundtruth":ori_act
})
print(originalScaledPrediction)

# accuracy calculation
deviation = np.where(
    ori_act != 0, 
    (ori_act - ori_pred) / ori_act, 
    0
)

comparison = pd.DataFrame({
    "Prediction": ori_pred,
    "Groundtruth":ori_act,
    "Deviation": deviation
})

accuracy = 1 - abs(np.mean(deviation))
print("Accuracy:", accuracy)
print(comparison.head())



