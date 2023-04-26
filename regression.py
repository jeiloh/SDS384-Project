import kllr as kl
import pandas as pd
import numpy as np
import matplotlib.pylab as plt  
import matplotlib

from scipy import stats
import xgboost
import pandas as pd

import seaborn as sns
# upload the data
url = "https://github.com/jeiloh/SDS384-Project/blob/main/data/merged_df.csv?raw=true"
df = pd.read_csv(url)


def assign_county_id(county):
  if county == 'Bastrop County':
    return 1
  if county == 'Travis County':
    return 2
  if county == 'Williamson County':
    return 3
  return 4

df['County_ID'] = df['County'].apply(assign_county_id)
df.head()

# linear Regression
# Prepare the input and output data
input_features = [ 'Day of Week', 'Hour','Men', 'Women', 'Hispanic',
       'White', 'Black', 'Native', 'Asian', 'Pacific','Income']
output_feature = "num_trips"

# Prepare input and output data
X = df[input_features]
y = df[output_feature]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

for feature in input_features:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature], y)
    plt.xlabel(feature)
    plt.ylabel('num_trips')
    plt.title(f"{feature} vs num_trips")
    plt.show()

#Ridge Regression
input_features = ['County_ID', 'Day of Week', 'Hour', 'Men', 'Women', 'Hispanic',
                  'White', 'Black', 'Native', 'Asian', 'Pacific', 'Income']
output_feature = "num_trips"

X = df[input_features]
y = df[output_feature]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Ridge regression model
ridge = Ridge(alpha=1.0)

# Fit the model
ridge.fit(X_train, y_train)

# Predict on test set
y_pred = ridge.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)
# Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot(np.arange(y_test.min(), y_test.max()), np.arange(y_test.min(), y_test.max()), color='red', linestyle='--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.xlim(0, 1000)
plt.ylim(0, 5000)
plt.title('Actual vs Predicted Results')
plt.show()

#GPR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kernel = 0.25 * RBF(length_scale=5, length_scale_bounds="fixed") + WhiteKernel(noise_level=1, noise_level_bounds=(10, 1e6))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
gpr.fit(X_train, y_train)

# Predict on test set
y_pred = gpr.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Calculate R^2 Score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Plotting the results
plt.figure(figsize=(10, 7))

# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred, color='steelblue', label='Travis County')

# Add a diagonal line for reference
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2)

plt.legend(loc=4, prop={'size': 15})
plt.xlabel('Actual', size=22)
plt.ylabel('Predicted', size=22)
plt.tick_params(labelbottom=True)
plt.grid()

plt.xlim(0, 700)
plt.ylim(0, 700)

plt.show()

# KLLR group by income
kernel_width=0.5
_, ax = kl.Plot_Fit_Summary_Split(df, 'Day of Week', 'num_trips', 'Income', split_bins=[0, 50000, 70000, 100000,150000], 
                                  split_mode = 'Data', y_err=None, bins=34,
                                  nBootstrap=50, verbose = True, percentile=[16., 84.], kernel_type='gaussian',
                                  kernel_width=kernel_width, fast_calc = False,
                                  show_data=False, xlog=False, ylog=False, color=None,
                                  labels=['Household Income', 'Ownership Rate', 'X'], cmap = None, ax=None)

ax.flat[1].set_visible(False)
ax.flat[2].set_visible(False)

L = ax[0].legend(prop={'size':19})
L.get_texts()[0].set_text('Income < 50000')
L.get_texts()[1].set_text('50000 < Income < 70000')
L.get_texts()[2].set_text('70000 < Income < 100000')
L.get_texts()[3].set_text('100000 < Income < 150000')


ax[0].set_xlabel('Day of Week', size=32)
ax[0].set_ylabel('num_trips', size=32)
ax[0].tick_params(labelbottom=True)
ax[0].set_xticks([1, 2, 3, 4, 5, 6,7])
ax[0].set_xticklabels(['1', '2', '3', '4', '5', '6','7'])

kernel_width=1
_, ax = kl.Plot_Fit_Summary_Split(df, 'Hour', 'num_trips', 'Income', split_bins=[0, 50000, 70000, 100000,150000], 
                                  split_mode = 'Data', y_err=None, bins=34,
                                  nBootstrap=50, verbose = True, percentile=[16., 84.], kernel_type='gaussian',
                                  kernel_width=kernel_width, fast_calc = False,
                                  show_data=False, xlog=False, ylog=False, color=None,
                                  labels=['Hour', 'num_trips', 'X'], cmap = None, ax=None)

ax.flat[1].set_visible(False)
ax.flat[2].set_visible(False)

L = ax[0].legend(prop={'size':19})
L.get_texts()[0].set_text('HH Income < 50000')
L.get_texts()[1].set_text('50000 < Income < 70000')
L.get_texts()[2].set_text('70000 < Income < 100000')
L.get_texts()[2].set_text('100000 < Income < 150000')
ax[0].set_xticks([0,5,10,15,20,24])
ax[0].set_xticklabels(['0', '5', '10', '15', '20', '24'])
ax[0].set_xlabel('Hour', size=32)
ax[0].tick_params(labelbottom=True)
ax[0].set_xlim(0, 24)

# KLLR group by race
kernel_width=1
_, ax = kl.Plot_Fit_Summary_Split(df, 'Day of Week', 'num_trips', 'Black', split_bins=[0, 3, 5, 10], 
                                  split_mode = 'Data', y_err=None, bins=34,
                                  nBootstrap=50, verbose = True, percentile=[16., 84.], kernel_type='gaussian',
                                  kernel_width=kernel_width, fast_calc = False,
                                  show_data=False, xlog=False, ylog=False, color=None,
                                  labels=['Day of Week', 'num_trips', 'X'], cmap = None, ax=None)

ax.flat[1].set_visible(False)
ax.flat[2].set_visible(False)

L = ax[0].legend(prop={'size':19})
L.get_texts()[0].set_text('Black < 3')
L.get_texts()[1].set_text('3 < Black < 5')
L.get_texts()[2].set_text('7 < Black < 10')


ax[0].set_xlabel('Day of Week', size=32)
ax[0].tick_params(labelbottom=True)
ax[0].set_xticks([1, 2, 3, 4, 5, 6,7])
ax[0].set_xticklabels(['1', '2', '3', '4', '5', '6','7'])
ax[0].set_xlim(0, 7)
ax[0].set_ylim(0, 1500)

kernel_width=1
_, ax = kl.Plot_Fit_Summary_Split(df, 'Hour', 'num_trips', 'Black', split_bins=[0, 3, 5, 10], 
                                  split_mode = 'Data', y_err=None, bins=34,
                                  nBootstrap=50, verbose = True, percentile=[16., 84.], kernel_type='gaussian',
                                  kernel_width=kernel_width, fast_calc = False,
                                  show_data=False, xlog=False, ylog=False, color=None,
                                  labels=['Hour', 'num_trips', 'X'], cmap = None, ax=None)

ax.flat[1].set_visible(False)
ax.flat[2].set_visible(False)

L = ax[0].legend(prop={'size':19})
L.get_texts()[0].set_text('Black < 3')
L.get_texts()[1].set_text('3 < Black < 5')
L.get_texts()[2].set_text('5 < Black < 10')


ax[0].set_xlabel('Hour', size=32)
ax[0].tick_params(labelbottom=True)
ax[0].set_xticks([0,5,10,15,20,24])
ax[0].set_xticklabels(['0', '5', '10', '15', '20', '24'])
ax[0].set_xlabel('Hour', size=32)
ax[0].tick_params(labelbottom=True)
ax[0].set_xlim(0, 24)
ax[0].set_ylim(0, 1500)