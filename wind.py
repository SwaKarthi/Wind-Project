import pandas as pd
import numpy as np
import xgboost as xg
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, Concatenate
from math import radians

# Reading the dataset as pandas DataFrame
file_path = "T1.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Converting all the column names to lower case
df.columns = [c.lower() for c in df.columns]

print('Show the first 5 rows')
print(df.head())

print('What are the variable data types?')
print(df.dtypes)

print('How many observations do we have?')
print(df.shape[0])

# Extracting a substring from columns to create month and hour variables
df['month'] = df['date/time'].str[3:5].astype(int)
df['hour'] = df['date/time'].str[11:13].astype(int)

print(df.head())

# Describe some numerical columns
pd.options.display.float_format = '{:.2f}'.format
print(df[['wind speed (m/s)', 'theoretical_power_curve (kwh)', 'lv activepower (kw)']].describe())

# Taking a random sample from the big data
sample_df = df.sample(frac=0.1, random_state=42)

# Visualizing the distributions with the sample data
columns = ['wind speed (m/s)', 'month', 'hour', 'theoretical_power_curve (kwh)', 'lv activepower (kw)']
plt.figure(figsize=(10,12))
for i, each in enumerate(columns, 1):
    plt.subplot(3, 2, i)
    sample_df[each].plot.hist(bins=12)
    plt.title(each)

plt.tight_layout()
plt.show()

# Average power production by month
monthly = df.groupby('month')['lv activepower (kw)'].mean().reset_index().sort_values('lv activepower (kw)')
sns.barplot(x='month', y='lv activepower (kw)', data=monthly)
plt.title('Months and Average Power Production')
plt.show()

# Average power production by hour
hourly = df.groupby('hour')['lv activepower (kw)'].mean().reset_index().sort_values('lv activepower (kw)')
sns.barplot(x='hour', y='lv activepower (kw)', data=hourly)
plt.title('Hours and Average Power Production')
plt.show()

# Correlation matrix and pairplot
print(sample_df[columns].corr())
sns.pairplot(sample_df[columns], markers='*')
plt.show()

# Average power production for 5 m/s wind speed increments
wind_speed = []
avg_power = []
for i in range(0, 25, 5):
    avg_value = df[(df['wind speed (m/s)'] > i) & (df['wind speed (m/s)'] <= i + 5)]['lv activepower (kw)'].mean()
    avg_power.append(avg_value)
    wind_speed.append(f"{i}-{i+5}")

sns.barplot(x=wind_speed, y=avg_power, color='orange')
plt.title('Avg Power Production for 5 m/s Wind Speed Increments')
plt.xlabel('Wind Speed')
plt.ylabel('Average Power Production')
plt.show()

# Creating the polar diagram
plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)
sns.scatterplot(x=[radians(x) for x in sample_df['wind speed (m/s)']], 
                y=sample_df['lv activepower (kw)'],
                size=sample_df['lv activepower (kw)'],
                hue=sample_df['lv activepower (kw)'],
                alpha=0.7, legend=None)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rlabel_position(110)
plt.title('Wind Speed - Wind Direction - Power Production Diagram')
plt.ylabel(None)
plt.show()

# Wind Speed and Power Production Chart
plt.figure(figsize=(10,6))
sns.scatterplot(x='wind speed (m/s)', y='lv activepower (kw)', color='orange', label='Real Production', alpha=0.5, data=sample_df)
sns.lineplot(x='wind speed (m/s)', y='theoretical_power_curve (kwh)', color='blue', label='Theoretical Production', data=sample_df)
plt.title('Wind Speed and Power Production Chart')
plt.ylabel('Power Production (kw)')
plt.legend()
plt.show()

# Filter the data where real and theoretical power productions are zero
zero_theo_power = df[(df['lv activepower (kw)'] == 0) & (df['theoretical_power_curve (kwh)'] == 0)]
print(zero_theo_power[['wind speed (m/s)', 'theoretical_power_curve (kwh)', 'lv activepower (kw)']].sample(5))

# Wind Speed Distribution for 0 Power Production
zero_theo_power['wind speed (m/s)'].hist()
plt.title('Wind Speed Distribution for 0 Power Production')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Counts for 0 Power Production')
plt.show()

# Observations for wind speed > 3m/s and power production = 0, but theoretical production exists
zero_power = df[(df['lv activepower (kw)'] == 0) & (df['theoretical_power_curve (kwh)'] != 0) & (df['wind speed (m/s)'] > 3)]
print('No of Observations (while Wind Speed > 3 m/s and Power Production = 0):', len(zero_power))

# Month-wise count of zero power production observations
sns.countplot(x='month', data=zero_power)
plt.title('Month-wise Count of Zero Power Production Observations')
plt.show()

# Excluding observations with zero power production when theoretical production should exist
df = df[~((df['lv activepower (kw)'] == 0) & (df['theoretical_power_curve (kwh)'] != 0) & (df['wind speed (m/s)'] > 3))]

# Boxplot of selected columns
columns = ['wind speed (m/s)', 'theoretical_power_curve (kwh)', 'lv activepower (kw)']
plt.figure(figsize=(20, 3))
for i, col in enumerate(columns, 1):
    plt.subplot(1, 4, i)
    sns.boxplot(data=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Define the condition for repair
def check_for_repair(row, threshold=0.5):
    # If the actual power is less than threshold*theoretical power, flag for repair
    return row['lv activepower (kw)'] < threshold * row['theoretical_power_curve (kwh)']

# Add a repair flag to the DataFrame
df['needs_repair'] = df.apply(check_for_repair, axis=1)

# Filtering the data for repair
repair_needed_df = df[df['needs_repair'] == True]
number_of_turbines_needing_repair = repair_needed_df.shape[0]
print(f'Number of turbines needing repair: {number_of_turbines_needing_repair}')

# Log the turbines needing repair to a file
with open('repair_log.txt', 'a') as log_file:
    log_file.write(f"Number of turbines needing repair: {number_of_turbines_needing_repair}\n")
    log_file.write(repair_needed_df.to_string())
    log_file.write("\n\n")

# Alarm to repair
if number_of_turbines_needing_repair > 0:
    print("ALARM: Some turbines need repair! Check the 'repair_log.txt' for details.")
else:
    print("All turbines are operating within acceptable parameters.")

# Plot turbines needing repair
plt.figure(figsize=(10, 6))
sns.scatterplot(x='wind speed (m/s)', y='lv activepower (kw)', color='red', label='Needs Repair', data=repair_needed_df)
plt.title('Turbines Needing Repair')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Actual Power Output (kW)')
plt.legend()
plt.show()

# Calculating quantiles and interquartile range for wind speed
Q1 = df['wind speed (m/s)'].quantile(0.25)
Q3 = df['wind speed (m/s)'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print('Quantile (0.25):', Q1, ' Quantile (0.75):', Q3)
print('Lower threshold:', lower, ' Upper threshold:', upper)

# Identifying outliers
outliers = df[(df['wind speed (m/s)'] < lower) | (df['wind speed (m/s)'] > upper)]
print('Total Number of Outliers:', len(outliers))
print('Some Examples of Outliers:')
print(outliers['wind speed (m/s)'].sample(10))

# Average power production for wind speed >= 19 m/s
print(df[df['wind speed (m/s)'] >= 19][['wind speed (m/s)', 'lv activepower (kw)']].describe().transpose())

# Removing outliers
df = df[(df['wind speed (m/s)'] >= lower) & (df['wind speed (m/s)'] <= upper)]
columns = ['wind speed (m/s)', 'theoretical_power_curve (kwh)', 'lv activepower (kw)']
print(df[columns].describe().transpose())

# Creating a new variable for log transformation
df['log_theoretical_power_curve (kwh)'] = np.log(df['theoretical_power_curve (kwh)'] + 1)

# Updating column names to be consistent
df.rename(columns={'wind speed (m/s)': 'wind_speed', 'wind direction (ø)': 'wind_direction', 'theoretical_power_curve (kwh)': 'theoretical_power_curve', 'lv activepower (kw)': 'lv_activepower'}, inplace=True)
print(df.head())

# Splitting the dataset into train and test sets for machine learning model training and prediction
X = df[['month', 'hour', 'wind_speed', 'wind_direction']]
y = df['lv_activepower']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for CNN and RNN
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)
X_train_rnn = np.expand_dims(X_train, axis=1)
X_test_rnn = np.expand_dims(X_test, axis=1)

input_shape_cnn = (X_train_cnn.shape[1], X_train_cnn.shape[2])

# CNN model for feature extraction
input_cnn = Input(shape=(X_train_cnn.shape[1], X_train_cnn.shape[2]))
cnn_layer = Conv1D(filters=512, kernel_size=2, activation='relu')(input_cnn)
cnn_layer = Conv1D(filters=128, kernel_size=2, activation='relu')(cnn_layer)
cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
cnn_layer = Flatten()(cnn_layer)
cnn_model = Model(inputs=input_cnn, outputs=cnn_layer)

# RNN model for feature extraction
input_rnn = Input(shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]))
rnn_layer = LSTM(50, return_sequences=True)(input_rnn)
rnn_layer = LSTM(50)(rnn_layer)
rnn_model = Model(inputs=input_rnn, outputs=rnn_layer)

# Concatenate CNN and RNN features
combined_input = Concatenate()([cnn_model.output, rnn_model.output])
combined_model = Model(inputs=[cnn_model.input, rnn_model.input], outputs=combined_input)
combined_model.summary()

# Extract features
train_features = combined_model.predict([X_train_cnn, X_train_rnn])
test_features = combined_model.predict([X_test_cnn, X_test_rnn])

# Machine learning model (e.g., XGBoost)
ml_model = xg.XGBRegressor(max_depth=10, n_estimators=1000, min_child_weight=0.5, colsample_bytree=0.8, subsample=0.8, eta=0.1, seed=42)

ml_model.fit(train_features, y_train)
predictions = ml_model.predict(test_features)

# Save model
# joblib.dump(ml_model, 'wind_power_generation_model.joblib')

# Evaluation
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f'RMSE: {rmse}')
print(f'R² Score: {r2}')
print(f'MAE: {mae}')

# Visualization
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test.values, y=predictions, scatter_kws={'alpha':0.5})
plt.xlabel('Actual Power Output (MW)')
plt.ylabel('Predicted Power Output (MW)')
plt.title('Actual vs Predicted Power Output')
plt.show()
