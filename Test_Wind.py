import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the combined CNN and RNN model
combined_model = load_model('combined_model.h5')

# Load the trained machine learning model
ml_model = joblib.load('wind_power_generation_model.joblib')

def preprocess_input(wind_speed, wind_direction, theoretical_power):
    # Create a DataFrame with the input data
    input_data = {
        'wind_speed': [wind_speed],
        'wind_direction': [wind_direction],
        'theoretical_power': [theoretical_power],
        'dummy_feature': [0]  # Adding dummy feature to match expected input shape
    }
    
    input_df = pd.DataFrame(input_data)
    
    return input_df

def predict_power_output(wind_speed, wind_direction, theoretical_power):
    # Preprocess the input
    input_X = preprocess_input(wind_speed, wind_direction, theoretical_power)
    
    # Reshape the data for CNN and RNN models
    input_X_cnn = np.expand_dims(input_X, axis=2)  # Shape: (1, 4, 1)
    input_X_rnn = np.expand_dims(input_X, axis=1)  # Shape: (1, 1, 4)
    
    # Extract features using the CNN and RNN models
    input_features = combined_model.predict([input_X_cnn, input_X_rnn])
    
    # Predict the power output
    predicted_power_output = ml_model.predict(input_features)
    
    return predicted_power_output[0]

# Get user input
wind_speed = float(input('Enter wind speed (m/s): '))
wind_direction = float(input('Enter wind direction (ø): '))
theoretical_power = float(input('Enter theoretical power (kW): '))

# Get the prediction
predicted_output = predict_power_output(wind_speed, wind_direction, theoretical_power)
print(f'Predicted Power Output (kW): {predicted_output}')

# Visualize the prediction
plt.figure(figsize=(6, 4))
plt.bar(['Predicted Power Output'], [predicted_output], color='orange')
plt.title('Predicted Power Output')
plt.ylabel('Power Output (kW)')
plt.show()
