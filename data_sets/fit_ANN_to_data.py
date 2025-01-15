import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as pl
import scienceplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Build the model
ANN_model = Sequential([
    Dense(6, activation='relu', input_shape=(3,)), # Input layer (3 neurons) + Hidden layer (6 neurons)
    Dense(4, activation='relu'),                   # Hidden layer (4 neurons)
    Dense(3)                                       # Output layer (3 neurons, no activation for regression)
])

# Compile the model
ANN_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Let's load the data stored in neo_hookean_hyperelastic_law/
number_cases = 37
name = "neo_hookean_hyperelastic_law/E_S_data_case_"
for case in range(1, number_cases):
    loaded_data = np.load(name + str(case) + ".npz")
    if case == 1:
        strain = loaded_data["strain_history"]
        stress = loaded_data["stress_history"]
    else:
        strain = np.vstack((strain, loaded_data["strain_history"]))
        stress = np.vstack((stress, loaded_data["stress_history"]))

# We must normalize the data
scaler = StandardScaler()
strain_scaled = scaler.fit_transform(strain)
stress_scaled = scaler.fit_transform(stress)

pl.style.use('science')
pl.plot(strain[:, 0], stress[:, 0], label=r"Ground truth $\varepsilon_{xx}$", marker='o', color="k",  markersize=6, markerfacecolor='none')
pl.plot(strain[:, 1], stress[:, 1], label=r"Ground truth $\varepsilon_{yy}$", marker='o', color="r",  markersize=6, markerfacecolor='none')
pl.plot(strain[:, 2], stress[:, 2], label=r"Ground truth $\gamma_{xy}$",      marker='o', color="b",  markersize=6, markerfacecolor='none')
pl.xlabel("Green-Lagrange Strain [-]")
pl.ylabel("PK2 Stress [Pa]")

# Train the ANN model
print("Data read...")
print("Training the ANN model now...")
ANN_model.fit(strain_scaled, stress_scaled, epochs=4000, verbose=1) # batch_size=20

# Evaluate the model
loss, mae = ANN_model.evaluate(strain_scaled, stress_scaled, verbose=1)
print(f"Training Loss: {loss:.4f}, Training MAE: {mae:.4f}")

# Undo the normalization
stress_ann_prediction = scaler.inverse_transform(ANN_model.predict(strain_scaled))

pl.plot(strain[:, 0], stress_ann_prediction[:, 0], label=r"ANN prediction $\varepsilon_{xx}$" , color = "k", linestyle="--", linewidth = 3)
pl.plot(strain[:, 1], stress_ann_prediction[:, 1], label=r"ANN prediction $\varepsilon_{yy}$" , color = "r", linestyle="--", linewidth = 3)
pl.plot(strain[:, 2], stress_ann_prediction[:, 2], label=r"ANN prediction $\gamma_{xy}$"      , color = "b", linestyle="--", linewidth = 3)
pl.legend(loc='best')
pl.show()