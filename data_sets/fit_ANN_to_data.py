import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as pl
import scienceplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Build the model
ANN_model = Sequential()
ANN_model.add(Dense(100, activation='sigmoid', input_dim=3))
ANN_model.add(Dense(50, activation='sigmoid'))
ANN_model.add(Dense(3, activation='sigmoid'))

# custom_Adam = keras.optimizers.Adam(
#     learning_rate=0.005,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-7,
#     amsgrad=False
# )
# custom_Adam = tf.keras.optimizers.Nadam(learning_rate=0.002)

# Compile the model
ANN_model.compile(optimizer='adamax', loss='mae', metrics=['mae']) # mse

# Let's load the data stored in the dir neo_hookean_hyperelastic_law/
number_cases = 37
max_iterations = 1500

name = "neo_hookean_hyperelastic_law/E_S_data_case_"
strain = np.load("neo_hookean_hyperelastic_law/E_S_data_case_1.npz")["strain_history"]
stress = np.load("neo_hookean_hyperelastic_law/E_S_data_case_1.npz")["stress_history"]
for case in range(1, number_cases):
    loaded_data = np.load(name + str(case) + ".npz")
    strain = np.vstack((strain, loaded_data["strain_history"]))
    stress = np.vstack((stress, loaded_data["stress_history"]))

# We must normalize the data
strain_scaler = StandardScaler()
stress_scaler = StandardScaler()
strain_scaled = strain_scaler.fit_transform(strain)
stress_scaled = stress_scaler.fit_transform(stress)

# Train the ANN model
print("Data read...")
print("Training the ANN model now...")
ANN_model.fit(strain_scaled, stress_scaled, epochs=max_iterations, verbose=1) # batch_size=20

# Evaluate the model
loss, mae = ANN_model.evaluate(strain_scaled, stress_scaled, verbose=1)
print("******************************************************")
print(f"Training Loss: {loss:.8f}, Training MAE: {mae:.4f}")
print("******************************************************")

# Let's print the predictions v.s. the truth
for case in range(1, number_cases + 1):
    loaded_data = np.load(name + str(case) + ".npz")
    case_strain = loaded_data["strain_history"]
    case_stress = loaded_data["stress_history"]
    stress_ann_prediction = stress_scaler.inverse_transform(ANN_model.predict(strain_scaler.fit_transform(case_strain)))

    pl.style.use('science')
    pl.plot(case_strain[:, 0], case_stress[:, 0], label=r"Ground truth $\varepsilon_{xx}$", marker='o', color="k",  markersize=2, markerfacecolor='none')
    pl.plot(case_strain[:, 1], case_stress[:, 1], label=r"Ground truth $\varepsilon_{yy}$", marker='o', color="r",  markersize=2, markerfacecolor='none')
    pl.plot(case_strain[:, 2], case_stress[:, 2], label=r"Ground truth $\gamma_{xy}$",      marker='o', color="b",  markersize=2, markerfacecolor='none')
    pl.xlabel("Green-Lagrange Strain [-]")
    pl.ylabel("PK2 Stress [Pa]")

    pl.plot(case_strain[:, 0], stress_ann_prediction[:, 0], label=r"ANN prediction $\varepsilon_{xx}$" , color = "k", linestyle="--", linewidth = 1)
    pl.plot(case_strain[:, 1], stress_ann_prediction[:, 1], label=r"ANN prediction $\varepsilon_{yy}$" , color = "r", linestyle="--", linewidth = 1)
    pl.plot(case_strain[:, 2], stress_ann_prediction[:, 2], label=r"ANN prediction $\gamma_{xy}$"      , color = "b", linestyle="--", linewidth = 1)
    pl.legend(loc='best')
    pl.savefig("ann_predictions/" + "E_S_case_" + str(case) + ".png", dpi=400, bbox_inches='tight')
    pl.close()