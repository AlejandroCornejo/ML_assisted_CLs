import numpy as np
import matplotlib.pyplot as pl

# Let's load the data stored in the dir neo_hookean_hyperelastic_law/
number_cases    = 325
number_of_steps = 25

# This is the flatted vector outer_prod(E, E)

name = "neo_hookean_hyperelastic_law/raw_data/E_S_data_case_"
strain = np.load(name + "1.npz")["strain_history"][:number_of_steps, :]
stress = np.load(name + "1.npz")["stress_history"][:number_of_steps, :]

for case in range(1, number_cases + 1):
    loaded_data = np.load(name + str(case) + ".npz")
    strain = np.vstack((strain, loaded_data["strain_history"][:number_of_steps, :]))
    stress = np.vstack((stress, loaded_data["stress_history"][:number_of_steps, :]))

# Let's fill the flatted gamma (gamma = flat(outer_prod(E,E)))
gamma_list = []
for row in strain:
    gamma_list.append([row[0]**2, row[1]**2, row[2]**2, row[0]*row[1], row[1]*row[2], row[0]*row[2]])

# Convert the list to a NumPy array
gamma = np.array(gamma_list)

# Computed elsewhere
C_aniso =  np.array([[7.16042643e+06, 4.27997603e+02, -2.18278728e-11],
            [4.27997603e+02,  7.14548324e+06,  7.09405867e-11],
            [-2.18278728e-11,  7.09405867e-11,  3.57285779e+06]])

delta_stress = stress.T - C_aniso @ strain.T
H_matrix = delta_stress @ np.linalg.pinv(gamma).T

for case in range(1, number_cases + 1):
    loaded_data = np.load(name + str(case) + ".npz")
    case_strain = loaded_data["strain_history"][:number_of_steps, :]
    case_stress = loaded_data["stress_history"][:number_of_steps, :]

    gamma_case_list = []
    for row in case_strain:
        gamma_case_list.append([row[0]**2, row[1]**2, row[2]**2, row[0]*row[1], row[1]*row[2], row[0]*row[2]])
    gamma_case = np.array(gamma_case_list)

    print("\t printing case: " + str(case))
    pl.plot(case_strain[:, 0], case_stress[:, 0], label=r"Ground truth $\varepsilon_{xx}$", marker='o', color="k",  markersize=2, markerfacecolor='none')
    pl.plot(case_strain[:, 1], case_stress[:, 1], label=r"Ground truth $\varepsilon_{yy}$", marker='o', color="r",  markersize=2, markerfacecolor='none')
    pl.plot(case_strain[:, 2], case_stress[:, 2], label=r"Ground truth $\gamma_{xy}$",      marker='o', color="b",  markersize=2, markerfacecolor='none')
    pl.xlabel("Green-Lagrange Strain [-]")
    pl.ylabel("PK2 Stress [Pa]")

    predicted_stress = (C_aniso @ case_strain.T).T + (H_matrix @ gamma_case.T).T
    pl.plot(case_strain[:, 0], predicted_stress[:, 0], label=r"Quadratic elastic prediction $\varepsilon_{xx}$" , color = "k", linestyle="--", linewidth = 1)
    pl.plot(case_strain[:, 1], predicted_stress[:, 1], label=r"Quadratic elastic prediction $\varepsilon_{yy}$" , color = "r", linestyle="--", linewidth = 1)
    pl.plot(case_strain[:, 2], predicted_stress[:, 2], label=r"Quadratic elastic prediction $\gamma_{xy}$"      , color = "b", linestyle="--", linewidth = 1)

    pl.legend(loc='best', fontsize='small')
    pl.savefig("quadratic_fit_predictions/" + "E_S_case_" + str(case) + ".png", dpi=300, bbox_inches='tight')
    pl.close()