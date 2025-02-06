import numpy as np
import matplotlib.pyplot as pl
import utilities as utils
import math

def AppendSelectedInvariants(my_list, I1, I2, I3, J2, J3, detF):
    my_list.append([
        I1,
        # I1**2,
        # I2,
        # I2**2,
        I3,
        # I3**2,
        # J2,
        # J2**2,
        # J3, # gives a kink
        # J3**2,
        # (detF - 1.0), # linear beh
        # (detF - 1.0)**2, # linear beh
        # math.log(detF)**2, # bad out
        # 1.0 / (detF - 1.0),
        # (detF - 1.0)**2
        # math.log(detF) # bad out
        (detF) # linear beh
    ])
    # my_list = np.log(my_list)

# Let's load the data stored in the dir neo_hookean_hyperelastic_law/
number_cases    = 325
initial_step    = 1
number_of_steps = 25

# This is the flatted vector outer_prod(E, E)

name = "neo_hookean_hyperelastic_law/raw_data/E_S_data_case_"
strain = np.load(name + "1.npz")["strain_history"][initial_step:number_of_steps:1, :]
stress = np.load(name + "1.npz")["stress_history"][initial_step:number_of_steps:1, :]

for case in range(1, number_cases + 1):
    loaded_data = np.load(name + str(case) + ".npz")
    strain = np.vstack((strain, loaded_data["strain_history"][initial_step:number_of_steps:1, :]))
    stress = np.vstack((stress, loaded_data["stress_history"][initial_step:number_of_steps:1, :]))

# Let's fill the gamma including the invariants
gamma_list = []
for row in strain:
    detF = utils.CalculateJacobian(row)
    # detF = 1.0
    # row = 2.0 * row + np.array([1.0, 1.0, 0.0])
    I1 = utils.CalculateI1(row)
    I2 = utils.CalculateI2(row, I1)
    I3 = utils.CalculateI3(row)
    J2 = utils.CalculateJ2(I1, I2)
    J3 = utils.CalculateJ3(I1, I2, I3)
    # print("I1 = ", I1)
    # print("I2 = ", I2)
    # print("I3 = ", I3)
    AppendSelectedInvariants(gamma_list, I1, I2, I3, J2, J3, detF)

# Convert the list to a NumPy array
gamma = np.array(gamma_list)

# Computed elsewhere
C_aniso =  np.array([[7.16042643e+06, 4.27997603e+02, -2.18278728e-11],
            [4.27997603e+02,  7.14548324e+06,  7.09405867e-11],
            [-2.18278728e-11,  7.09405867e-11,  3.57285779e+06]])

# TODO REMOVE
# C_aniso.fill(0.0)

delta_stress = stress.T - C_aniso @ strain.T
H_matrix = (delta_stress) @ np.linalg.pinv(gamma).T
# H_matrix = np.log(delta_stress) @ np.linalg.pinv(gamma).T

for case in range(1, number_cases + 1):
    loaded_data = np.load(name + str(case) + ".npz")
    case_strain = loaded_data["strain_history"][initial_step:number_of_steps:1, :]
    case_stress = loaded_data["stress_history"][initial_step:number_of_steps:1, :]

    gamma_case_list = []
    for row in case_strain:
        detF = utils.CalculateJacobian(row)
        # detF = 1.0
        # row = 2.0 * row + np.array([1.0, 1.0, 0.0])
        I1 = utils.CalculateI1(row)
        I2 = utils.CalculateI2(row, I1)
        I3 = utils.CalculateI3(row)
        J2 = utils.CalculateJ2(I1, I2)
        J3 = utils.CalculateJ3(I1, I2, I3)
        AppendSelectedInvariants(gamma_case_list, I1, I2, I3, J2, J3, detF)

    gamma_case = np.array(gamma_case_list)

    print("\t printing case: " + str(case))
    pl.plot(case_strain[:, 0], case_stress[:, 0], label=r"Ground truth $\varepsilon_{xx}$", marker='o', color="k",  markersize=2, markerfacecolor='none')
    pl.plot(case_strain[:, 1], case_stress[:, 1], label=r"Ground truth $\varepsilon_{yy}$", marker='o', color="r",  markersize=2, markerfacecolor='none')
    pl.plot(case_strain[:, 2], case_stress[:, 2], label=r"Ground truth $\gamma_{xy}$",      marker='o', color="b",  markersize=2, markerfacecolor='none')
    pl.xlabel("Green-Lagrange Strain [-]")
    pl.ylabel("PK2 Stress [Pa]")

    predicted_stress = (C_aniso @ case_strain.T).T + (H_matrix @ gamma_case.T).T
    # predicted_stress = np.exp((C_aniso @ case_strain.T).T + (H_matrix @ gamma_case.T).T)
    pl.plot(case_strain[:, 0], predicted_stress[:, 0], label=r"Quadratic elastic prediction $\varepsilon_{xx}$" , color = "k", linestyle="--", linewidth = 1)
    pl.plot(case_strain[:, 1], predicted_stress[:, 1], label=r"Quadratic elastic prediction $\varepsilon_{yy}$" , color = "r", linestyle="--", linewidth = 1)
    pl.plot(case_strain[:, 2], predicted_stress[:, 2], label=r"Quadratic elastic prediction $\gamma_{xy}$"      , color = "b", linestyle="--", linewidth = 1)

    pl.legend(loc='best', fontsize='small')
    pl.savefig("invariants_fit_predictions/" + "E_S_case_" + str(case) + ".png", dpi=300, bbox_inches='tight')
    pl.close()