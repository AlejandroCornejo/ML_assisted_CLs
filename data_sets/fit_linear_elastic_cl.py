import numpy as np
import matplotlib.pyplot as pl
import scienceplots

# Let's load the data stored in the dir neo_hookean_hyperelastic_law/
number_cases = 325

name = "neo_hookean_hyperelastic_law/raw_data/E_S_data_case_"
strain = np.load(name + "1.npz")["strain_history"]
stress = np.load(name + "1.npz")["stress_history"]
for case in range(1, number_cases + 1):
    loaded_data = np.load(name + str(case) + ".npz")
    strain = np.vstack((strain, loaded_data["strain_history"]))
    stress = np.vstack((stress, loaded_data["stress_history"]))

# We compute the equivalent elastic constitutive matrix
C_aniso = stress.T @ np.linalg.pinv(strain.T)
C_aniso = 0.5*(C_aniso + C_aniso.T)

fitted_stress_prediction = strain @ C_aniso

# Let's print the predictions v.s. the truth
pl.style.use('science')
for case in range(1, number_cases + 1):
    loaded_data = np.load(name + str(case) + ".npz")
    case_strain = loaded_data["strain_history"]
    case_stress = loaded_data["stress_history"]
    steps = case_strain.shape[0]

    pl.plot(case_strain[:, 0], case_stress[:, 0], label=r"Ground truth $\varepsilon_{xx}$", marker='o', color="k",  markersize=2, markerfacecolor='none')
    pl.plot(case_strain[:, 1], case_stress[:, 1], label=r"Ground truth $\varepsilon_{yy}$", marker='o', color="r",  markersize=2, markerfacecolor='none')
    pl.plot(case_strain[:, 2], case_stress[:, 2], label=r"Ground truth $\gamma_{xy}$",      marker='o', color="b",  markersize=2, markerfacecolor='none')
    pl.xlabel("Green-Lagrange Strain [-]")
    pl.ylabel("PK2 Stress [Pa]")

    pl.plot(case_strain[:, 0], fitted_stress_prediction[(case - 1) * steps : (case) * steps, 0], label=r"ANN prediction $\varepsilon_{xx}$" , color = "k", linestyle="--", linewidth = 1)
    pl.plot(case_strain[:, 1], fitted_stress_prediction[(case - 1) * steps : (case) * steps, 1], label=r"ANN prediction $\varepsilon_{yy}$" , color = "r", linestyle="--", linewidth = 1)
    pl.plot(case_strain[:, 2], fitted_stress_prediction[(case - 1) * steps : (case) * steps, 2], label=r"ANN prediction $\gamma_{xy}$"      , color = "b", linestyle="--", linewidth = 1)
    pl.legend(loc='best')
    pl.savefig("elastic_fit_predictions/" + "E_S_case_" + str(case) + ".png", dpi=400, bbox_inches='tight')
    pl.close()