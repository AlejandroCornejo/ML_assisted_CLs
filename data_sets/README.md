
Guide:

 * data_set_neohookean_E_generator.py
    - Generates a representative strain-stress dataset and stores the information in "data_sets/neo_hookean_hyperelastic_law/raw_data" in .npz format.
    - Then, the strain and stress stacks can be accessed via:
            loaded_data = np.load("name_of_file.npz")
            strain_steps = loaded_data["strain_history"][:number_of_steps, :]
            stress_steps = loaded_data["stress_history"][:number_of_steps, :]

 * fit_linear_elastic_cl.py
    - This file reads the data set and fits the elastic constitutive matrix C_aniso. Then it plots the 
    ground thuth data against the prediction of a linear elastic CL: stress = C_aniso @ strain

 * fit_quadratic_hyperelastic_cl.py
 