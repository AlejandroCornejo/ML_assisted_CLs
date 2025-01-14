import numpy as np
import matplotlib.pyplot as pl
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.ConstitutiveLawsApplication    as CLA
import json
import math

#====================================================================================
#====================================================================================

def CreateGeometry(model_part, dim):
    # Create new nodes
    node1 = model_part.CreateNewNode(1, 0.0, 0.0, 0.0)
    node2 = model_part.CreateNewNode(2, 1.0, 0.0, 0.0)
    node3 = model_part.CreateNewNode(3, 0.0, 1.0, 0.0)

    if (dim == 2):
        nnodes = 3

        # Allocate a geometry
        geom = KM.Triangle2D3(node1,node2,node3)
    elif (dim == 3):
        nnodes = 4
        node4 = model_part.CreateNewNode(4, 0.0, 0.0, 1.0)

        # Allocate a geometry
        geom = KM.Tetrahedra3D4(node1,node2,node3,node4)
    else:
        raise Exception("Error: bad dimension value: ", dim)
    return [geom, nnodes]

#====================================================================================
#====================================================================================

def _set_cl_parameters(cl_options, F, detF, strain_vector, stress_vector, constitutive_matrix, N, DN_DX, model_part, properties, geom):
    # Setting the parameters - note that a constitutive law may not need them all!
    cl_params = KM.ConstitutiveLawParameters()
    cl_params.SetOptions(cl_options)
    cl_params.SetDeformationGradientF(F)
    cl_params.SetDeterminantF(detF)
    cl_params.SetStrainVector(strain_vector)
    cl_params.SetStressVector(stress_vector)
    cl_params.SetConstitutiveMatrix(constitutive_matrix)
    cl_params.SetShapeFunctionsValues(N)
    cl_params.SetShapeFunctionsDerivatives(DN_DX)
    cl_params.SetProcessInfo(model_part.ProcessInfo)
    cl_params.SetMaterialProperties(properties)
    cl_params.SetElementGeometry(geom)

    ## Do all sort of checks
    cl_params.CheckAllParameters() # Can not use this until the geometry is correctly exported to python
    cl_params.CheckMechanicalVariables()
    cl_params.CheckShapeFunctions()
    return cl_params

#====================================================================================
#====================================================================================

def CalculateStressFromDeformationGradient(ConstitutiveLaw, Properties, Geometry, ModelPart, DeformationGradient):
    # The selected CL
    cl = ConstitutiveLaw

    dimension  = cl.WorkingSpaceDimension()
    voigt_size = cl.GetStrainSize()

    N = KM.Vector()
    DN_DX = KM.Matrix()

    cl.Check(Properties, Geometry, ModelPart.ProcessInfo)

    cl_options = KM.Flags()
    cl_options.Set(KM.ConstitutiveLaw.USE_ELEMENT_PROVIDED_STRAIN, False)
    cl_options.Set(KM.ConstitutiveLaw.COMPUTE_CONSTITUTIVE_TENSOR, False)
    cl_options.Set(KM.ConstitutiveLaw.COMPUTE_STRESS, True)

    # Define deformation gradient
    F = DeformationGradient
    detF = 1.0

    stress_vector = KM.Vector(voigt_size)
    strain_vector = KM.Vector(voigt_size)
    constitutive_matrix = KM.Matrix(voigt_size,voigt_size)

    # Let's initialize the values
    stress_vector.fill(0.0)
    strain_vector.fill(0.0)
    constitutive_matrix.fill(0.0)
    # s = np.array(stress_vector, copy=False)

    # Setting the parameters - note that a constitutive law may not need them all!
    cl_params = _set_cl_parameters(cl_options, F, detF, strain_vector, stress_vector, constitutive_matrix, N, DN_DX, ModelPart, Properties, Geometry)

    # We run them in case we need to account for history
    cl.InitializeMaterialResponsePK2(cl_params)
    cl.CalculateMaterialResponsePK2(cl_params)
    cl.FinalizeMaterialResponsePK2(cl_params)

    # Let's compute the strain and stress output
    strain_calculate_value = KM.Vector()
    stress_calculate_value = KM.Vector()
    strain_measure_variable = KM.GREEN_LAGRANGE_STRAIN_VECTOR # ALMANSI_STRAIN_VECTOR HENCKY_STRAIN_VECTOR GREEN_LAGRANGE_STRAIN_VECTOR
    stress_measure_variable = KM.PK2_STRESS_VECTOR            # CAUCHY_STRESS_VECTOR  KIRCHHOFF_STRESS_VECTOR PK2_STRESS_VECTOR

    cl.CalculateValue(cl_params, strain_measure_variable, strain_calculate_value)
    cl.CalculateValue(cl_params, stress_measure_variable, stress_calculate_value)

    # return strain_vector, stress_vector
    return strain_calculate_value, stress_calculate_value

#====================================================================================
#====================================================================================

def PredictStrainNormFromF(ConstitutiveLaw, Properties, Geometry, ModelPart, DeformationGradient):
    # The selected CL
    cl = ConstitutiveLaw

    dimension  = cl.WorkingSpaceDimension()
    voigt_size = cl.GetStrainSize()

    N = KM.Vector()
    DN_DX = KM.Matrix()

    cl.Check(Properties, Geometry, ModelPart.ProcessInfo)

    cl_options = KM.Flags()
    cl_options.Set(KM.ConstitutiveLaw.USE_ELEMENT_PROVIDED_STRAIN, False)
    cl_options.Set(KM.ConstitutiveLaw.COMPUTE_CONSTITUTIVE_TENSOR, False)
    cl_options.Set(KM.ConstitutiveLaw.COMPUTE_STRESS, True)

    # Define deformation gradient
    F = KM.Matrix(dimension, dimension)
    F[0,0] = DeformationGradient[0,0]
    F[0,1] = DeformationGradient[0,1]
    F[1,0] = DeformationGradient[1,0]
    F[1,1] = DeformationGradient[1,1]
    detF = 1.0

    stress_vector = KM.Vector(voigt_size)
    strain_vector = KM.Vector(voigt_size)
    constitutive_matrix = KM.Matrix(voigt_size,voigt_size)

    # Let's initialize the values
    strain_vector.fill(0.0)

    # Setting the parameters - note that a constitutive law may not need them all!
    cl_params = _set_cl_parameters(cl_options, DeformationGradient, detF, strain_vector, stress_vector, constitutive_matrix, N, DN_DX, ModelPart, Properties, Geometry)

    # Let's compute the strain and stress output
    strain_calculate_value = KM.Vector(voigt_size)
    strain_measure_variable = KM.GREEN_LAGRANGE_STRAIN_VECTOR # ALMANSI_STRAIN_VECTOR HENCKY_STRAIN_VECTOR GREEN_LAGRANGE_STRAIN_VECTOR

    cl.CalculateValue(cl_params, strain_measure_variable, strain_calculate_value)

    # return strain_vector, stress_vector
    return np.linalg.norm(strain_calculate_value)

#====================================================================================
#====================================================================================

def ResetF(F):
    F[0, 0] = 1.0
    F[0, 1] = 0.0
    F[1, 0] = 0.0
    F[1, 1] = 1.0

#====================================================================================
#====================================================================================


'''
Definition of the problem to solve

inputs:
    dimension of the problem
    Deformation gradient matrix as input

    x = Phi(X, t) --> x: Updated coordinates
                      Phi: Mapping operator
                      X: Initial coordinates
                      t: Time

    F = Grad(Phi) --> F = dx_i / dX_j

    Notation: x: Spatial  coords
              X: Material coords

             x_2, v
            ^
            |
      O-----------o
      |     |     |
      |     |     |
      |     +---- | --> x_1, u
      |           |
      |           |
      O-----------O
'''

#====================================================================================
#====================================================================================

case_number = 1
current_model = KM.Model()
model_part = current_model.CreateModelPart("NeoHookean")

cl = CLA.HyperElasticPlaneStrain2DLaw()
dimension = cl.WorkingSpaceDimension()
voigt_size = cl.GetStrainSize()

properties = KM.Properties(1)
E = 1e7
nu = 0.4
properties.SetValue(KM.YOUNG_MODULUS, 1e7)
properties.SetValue(KM.POISSON_RATIO, nu)
properties.SetValue(KM.CONSTITUTIVE_LAW, cl)

[geometry, nnodes] = CreateGeometry(model_part, dimension)

# Initialize the F
F = KM.Matrix(dimension, dimension)
F[0, 0] = 1.0
F[0, 1] = 0.0
F[1, 0] = 0.0
F[1, 1] = 1.0

# Initialize the material
cl.InitializeMaterial(properties, geometry, KM.Vector(nnodes))

# ----- Start the case loop
angle_increment_deg = 10.0
n_steps             = 25 # step in the loading history
max_stretch_factor  = 0.25 # lambda
theta               = 0.0
phi                 = 0.0
# -----

with open("neo_hookean_hyperelastic_law/data_set.log", "w") as file:
    file.write("Dataset generated for a HyperElasticPlaneStrain2DLaw constitutive law \n \n")
    file.write("Data used for the generation: \n")
    file.write("\tYoung's Modulus = " + str(E) + "\n")
    file.write("\tPoisson's ratio = " + str(nu) + "\n")
    file.write("\tAngle increment = " + str(angle_increment_deg) + " deg \n")
    file.write("\tNumber of steps per load history = " + str(n_steps) + "\n")

strain_history = np.zeros((n_steps, voigt_size))
stress_history = strain_history.copy()

while theta <= 360.0 and phi <= 360.0:
    # This is the dF such that F = Id + dF
    dF11 = math.cos(math.radians(theta)) * max_stretch_factor
    dF12 = 2.0 * math.sin(math.radians(theta)) * math.sin(math.radians(phi)) * max_stretch_factor
    dF21 = dF12
    dF22 = 2.0 * math.sin(math.radians(theta)) * math.cos(math.radians(phi)) * max_stretch_factor

    # We check how much is the strain, hence we correct to maintain the norm of E
    F_trial = KM.Matrix(dimension, dimension)
    F_trial[0, 0] = 1.0 + dF11
    F_trial[0, 1] = dF12
    F_trial[1, 0] = F_trial[0, 1]
    F_trial[1, 1] = 1.0 + dF22
    predicted_strain_norm = PredictStrainNormFromF(cl, properties, geometry, model_part, F_trial)

    for step in range(n_steps):
        # We increment the dF as: Fn = Id + alpha*dF
        alpha = (step + 1) / n_steps * max_stretch_factor / predicted_strain_norm
        F[0, 0] = 1.0 + dF11 * alpha
        F[0, 1] = dF12 * alpha
        F[1, 0] = F[0, 1]
        F[1, 1] = 1.0 + dF22 * alpha

        strain, stress = CalculateStressFromDeformationGradient(cl, properties, geometry, model_part, F)
        strain_history[step, :] = strain
        stress_history[step, :] = stress

    output_type = "plot" # print plot

    if output_type == "plot":
        name = "neo_hookean_hyperelastic_law/strain_stress_data_case_" + str(case_number) + ".png"
        title = "Theta = " + str(theta) + " ; Phi = " + str(phi)
        pl.plot(strain_history[:, 0], stress_history[:, 0], label="Ground truth XX", marker='X', color="k",  markersize=5, markerfacecolor='none')
        pl.plot(strain_history[:, 1], stress_history[:, 1], label="Ground truth YY", marker='X', color="r",  markersize=5, markerfacecolor='none')
        pl.plot(strain_history[:, 2], stress_history[:, 2], label="Ground truth XY", marker='X', color="b",  markersize=5, markerfacecolor='none')
        pl.xlabel("Strain [-]")
        pl.ylabel("Stress [Pa]")
        pl.title(title)
        pl.legend(loc='best')
        pl.grid()
        # pl.show()
        pl.savefig(name, dpi=600, bbox_inches='tight')
        pl.close()

        name = "neo_hookean_hyperelastic_law/strain_history_case_" + str(case_number) + ".png"
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(strain_history[:, 0], strain_history[:, 1], strain_history[:, 2], c='r', marker='o', label="Strain history")
        ax.set_xlabel('Strain XX')
        ax.set_ylabel('Strain YY')
        ax.set_zlabel('Strain XY')
        pl.title(title)
        pl.savefig(name, dpi=600, bbox_inches='tight')
        pl.close()
        # pl.show()

    with open("neo_hookean_hyperelastic_law/data_set.log", "a") as file:
        file.write("CASE " + str(case_number) + "\n")
        file.write("\tImposed F = " + str(F) + "\n")
        file.write("\tImposed E = " + str(strain_history[n_steps-1, :]) + "\n")
        file.write("\tTheta = " + str(theta) + ", Phi = " + str(phi) + "\n")
        file.write("\tNorm of E = " + str(np.linalg.norm(strain_history[n_steps-1, :])) + "\n\n")

    name = "neo_hookean_hyperelastic_law/E_S_data_case_" + str(case_number) + ".npz"
    np.savez(name, strain_history = strain_history, stress_history = stress_history)
    print("\t --> Case: ", case_number, "Data saved to ", name)

    '''
    NOTE:
    Then we can load them by

    loaded_data = np.load(name)

    loaded_strain_history = loaded_data["strain_history"]
    loaded_stress_history = loaded_data["stress_history"]
    '''
    theta += angle_increment_deg
    phi   += angle_increment_deg
    case_number += 1
    ResetF(F)


#====================================================================================
#====================================================================================
