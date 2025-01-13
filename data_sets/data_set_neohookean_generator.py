
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.ConstitutiveLawsApplication    as CLA

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

def _set_cl_parameters(cl_options, F, detF, strain_vector, stress_vector,
                       constitutive_matrix, N, DN_DX, model_part, properties, geom):
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

def CalculateStressFromDeformationGradient(DeformationGradient):
    # The selected CL
    cl = CLA.HyperElasticPlaneStrain2DLaw() # Neo-Hookean

    current_model = KM.Model()
    mdpa = current_model.CreateModelPart("NeoHookean")

    # properties = mdpa.CreateProperties(1)
    properties = KM.Properties(1)
    properties.SetValue(KM.YOUNG_MODULUS, 210e9)
    properties.SetValue(KM.POISSON_RATIO, 0.3)
    properties.SetValue(KM.CONSTITUTIVE_LAW, cl)

    dimension  = cl.WorkingSpaceDimension()
    voigt_size = cl.GetStrainSize()

    [geom, nnodes] = CreateGeometry(mdpa, dimension)

    N = KM.Vector(nnodes)
    DN_DX = KM.Matrix(nnodes, dimension)

    cl.Check(properties, geom, mdpa.ProcessInfo)

    cl_options = KM.Flags()
    cl_options.Set(KM.ConstitutiveLaw.USE_ELEMENT_PROVIDED_STRAIN, False)
    cl_options.Set(KM.ConstitutiveLaw.COMPUTE_CONSTITUTIVE_TENSOR, True)
    cl_options.Set(KM.ConstitutiveLaw.COMPUTE_STRESS, True)

    # Define deformation gradient
    F = DeformationGradient
    detF = 1.0

    stress_vector = KM.Vector(voigt_size)
    strain_vector = KM.Vector(voigt_size)
    constitutive_matrix = KM.Matrix(voigt_size,voigt_size)

    # Let's initialize the values
    for i in range(0, voigt_size):
        stress_vector[i] = 0.0
        strain_vector[i] = 0.0
        for j in range(0, cl.GetStrainSize()):
            constitutive_matrix[i, j] = 0.0

    # Setting the parameters - note that a constitutive law may not need them all!
    cl_params = _set_cl_parameters(cl_options, F, detF, strain_vector, stress_vector, constitutive_matrix, N, DN_DX, mdpa, properties, geom)
    cl.InitializeMaterial(properties, geom, N)

    cl.InitializeMaterial(properties, geom, KM.Vector(voigt_size))
    cl.CalculateMaterialResponseCauchy(cl_params)
    cl.FinalizeMaterialResponseCauchy(cl_params)

    return strain_vector, stress_vector

#====================================================================================
#====================================================================================

'''
Definition of the problem to solve

inputs:
    dimension of the problem
    Deformation gradient matrix as input
'''
dimension = 2
F = KM.Matrix(dimension, dimension)

F[0, 0] = 1.0
F[1, 1] = 1.0

F[0, 1] = 0.0
F[1, 0] = 0.0

strain, stress = CalculateStressFromDeformationGradient(F)

print(strain)
print(stress)
