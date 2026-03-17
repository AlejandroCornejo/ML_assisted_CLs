#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.ConstitutiveLawsApplication    as CLA
import math
import os
from j2_plane_stress_plastic_strain_simo import VonMisesIsotropicPlasticityPlaneStress

#====================================================================================
# Helpers
#====================================================================================

def CreateGeometry(model_part, dim):
    """
    Simple 2D geometry just to satisfy the ConstitutiveLaw interface.
    We will use dim = 2 (Triangle2D3).
    """
    node1 = model_part.CreateNewNode(1, 0.0, 0.0, 0.0)
    node2 = model_part.CreateNewNode(2, 1.0, 0.0, 0.0)
    node3 = model_part.CreateNewNode(3, 0.0, 1.0, 0.0)

    if dim == 2:
        nnodes = 3
        geom = KM.Triangle2D3(node1, node2, node3)
    else:
        raise Exception("Only 2D is supported for J2 plane stress.")

    return [geom, nnodes]


def _set_cl_parameters(cl_options, F, detF,
                       strain_vector, stress_vector, constitutive_matrix,
                       N, DN_DX,
                       model_part, properties, geom):
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

    cl_params.CheckAllParameters()
    cl_params.CheckMechanicalVariables()
    cl_params.CheckShapeFunctions()
    return cl_params


def CalculateStressFromE(ConstitutiveLaw, Properties, Geometry, ModelPart, eps):
    """
    Given a strain vector eps (Voigt: [εxx, εyy, γxy]),
    call the Kratos constitutive law and return σ = [σxx, σyy, τxy].
    """
    cl = ConstitutiveLaw

    dimension  = cl.WorkingSpaceDimension()
    voigt_size = cl.GetStrainSize()

    # dummy shape functions (unused here)
    N     = KM.Vector()
    DN_DX = KM.Matrix()

    cl.Check(Properties, Geometry, ModelPart.ProcessInfo)

    cl_options = KM.Flags()
    cl_options.Set(KM.ConstitutiveLaw.USE_ELEMENT_PROVIDED_STRAIN, True)
    cl_options.Set(KM.ConstitutiveLaw.COMPUTE_CONSTITUTIVE_TENSOR, False)
    cl_options.Set(KM.ConstitutiveLaw.COMPUTE_STRESS, True)

    F    = KM.Matrix(dimension, dimension)
    detF = 1.0

    strain_vector       = eps
    stress_vector       = KM.Vector(voigt_size)
    constitutive_matrix = KM.Matrix(voigt_size, voigt_size)

    stress_vector.fill(0.0)
    constitutive_matrix.fill(0.0)
    F.fill(0.0)

    cl_params = _set_cl_parameters(
        cl_options,
        F, detF,
        strain_vector, stress_vector, constitutive_matrix,
        N, DN_DX,
        ModelPart, Properties, Geometry
    )

    # Small strain J2 is usually implemented in Cauchy
    cl.InitializeMaterialResponseCauchy(cl_params)
    cl.CalculateMaterialResponseCauchy(cl_params)
    cl.FinalizeMaterialResponseCauchy(cl_params)

    return cl_params.GetStressVector()


def ResetE(E):
    E[0] = 0.0
    E[1] = 0.0
    E[2] = 0.0

#====================================================================================
# MAIN SCRIPT (J2 plane stress)
#====================================================================================

base_folder = "j2_plane_stress_von_mises_law"
os.makedirs(base_folder, exist_ok=True)
os.makedirs(os.path.join(base_folder, "strain_stress_plots"), exist_ok=True)
os.makedirs(os.path.join(base_folder, "strain_histories_plots"), exist_ok=True)
os.makedirs(os.path.join(base_folder, "raw_data"), exist_ok=True)

# to store all final target strains (for the sphere)
strains_matrix = np.zeros((0, 3))

# simple log file
log_path = os.path.join(base_folder, "data_set.log")
with open(log_path, "w") as f:
    f.write("Dataset generated for SmallStrainIsotropicPlasticityPlaneStressVonMisesVonMises\n\n")

case_number  = 1
current_model = KM.Model()
model_part    = current_model.CreateModelPart("J2PlaneStressPoint")

# --- Use a prototype CL only to get dimension/voigt size
cl_proto   = CLA.SmallStrainIsotropicPlasticityPlaneStressVonMisesVonMises()
dimension  = cl_proto.WorkingSpaceDimension()
voigt_size = cl_proto.GetStrainSize()

# --- Material properties
properties = KM.Properties(1)

E  = 206.9e9
nu = 0.29
rho = 7850.0
thickness = 0.05
sigma_y = 0.24e9
Gf = 1.0e12
hardening_curve = 3

properties.SetValue(KM.YOUNG_MODULUS, E)
properties.SetValue(KM.POISSON_RATIO, nu)
properties.SetValue(KM.DENSITY, rho)
properties.SetValue(KM.THICKNESS, thickness)
properties.SetValue(KM.YIELD_STRESS, sigma_y)
properties.SetValue(KM.FRACTURE_ENERGY, Gf)
properties.SetValue(CLA.HARDENING_CURVE, hardening_curve)
# (this value is not actually used in our direct calls, but we keep it for completeness)
properties.SetValue(KM.CONSTITUTIVE_LAW, cl_proto)

# --- Minimal geometry
[geometry, nnodes] = CreateGeometry(model_part, dimension)

# --- Strain vector
eps = KM.Vector(voigt_size)
ResetE(eps)

# --- Loading parameters
angle_increment_deg = 20.0
n_steps             = 25
max_stretch_factor  = 0.005
theta               = 0.0
phi                 = 0.0

strain_history = np.zeros((n_steps, voigt_size))
stress_history = np.zeros((n_steps, voigt_size))

strain_history_ours = np.zeros((n_steps, voigt_size))
stress_history_ours = np.zeros((n_steps, voigt_size))

# ====================================================================================
# Load-case loop
# ====================================================================================
while theta <= 360.0:
    while phi <= 360.0:

        # Final target strain
        dExx = math.cos(math.radians(theta)) * max_stretch_factor
        dEyy = math.sin(math.radians(theta)) * math.cos(math.radians(phi)) * max_stretch_factor
        dExy = 2.0 * math.sin(math.radians(theta)) * math.sin(math.radians(phi)) * max_stretch_factor

        # store it for the sphere
        strains_matrix = np.vstack((strains_matrix,
                                    np.array([dExx, dEyy, dExy])))

        # log
        with open(log_path, "a") as f:
            f.write(f"CASE {case_number}\n")
            f.write(f"\tTheta = {theta}, Phi = {phi}\n")
            f.write(f"\tTarget strain = [{dExx}, {dEyy}, {dExy}]\n\n")

        # ------------------------------------------------------------------
        # NEW: fresh constitutive law instance PER CASE  (no history carry-over)
        # ------------------------------------------------------------------
        cl = CLA.SmallStrainIsotropicPlasticityPlaneStressVonMisesVonMises()
        cl.InitializeMaterial(properties, geometry, KM.Vector(nnodes))
        ResetE(eps)

        #############
        # match Kratos material
        my_j2_model = VonMisesIsotropicPlasticityPlaneStress(E, nu, sigma_y, H=0.0)
        #############

        for step in range(n_steps):
            # linear ramp from 0 to target strain
            alpha = step / (n_steps - 1)
            eps[0] = dExx * alpha
            eps[1] = dEyy * alpha
            eps[2] = dExy * alpha

            sigma = CalculateStressFromE(cl, properties, geometry, model_part, eps)

            # --- our J2 model stress
            eps_numpy = np.array([eps[0], eps[1], eps[2]])
            sigma_ours = my_j2_model.CalculateMaterialResponse(eps_numpy)

            # DEBUG PRINT
            # --- relative errors per component (avoid division by zero)
            #tiny = 1e-16
            #rel_err_xx = abs(sigma_ours[0] - sigma[0]) / max(abs(sigma[0]), tiny)
            #rel_err_yy = abs(sigma_ours[1] - sigma[1]) / max(abs(sigma[1]), tiny)
            #rel_err_xy = abs(sigma_ours[2] - sigma[2]) / max(abs(sigma[2]), tiny)
            
            #print(
            #    f"COMPARE | case={case_number} | step={step:02d} | "
            #    f"eps=[{eps_numpy[0]:+.6e}, {eps_numpy[1]:+.6e}, {eps_numpy[2]:+.6e}] | "
            #    f"KRATOS=[{sigma[0]:+.6e}, {sigma[1]:+.6e}, {sigma[2]:+.6e}] | "
            #    f"OURS=[{sigma_ours[0]:+.6e}, {sigma_ours[1]:+.6e}, {sigma_ours[2]:+.6e}] | "
            #    f"rel.err=[{rel_err_xx:.3e}, {rel_err_yy:.3e}, {rel_err_xy:.3e}]"
            #)

            stress_history_ours[step, :] = [sigma_ours[0], sigma_ours[1], sigma_ours[2]]

            strain_history[step, :] = [eps[0], eps[1], eps[2]]
            stress_history[step, :] = [sigma[0], sigma[1], sigma[2]]

        # =========================================================================
        # PLOTS
        # =========================================================================

        # 1) Stress–strain curves
        plot_name = os.path.join(base_folder, "strain_stress_plots",
                                 f"strain_stress_data_case_{case_number}.png")

        pl.figure()

        # --- Kratos stresses
        pl.plot(strain_history[:, 0], stress_history[:, 0], 'k-o',
                label=r"$\sigma_{xx}$ (Kratos)", markersize=3)
        pl.plot(strain_history[:, 1], stress_history[:, 1], 'r-o',
                label=r"$\sigma_{yy}$ (Kratos)", markersize=3)
        pl.plot(strain_history[:, 2], stress_history[:, 2], 'b-o',
                label=r"$\tau_{xy}$ (Kratos)", markersize=3)

        # --- Our model stresses
        pl.plot(strain_history[:, 0], stress_history_ours[:, 0], 'k--x',
                label=r"$\sigma_{xx}$ (Ours)", markersize=4)
        pl.plot(strain_history[:, 1], stress_history_ours[:, 1], 'r--x',
                label=r"$\sigma_{yy}$ (Ours)", markersize=4)
        pl.plot(strain_history[:, 2], stress_history_ours[:, 2], 'b--x',
                label=r"$\tau_{xy}$ (Ours)", markersize=4)

        pl.xlabel("Strain component")
        pl.ylabel("Stress [Pa]")
        pl.title(f"theta={theta}, phi={phi}")
        pl.grid(True)
        pl.legend()
        pl.tight_layout()
        pl.savefig(plot_name, dpi=300)
        pl.close()

        # 2) 3D Strain trajectory
        plot_name2 = os.path.join(base_folder, "strain_histories_plots",
                                  f"strain_history_case_{case_number}.png")

        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(strain_history[:, 0],
                strain_history[:, 1],
                strain_history[:, 2],
                'ko-')
        ax.set_xlabel(r"$\varepsilon_{xx}$")
        ax.set_ylabel(r"$\varepsilon_{yy}$")
        ax.set_zlabel(r"$\gamma_{xy}$")
        ax.set_title(f"theta={theta}, phi={phi}")
        pl.tight_layout()
        pl.savefig(plot_name2, dpi=300)
        pl.close()

        # =========================================================================
        # SAVE data
        # =========================================================================
        npz_name = os.path.join(
            base_folder, "raw_data",
            f"E_S_data_case_{case_number}.npz"
        )

        np.savez(npz_name,
                 strain_history=strain_history,
                 stress_history=stress_history)

        print(f"[CASE {case_number}] Saved: {npz_name}")

        # Next case
        case_number += 1
        phi += angle_increment_deg
        if theta == 0.0 or theta == 360.0:
            break

    theta += angle_increment_deg
    phi = 0.0

print("\nDataset generation complete.")

# ====================================================================================
# Plot the "strain sphere" of final targets
# ====================================================================================
sphere_path = os.path.join(base_folder, "strain_sphere.png")
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(strains_matrix[:, 0], strains_matrix[:, 1], strains_matrix[:, 2],
           c='green', marker='o', label="Strain Cases")
ax.set_xlabel(r"$\varepsilon_{xx}$")
ax.set_ylabel(r"$\varepsilon_{yy}$")
ax.set_zlabel(r"$\gamma_{xy}$")
ax.set_title("Strain sphere of all load cases")
pl.legend(loc='best')
pl.tight_layout()
pl.savefig(sphere_path, dpi=300, bbox_inches=None)
pl.close()

print("Log written to:", log_path)
print("Strain-sphere plot written to:", sphere_path)
