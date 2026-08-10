import importlib

import numpy as np
import KratosMultiphysics
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
    StructuralMechanicsAnalysis,
)

with open("ProjectParameters.json", "r") as f:
    parameters = KratosMultiphysics.Parameters(f.read())

global_model = KratosMultiphysics.Model()
simulation = StructuralMechanicsAnalysis(global_model, parameters)
simulation.Run()

model_part = global_model.GetModelPart("Structure")

e11_all, e22_all, e12_all = [], [], []
for element in model_part.Elements:
    strain_vectors = element.CalculateOnIntegrationPoints(
        KratosMultiphysics.GREEN_LAGRANGE_STRAIN_VECTOR, model_part.ProcessInfo
    )
    for sv in strain_vectors:
        e11_all.append(sv[0])
        e22_all.append(sv[1])
        e12_all.append(sv[2])

e11_all = np.array(e11_all)
e22_all = np.array(e22_all)
e12_all = np.array(e12_all)

print(f"n_gauss_points = {len(e11_all)}")
print(f"E11 range: [{e11_all.min():.4f}, {e11_all.max():.4f}]")
print(f"E22 range: [{e22_all.min():.4f}, {e22_all.max():.4f}]")
print(f"E12 range: [{e12_all.min():.4f}, {e12_all.max():.4f}]")
print(f"gamma12=2E12 range: [{2*e12_all.min():.4f}, {2*e12_all.max():.4f}]")

# tip displacement (max Y among RightEdge nodes) for a physical sanity check
right_edge = model_part.GetSubModelPart("RightEdge")
uy = [node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y) for node in right_edge.Nodes]
print(f"RightEdge uy range: [{min(uy):.4f}, {max(uy):.4f}]")
