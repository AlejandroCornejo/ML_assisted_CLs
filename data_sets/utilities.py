import math
import numpy as np

# Ref: pg 102 Oliver Saracibar
def CalculateI1(StrainVector):
    return StrainVector[0] + StrainVector[1]

def CalculateI2(StrainVector, I1):
    return 0.5*(StrainVector[0]**2 + StrainVector[1]**2 + 0.5*StrainVector[2]**2 - I1**2)

def CalculateI3(StrainVector):
    return StrainVector[0]*StrainVector[1] - 0.25*StrainVector[2]**2

def CalculateJ2(I1, I2):
    return 0.5*(I1**2 + 2.0*I2)

def CalculateJ3(I1, I2, I3):
    return (I1**3 + 3.0*I1*I2 + 3.0*I3) / 3.0

def RotateStrainVector(angle, StrainVector):
    c = math.cos(math.radians(angle))
    s = math.sin(math.radians(angle))

    T = np.array([[c**2, s**2, s*c],
                   [s**2, c**2, -s*c],
                   [-2.0*s*c, 2*s*c, c**2-s**2]])
    return T @ StrainVector

# test...
# strain_vector = np.array([0.001, 0.0002, -0.0002]).T

# print("Original E = ", strain_vector)
# I1 = CalculateI1(strain_vector)
# I2 = CalculateI2(strain_vector, CalculateI1(strain_vector))
# I3 = CalculateI3(strain_vector)
# J2 = CalculateJ2(I1, I2)
# J3 = CalculateJ3(I1, I2, I3)
# print("I1 = ", I1)
# print("I2 = ", I2)
# print("I3 = ", I3)
# print("J2 = ", J2)
# print("J3 = ", J3)

# rotated_E = RotateStrainVector(22.5, strain_vector)
# print("Rotated E = ", rotated_E)
# I1 = CalculateI1(rotated_E)
# I2 = CalculateI2(rotated_E, I1)
# I3 = CalculateI3(rotated_E)
# J2 = CalculateJ2(I1, I2)
# J3 = CalculateJ3(I1, I2, I3)
# print("I1 = ", I1)
# print("I2 = ", I2)
# print("I3 = ", I3)
# print("J2 = ", J2)
# print("J3 = ", J3)

# OK