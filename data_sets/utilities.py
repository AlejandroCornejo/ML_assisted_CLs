import math

def CalculateI1(StrainVector):
    return StrainVector[0] + StrainVector[1]

def CalculateI2(StrainVector, I1):
    return 0.5*(StrainVector[0]**2 + StrainVector[1]**2 + 0.5*StrainVector[2]**2 - I1**2)

def CalculateI3(StrainVector):
    return StrainVector[0]*StrainVector[1] - 0.25*StrainVector[2]

def CalculateJ2(I1, I2):
    return 0.5*(I1**2 + 2*I2)

def CalculateJ3(I1, I2, I3):
    return (I1**3 + 3.0*I1*I2 + 3.0*I3) / 3.0