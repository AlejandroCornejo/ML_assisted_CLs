import numpy as np
import matplotlib.pyplot as plt

class VonMisesIsotropicPlasticityPlaneStress:
    """
    Class for plane stress isotropic plasticity with Von Mises yield criterion.

    pg 371 Souza et al.
    """

    def __init__(self, E, nu, sigma_y, G, hardening=0):
        self.dimension = 2
        self.voigt_size = 3

        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio
        self.G = G # fracture energy
        self.HardeningCurve = hardening # 0 is exponential softening

        # historical variables
        self.PlasticStrain = np.zeros(self.voigt_size)
        self.Threshold = sigma_y  # Yield stress
        self.PlasticDissipation = 0.0

    def GetPMatrix(self):
        return np.array([[2.0, -1.0, 0.0],
                     [-1.0, 2.0, 0.0],
                     [0.0 ,0.0, 6.0]]) / 3.0

    def CalculateConstitutiveMatrix(self):
        """
        Calculate the constitutive matrix for plane stress conditions.
        """
        E = self.E
        nu = self.nu

        # Plane stress constitutive matrix
        C = np.zeros((self.voigt_size, self.voigt_size))
        aux = E / (1 - nu**2)
        C[0, 0] = aux
        C[0, 1] = E * nu / (1 - nu**2)
        C[1, 0] = C[0, 1]
        C[1, 1] = aux
        C[2, 2] = self.G

        return C

    def CalculateEquivalentStress(self, stress):
        P = self.GetPMatrix()
        return np.sqrt(1.5 * np.dot(stress, np.dot(P, stress)))


    def CalculateYieldCondition(self, stress):
        return self.CalculateEquivalentStress(stress) - self.CalculateStressThreshold()

    def MacaulayBracket(self, value):
        """
        Macaulay bracket function, returns value if positive, else zero.
        """
        return max(value, 0.0)

    def UpdatePlasticDissipation(self, plastic_strain_increment, stress):
        """
        Update the plastic dissipation based on the plastic strain increment and current stress.
        """
        # Placeholder for actual implementation
        self.PlasticDissipation += self.MacaulayBracket(np.dot(stress, plastic_strain_increment)) / self.G

    def CalculateStressThreshold(self):
        return self.Threshold * (1.0 - self.PlasticDissipation)
        # return self.Threshold

    def Calculate_dThreshold_dKappap(self):
        return -self.Threshold
        # return 0.0

    def CalculatePlasticFlow(self, stress):
        """
        Calculate the plastic flow direction based on the current stress state.
        """
        P = self.GetPMatrix()
        return np.dot(P, stress) / np.sqrt(np.dot(stress, np.dot(P, stress))) * np.sqrt(1.5)

    def CalculatePlasticMultiplier(self, F, stress, D, PlasticFlow, dThreshold_dKappap):
        return F / (np.dot(PlasticFlow, np.dot(D, PlasticFlow)) + dThreshold_dKappap * np.dot(PlasticFlow, stress) / self.G)

    def CalculateMaterialResponse(self, strain):
        D = self.CalculateConstitutiveMatrix()
        predictive_stress = np.dot(D, strain - self.PlasticStrain)


        F = self.CalculateYieldCondition(predictive_stress)
        if F <= 0:
            # Elastic response
            return predictive_stress
        else:
            tolerance = 1e-10
            max_iterations = 100

            iteration = 0
            while F > tolerance and iteration < max_iterations:
                iteration += 1

                # Calculate plastic flow direction
                PlasticFlow = self.CalculatePlasticFlow(predictive_stress)

                # Calculate the derivative of the yield condition with respect to plastic strain
                dThreshold_dKappap = self.Calculate_dThreshold_dKappap()

                # Calculate the plastic multiplier
                plastic_multiplier = self.CalculatePlasticMultiplier(F, predictive_stress, D, PlasticFlow, dThreshold_dKappap)

                if plastic_multiplier < 0:
                    print("Warning: Negative plastic multiplier encountered. Adjusting to zero.")
                    plastic_multiplier = 0.0

                plastic_strain_increment = plastic_multiplier * PlasticFlow

                # Update the plastic strain
                self.PlasticStrain += plastic_strain_increment

                # Update the stress state
                predictive_stress -= np.dot(D, plastic_strain_increment)

                self.UpdatePlasticDissipation(plastic_strain_increment, predictive_stress)

                # Update the yield condition
                F = self.CalculateYieldCondition(predictive_stress)

                # Check to see if the plane stress condition is satisfied when plastic strain is updated
                strain_z = -self.nu / self.E * (predictive_stress[0] + predictive_stress[1])
                elastic_strain = strain - self.PlasticStrain
                stress_z = self.E * (1-self.nu) / (1+self.nu) / (1-2*self.nu) * (strain_z + self.nu / (1-self.nu) * (elastic_strain[0] + elastic_strain[1]))
                print("stress_z= ", stress_z)

            print(f"Converged in {iteration} iterations.")

            if iteration >= max_iterations:
                print("Warning: Maximum iterations reached without convergence.")

            return predictive_stress



'''
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
n_steps = 100
strain_history = np.zeros((3, n_steps))
stress_history = np.zeros((3, n_steps))
equivalent_stress = np.zeros(n_steps)

constitutive_law = VonMisesIsotropicPlasticityPlaneStress(
    E=210e9,  # Young's modulus in Pascals
    nu=0.3,  # Poisson's ratio
    sigma_y=500e6,  # Yield stress in Pascals
    G=1e7,  # Fracture energy in Pascals
    hardening=0  # Hardening parameter (0 for no hardening)
)

# Simulate strain history
step = 0
linspace = np.linspace(0, 14e-3, n_steps)  # Example linear strain increment

for step in range(n_steps):
    step += 1

    strain_x  = 1.0 * linspace[step - 1]  # Example strain in x-direction
    strain_y  = 0.0 * linspace[step - 1]  # Example strain in y-direction
    strain_xy = 0.0 * linspace[step - 1]  # No shear strain for plane stress

    step_strain = np.array([strain_x, strain_y, strain_xy])
    strain_history[:, step - 1] = step_strain

    step_stress = constitutive_law.CalculateMaterialResponse(step_strain)
    stress_history[:, step - 1] = step_stress
    equivalent_stress[step - 1] = constitutive_law.CalculateEquivalentStress(step_stress)

# Graficar Exx vs esfuerzo equivalente
plt.figure()
plt.plot(strain_history[0, :], equivalent_stress, marker='o')
plt.xlabel('Exx')
plt.ylabel('Eq. Stress [Pa]')
# plt.title('Exx vs Esfuerzo equivalente')
plt.grid(True)
plt.show()
'''