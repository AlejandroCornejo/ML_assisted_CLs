
import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
import importlib
from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural as structural_solvers
import KratosMultiphysics.StructuralMechanicsApplication as SMApp
import numpy as np

""""

"""

class RVE_homogenization_dataset_generator(analysis_stage.AnalysisStage):

    def __init__(self, model, project_parameters):
        print("RVE_homogenization_dataset_generator initialized\n")
        super().__init__(model, project_parameters)

    def Run(self):
        print("RVE_homogenization_dataset_generator run started\n")
        self.Initialize()
        self.RunSolutionLoop()
        self.Finalize()

    def RunSolutionLoop(self):
        print("RVE_homogenization_dataset_generator run solution loop started\n")
        while self.KeepAdvancingSolutionLoop():
            self.time = self._AdvanceTime()
            self.InitializeSolutionStep()
            self._GetSolver().Predict()
            is_converged = self._GetSolver().SolveSolutionStep()
            self.FinalizeSolutionStep()
            self.OutputSolutionStep()

    def Initialize(self):

        self._GetSolver().ImportModelPart()
        self._GetSolver().PrepareModelPart()
        self._GetSolver().AddDofs()

        ##here we initialize user-provided processes
        self.__CreateListOfProcesses() # has to be done after importing and preparing the ModelPart
        for process in self._GetListOfProcesses():
            process.ExecuteInitialize()

        self._GetSolver().Initialize()
        self.Check()

        for process in self._GetListOfProcesses():
            process.ExecuteBeforeSolutionLoop()

        ## Stepping and time settings
        self.end_time = self.project_parameters["problem_data"]["end_time"].GetDouble()

        self.time = self.project_parameters["problem_data"]["start_time"].GetDouble()
        self._GetSolver().GetComputingModelPart().ProcessInfo[KM.TIME] = self.time

        KM.Logger.PrintInfo(self._GetSimulationName(), "Analysis -START- ")

    # def FinalizeSolutionStep(self):
    #     super().FinalizeSolutionStep()

    def _CreateSolver(self):
        return structural_solvers.CreateSolver(self.model, self.project_parameters)

    def __CreateListOfProcesses(self):
        order_processes_initialization = self._GetOrderOfProcessesInitialization()
        self._list_of_processes        = self._CreateProcesses("processes", order_processes_initialization)
        deprecated_output_processes    = self._CheckDeprecatedOutputProcesses(self._list_of_processes)
        order_processes_initialization = self._GetOrderOfOutputProcessesInitialization()
        self._list_of_output_processes = self._CreateProcesses("output_processes", order_processes_initialization)
        self._list_of_processes.extend(self._list_of_output_processes) # Adding the output processes to the regular processes
        self._list_of_output_processes.extend(deprecated_output_processes)

    def ApplyBoundaryConditions(self):
        super().ApplyBoundaryConditions()

        for node in self._GetSolver().GetComputingModelPart().Nodes:
            x_coord = node.X0
            y_coord = node.Y0
            z_coord = node.Z0
            displ_x = 1.0e-3 * x_coord
            displ_y = 2.0e-3 * y_coord
            displ_z = 1.0e-3 * z_coord
            if node.IsFixed(KM.DISPLACEMENT_X):
                node.SetSolutionStepValue(KM.DISPLACEMENT_X, displ_x)
            if node.IsFixed(KM.DISPLACEMENT_Y):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Y, displ_y)
            if node.IsFixed(KM.DISPLACEMENT_Z):
                node.SetSolutionStepValue(KM.DISPLACEMENT_Z, displ_z)


    def CalculateHomogenizedStressAndStrain(self):

        process_info = self._GetSolver().GetComputingModelPart().ProcessInfo
        computing_model_part = self._GetSolver().GetComputingModelPart()

        for element in computing_model_part.Elements:
            dummy_strain = np.array(element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info))
            break
        # NOTE: this assumes that all elements have the same number of integration points and the same strain vector size
        n_ips = dummy_strain.shape[0]
        voigt_size  = dummy_strain.shape[1]

        homogenized_stress = np.zeros(voigt_size)
        homogenized_srain  = np.zeros(voigt_size)
        RVE_area = 0.0

        for element in computing_model_part.Elements:
            strain = element.CalculateOnIntegrationPoints(KM.GREEN_LAGRANGE_STRAIN_VECTOR, process_info)
            stress = element.CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, process_info)

            stress_vector_sum_ip = np.sum(np.array(stress), axis=0)
            strain_vector_sum_ip = np.sum(np.array(strain), axis=0)

            element_area = element.GetGeometry().Area()

            RVE_area += element_area
            homogenized_stress += element_area * stress_vector_sum_ip / n_ips
            homogenized_srain  += element_area * strain_vector_sum_ip / n_ips

        homogenized_stress /= RVE_area
        homogenized_srain  /= RVE_area

        print("Homogenized stress: ", homogenized_stress)
        print("Homogenized strain: ", homogenized_srain)
        print("RVE_area: ", RVE_area)

        return homogenized_stress, homogenized_srain

#====================================================================================================
#====================================================================================================


with open("ProjectParameters.json", 'r') as parameter_file:
    parameters = KM.Parameters(parameter_file.read())

analysis_stage_module_name = parameters["analysis_stage"].GetString()
analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

analysis_stage_module = importlib.import_module(analysis_stage_module_name)
analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

global_model = KM.Model()
simulation = RVE_homogenization_dataset_generator(global_model, parameters)
simulation.Run()

simulation.CalculateHomogenizedStressAndStrain()
