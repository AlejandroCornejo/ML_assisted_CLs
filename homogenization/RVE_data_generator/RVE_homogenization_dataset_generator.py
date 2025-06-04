
import KratosMultiphysics as KM
import KratosMultiphysics.analysis_stage as analysis_stage
import importlib
from KratosMultiphysics.StructuralMechanicsApplication import python_solvers_wrapper_structural as structural_solvers


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

        # TODO to be modified

    def RunSolutionLoop(self):
        print("RVE_homogenization_dataset_generator run solution loop started\n")
        while self.KeepAdvancingSolutionLoop():
            self.time = self._AdvanceTime()
            self.InitializeSolutionStep()
            self._GetSolver().Predict()
            is_converged = self._GetSolver().SolveSolutionStep()
            self.__CheckIfSolveSolutionStepReturnsAValue(is_converged)
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
        self._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.TIME] = self.time

        KratosMultiphysics.Logger.PrintInfo(self._GetSimulationName(), "Analysis -START- ")

    #### Internal functions ####
    def _CreateSolver(self):
        return structural_solvers.CreateSolver(self.model, self.project_parameters)

    def __CreateListOfProcesses(self):
        """This function creates the processes and the output-processes
        """
        order_processes_initialization = self._GetOrderOfProcessesInitialization()
        self._list_of_processes        = self._CreateProcesses("processes", order_processes_initialization)
        deprecated_output_processes    = self._CheckDeprecatedOutputProcesses(self._list_of_processes)
        order_processes_initialization = self._GetOrderOfOutputProcessesInitialization()
        self._list_of_output_processes = self._CreateProcesses("output_processes", order_processes_initialization)
        self._list_of_processes.extend(self._list_of_output_processes) # Adding the output processes to the regular processes
        self._list_of_output_processes.extend(deprecated_output_processes)

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
