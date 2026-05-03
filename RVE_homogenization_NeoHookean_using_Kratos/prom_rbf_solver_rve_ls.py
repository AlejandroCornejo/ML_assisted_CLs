from prom_rbf_solver_rve import (
    LoadPromRbfModel as _LoadPromRbfModel,
    RunPromRbfBatchSimulation as _RunPromRbfBatchSimulation,
)


def LoadPromRbfModel(basis_dir="stage_2_pod_rve", rbf_data_dir="stage_7_rbf_data_ls"):
    return _LoadPromRbfModel(basis_dir=basis_dir, rbf_data_dir=rbf_data_dir)


def RunPromRbfBatchSimulation(*args, **kwargs):
    return _RunPromRbfBatchSimulation(*args, **kwargs)
