import os
import sys
import time
import torch
import numpy as np
try:
    from torch.func import jacfwd as _torch_jacfwd
except Exception:
    _torch_jacfwd = None

# Add Kratos path
KRATOS_PATH = "/home/kratos/Kratos_Eigen_Check/bin/Release"
if KRATOS_PATH not in sys.path:
    sys.path.append(KRATOS_PATH)

import KratosMultiphysics as KM
from fom_solver_rve import VectorizedAssembler
from fom_solver_rve import (
    DeformationGradientFromGreenLagrange2D,
    RVEHomogenizationDatasetGenerator,
    SetUpDofEquationIdsAndDisplacementAdaptor,
    SetDisplacementFromEquationVector,
    UpdateCurrentCoordinatesFromDisplacement,
    AssembleGlobalSystem,
    InitializeNonLinearIteration,
    FinalizeNonLinearIteration,
    BuildDynamicSegmentSteps,
    CalculateHomogenizedFromAssemblerWithElementWeights,
    REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    MIN_STEPS_PER_SEGMENT,
    USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    NEWTON_TOL_ABS,
    DISP_TOL_ABS,
)
from stage7b_train_ann_manifold import ManifoldANN


def _load_optional_mu_affine_initializer(ann_data_dir, n_primary):
    qm_path = os.path.join(ann_data_dir, "qm_init_mu_affine.npz")
    init_path = qm_path if os.path.exists(qm_path) else os.path.join(
        ann_data_dir, "qp_init_mu_affine.npz"
    )
    if not os.path.exists(init_path):
        return None
    data = np.load(init_path, allow_pickle=True)
    if "b_aff" not in data:
        raise RuntimeError(f"[PROM-ANN] Invalid initializer file (missing b_aff): {init_path}")
    b_aff = np.asarray(data["b_aff"], dtype=float)
    qp_dim = int(np.ravel(data["qp_dim"])[0]) if "qp_dim" in data else int(b_aff.shape[1])
    mu_dim = int(np.ravel(data["mu_dim"])[0]) if "mu_dim" in data else int(b_aff.shape[0] - 1)
    if b_aff.ndim != 2 or b_aff.shape != (mu_dim + 1, qp_dim):
        raise RuntimeError(f"[PROM-ANN] Invalid affine initializer shape in {init_path}: {b_aff.shape}")
    if qp_dim != int(n_primary):
        raise RuntimeError(
            f"[PROM-ANN] q_m initializer dimension={qp_dim} incompatible with n_primary={n_primary}."
        )
    return {
        "path": init_path,
        "b_aff": b_aff,
        "mu_dim": mu_dim,
        "qp_dim": qp_dim,
    }


def LoadPromAnnModel(basis_dir="stage_2_pod_rve", ann_data_dir="stage_7_ann_data"):
    phi_m_path = os.path.join(ann_data_dir, "phi_m.npy")
    if not os.path.exists(phi_m_path):
        raise FileNotFoundError(
            f"{phi_m_path} is required by the Joaquin-style LS decoder. Rebuild Stage 7a/7b."
        )
    phi_p = np.load(phi_m_path)
    phi_s = np.load(os.path.join(ann_data_dir, "phi_s.npy"))
    free_dofs = np.load(os.path.join(basis_dir, "free_dofs.npy"))
    dir_dofs = np.load(os.path.join(basis_dir, "dirichlet_dofs.npy"))
    eq_map = np.load(os.path.join(basis_dir, "eq_map.npy"))
    
    # Load Metadata & ANN
    meta = np.load(os.path.join(ann_data_dir, "manifold_ann_metadata.npz"), allow_pickle=True)
    dataset_meta_path = os.path.join(ann_data_dir, "ann_dataset_metadata.npz")
    dataset_meta = np.load(dataset_meta_path, allow_pickle=True) if os.path.exists(dataset_meta_path) else None
    n_p = int(phi_p.shape[1])
    n_s = int(phi_s.shape[1])

    if "input_dim" in meta:
        input_dim = int(np.ravel(meta["input_dim"])[0])
    elif dataset_meta is not None and "input_dim" in dataset_meta:
        input_dim = int(np.ravel(dataset_meta["input_dim"])[0])
    else:
        input_dim = int(meta["x_mean"].shape[0])

    include_macro_strain_input = False
    if "include_macro_strain_input" in meta:
        include_macro_strain_input = bool(int(np.ravel(meta["include_macro_strain_input"])[0]))
    elif dataset_meta is not None and "include_macro_strain_input" in dataset_meta:
        include_macro_strain_input = bool(int(np.ravel(dataset_meta["include_macro_strain_input"])[0]))
    elif input_dim == (n_p + 3):
        include_macro_strain_input = True

    if include_macro_strain_input:
        raise RuntimeError(
            "ANN model was trained with macro-strain manifold inputs (N(q,mu)), "
            "which is no longer supported. Retrain Stage 7 without macro inputs."
        )
    if input_dim != n_p:
        raise ValueError(
            f"ANN input_dim={input_dim} incompatible with n_primary={n_p}."
        )

    if meta["x_mean"].shape[0] != input_dim or meta["y_mean"].shape[0] != n_s:
        raise ValueError(
            "ANN metadata dimensions do not match phi_p/phi_s partition "
            f"(x={meta['x_mean'].shape[0]}, expected_input={input_dim}, "
            f"y={meta['y_mean'].shape[0]}, n_s={n_s})."
        )
    a_m_path = os.path.join(ann_data_dir, "A_m.npy")
    if not os.path.exists(a_m_path):
        raise FileNotFoundError(
            f"{a_m_path} is required by the Joaquin-style LS decoder. Rebuild Stage 7a/7b."
        )
    a_m = np.asarray(np.load(a_m_path), dtype=float)
    if a_m.shape != (n_p, n_p):
        raise RuntimeError(f"A_m shape mismatch: got {a_m.shape}, expected {(n_p, n_p)}.")

    hidden_layers = (128, 128, 128, 128)
    if "hidden_layers" in meta:
        hidden_layers = tuple(int(v) for v in np.asarray(meta["hidden_layers"], dtype=np.int64).reshape(-1))
    activation = str(np.ravel(meta["activation"])[0]).strip().lower() if "activation" in meta else "silu"
    dropout = float(np.ravel(meta["dropout"])[0]) if "dropout" in meta else 0.0
    use_batchnorm = bool(int(np.ravel(meta["use_batchnorm"])[0])) if "use_batchnorm" in meta else True
    origin_anchored = (
        bool(int(np.ravel(meta["origin_anchored"])[0]))
        if "origin_anchored" in meta
        else False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ManifoldANN(
        meta["x_mean"], meta["x_std"], meta["y_mean"], meta["y_std"],
        in_dim=input_dim,
        out_dim=n_s,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        origin_anchored=origin_anchored,
    ).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(ann_data_dir, "manifold_ann.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    model.a_m_np = a_m
    model.qp_init_mu_affine = _load_optional_mu_affine_initializer(
        ann_data_dir,
        n_primary=n_p,
    )
    
    return phi_p, phi_s, free_dofs, dir_dofs, eq_map, model, device, False

def RunPromAnnBatchSimulation(
    parameters,
    phi_p,
    phi_s,
    free_dofs,
    ann_model,
    device,
    strain_path,
    out_dir=None,
    relnorm_cutoff=1e-5,
    max_its=25,
    abs_res_cutoff=NEWTON_TOL_ABS,
    dq_abs_cutoff=1.0e-6,
    normalized_dq_cutoff=1.0e-4,
    max_res_for_rel_convergence=1.0e-1,
    min_rel_drop_stop=1.0e-2,
    stagnation_relnorm_gate=1.0e-4,
    max_dq_norm=0.5,
    old_stiffness_residual_cutoff=1.0e5,
    regularization=1.0e-10,
    use_fast_dirichlet_bc=True,
    reference_amplitude=None,
    reference_steps=REFERENCE_STEPS_FOR_UNIT_AMPLITUDE,
    use_old_stiffness_in_first_iteration=USE_OLD_STIFFNESS_IN_FIRST_ITERATION,
    verbose_iterations=False,
    qp_init_mode="continuation",
    fail_on_nonconvergence=True,
):
    t_wall_total_start = time.perf_counter()

    max_its = int(max_its)
    if max_its < 0:
        raise ValueError(f"max_its must be >= 0, got {max_its}.")

    free_dofs = np.asarray(free_dofs, dtype=np.int64)
    if phi_p.shape[0] != phi_s.shape[0]:
        raise ValueError("phi_p and phi_s must have the same number of rows (n_free).")

    n_primary = int(phi_p.shape[1])
    n_secondary = int(phi_s.shape[1])

    dt = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()
    E_wp = np.array(strain_path, dtype=float)
    n_seg = len(E_wp) - 1
    seg_steps, _ = BuildDynamicSegmentSteps(
        E_wp,
        reference_steps=reference_steps,
        min_steps=MIN_STEPS_PER_SEGMENT,
        reference_amplitude=reference_amplitude,
    )
    step_offsets = np.concatenate(([0], np.cumsum(seg_steps)))
    n_steps_total = int(step_offsets[-1])
    end_time = dt * float(n_steps_total)

    model_kratos = KM.Model()
    sim = RVEHomogenizationDatasetGenerator(model_kratos, parameters)
    sim.Initialize()
    mp = sim._GetSolver().GetComputingModelPart()
    
    n_total_dof, eq_id_map, ta = SetUpDofEquationIdsAndDisplacementAdaptor(mp)
    vec_assembler = VectorizedAssembler(mp, n_total_dof, eq_id_map)
    hom_reference_measure = float(np.sum(np.asarray(vec_assembler.area_e, dtype=float)))
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "reference_measure_A0.txt"), "w", encoding="utf-8") as f:
            f.write(f"{hom_reference_measure:.16e}\n")
    elements = list(mp.Elements)
    entities = list(mp.Elements) + list(mp.Conditions)

    # Affine lifting helper for free DOFs: u_aff = (F-I)(X-Xc)
    sim._InitializeDomainCenterIfNeeded(mp)
    x0c, y0c = float(sim._x0c), float(sim._y0c)
    dof_x = np.zeros(n_total_dof, dtype=float)
    dof_y = np.zeros(n_total_dof, dtype=float)
    is_x_dof = np.zeros(n_total_dof, dtype=bool)
    for i, node in enumerate(mp.Nodes):
        xr = float(node.X0) - x0c
        yr = float(node.Y0) - y0c
        idx_x = int(eq_id_map[i, 0])
        idx_y = int(eq_id_map[i, 1])
        if 0 <= idx_x < n_total_dof:
            dof_x[idx_x] = xr
            dof_y[idx_x] = yr
            is_x_dof[idx_x] = True
        if 0 <= idx_y < n_total_dof:
            dof_x[idx_y] = xr
            dof_y[idx_y] = yr
            is_x_dof[idx_y] = False
    x_free = dof_x[free_dofs]
    y_free = dof_y[free_dofs]
    is_x_free = is_x_dof[free_dofs]
    free_mask = np.zeros(n_total_dof, dtype=bool)
    free_mask[free_dofs] = True
    dir_dofs = np.nonzero(~free_mask)[0].astype(np.int64)
    x_dir = dof_x[dir_dofs]
    y_dir = dof_y[dir_dofs]
    is_x_dir = is_x_dof[dir_dofs]
    
    # Pre-process Basis
    Vs = torch.from_numpy(phi_s.astype(np.float32)).to(device)

    results_eps = [np.zeros(3, dtype=float)]
    results_sig = [np.zeros(3, dtype=float)]
    q_hist = [np.zeros(n_primary, dtype=float)]
    
    ann_input_dim = int(ann_model.input_scaler.mean.numel())
    expected_input_dim = int(n_primary)
    if ann_input_dim != expected_input_dim:
        raise ValueError(
            f"ANN input size mismatch: model expects {ann_input_dim}, "
            f"but solver was configured for {expected_input_dim}."
        )
    qp_init_mode = str(qp_init_mode).strip().lower()
    if qp_init_mode not in ("continuation", "previous", "zero", "mu_affine"):
        raise ValueError(
            f"Unsupported qp_init_mode='{qp_init_mode}'. "
            "Use one of: continuation, previous, zero, mu_affine."
        )
    qp_aff = getattr(ann_model, "qp_init_mu_affine", None)
    if qp_init_mode in ("continuation", "mu_affine") and qp_aff is None:
        raise RuntimeError(
            f"[PROM-ANN] qp_init_mode='{qp_init_mode}' requires qm_init_mu_affine.npz "
            "in the ANN model directory."
        )

    def _build_ann_input(qp_tensor, e_tensor):
        return qp_tensor

    jac_backend = "jacfwd" if _torch_jacfwd is not None else "autograd"

    def compute_manifold_jacobian(qp_val, e_tensor):
        nonlocal jac_backend
        with torch.enable_grad():
            qp_in = qp_val.reshape(-1).clone().detach().requires_grad_(True)

            def ann_from_qvec(q_local_vec):
                q_local = q_local_vec.view(1, -1)
                return ann_model(_build_ann_input(q_local, e_tensor)).reshape(-1)

            if jac_backend == "jacfwd":
                try:
                    jac_ann = _torch_jacfwd(ann_from_qvec)(qp_in)
                except Exception:
                    jac_backend = "autograd"
                    jac_ann = torch.autograd.functional.jacobian(ann_from_qvec, qp_in)
            else:
                jac_ann = torch.autograd.functional.jacobian(ann_from_qvec, qp_in)
            jac_ann = jac_ann.reshape(n_secondary, n_primary)
        return jac_ann

    def compute_weighted_decoder_hessian(qp_val, e_tensor, output_weights):
        weights = torch.as_tensor(
            output_weights,
            dtype=qp_val.dtype,
            device=qp_val.device,
        ).reshape(-1)
        if weights.numel() != n_secondary:
            raise RuntimeError(
                "Invalid decoder-Hessian weights: "
                f"got {weights.numel()}, expected {n_secondary}."
            )

        with torch.enable_grad():
            qp_in = qp_val.reshape(-1).clone().detach().requires_grad_(True)

            def weighted_ann_output(q_local_vec):
                q_local = q_local_vec.view(1, -1)
                q_s_raw = ann_model(_build_ann_input(q_local, e_tensor)).reshape(-1)
                return torch.dot(weights, q_s_raw)

            hessian = torch.autograd.functional.hessian(
                weighted_ann_output,
                qp_in,
                vectorize=True,
            )
        return hessian.reshape(n_primary, n_primary)

    def _is_finite(arr):
        return bool(np.all(np.isfinite(np.asarray(arr))))

    def _compute_affine_component(e_vec, x_loc, y_loc, is_x_loc):
        F = DeformationGradientFromGreenLagrange2D(e_vec)
        ux = (F[0, 0] - 1.0) * x_loc + F[0, 1] * y_loc
        uy = F[1, 0] * x_loc + (F[1, 1] - 1.0) * y_loc
        return np.where(is_x_loc, ux, uy)

    def _compute_affine_free_displacement(e_vec):
        return _compute_affine_component(e_vec, x_free, y_free, is_x_free)

    def _compute_affine_dirichlet_displacement(e_vec):
        return _compute_affine_component(e_vec, x_dir, y_dir, is_x_dir)

    def _initial_qp_guess(e_vec, q_prev, step_index):
        if qp_init_mode == "continuation" and int(step_index) > 1:
            return np.asarray(q_prev, dtype=float).copy()
        if qp_init_mode == "previous":
            return np.asarray(q_prev, dtype=float).copy()
        if qp_init_mode == "zero":
            return np.zeros(n_primary, dtype=float)
        mu_dim = int(qp_aff["mu_dim"])
        mu = np.asarray(e_vec, dtype=float).reshape(-1)[:mu_dim]
        if mu.size < mu_dim:
            raise RuntimeError(
                f"[PROM-ANN] q_m initializer expects mu_dim={mu_dim}, got {mu.size}."
            )
        return np.concatenate([mu, np.array([1.0])]) @ np.asarray(
            qp_aff["b_aff"],
            dtype=float,
        )

    def _capture_current_displacement_vector():
        disp_vec = np.zeros(n_total_dof, dtype=float)
        for i, node in enumerate(mp.Nodes):
            d = node.GetSolutionStepValue(KM.DISPLACEMENT)
            idx_x, idx_y = eq_id_map[i, 0], eq_id_map[i, 1]
            if idx_x < n_total_dof:
                disp_vec[idx_x] = d[0]
            if idx_y < n_total_dof:
                disp_vec[idx_y] = d[1]
        return disp_vec

    def _apply_total_free_displacement(u_total_free, base_disp_vec=None):
        if base_disp_vec is None:
            disp_vec = _capture_current_displacement_vector()
        else:
            disp_vec = np.asarray(base_disp_vec, dtype=float).copy()
        disp_vec[free_dofs] = np.asarray(u_total_free, dtype=float).reshape(-1)
        SetDisplacementFromEquationVector(disp_vec, eq_id_map, ta)
        UpdateCurrentCoordinatesFromDisplacement(mp, step=0)
        return disp_vec

    def _solve_reduced_system(K_sys, rhs):
        try:
            dq_loc = np.linalg.solve(K_sys, rhs)
        except np.linalg.LinAlgError:
            dq_loc, *_ = np.linalg.lstsq(K_sys, rhs, rcond=None)
        if _is_finite(dq_loc):
            return dq_loc

        K_reg = K_sys + float(regularization) * np.eye(K_sys.shape[0], dtype=K_sys.dtype)
        try:
            dq_loc = np.linalg.solve(K_reg, rhs)
        except np.linalg.LinAlgError:
            dq_loc, *_ = np.linalg.lstsq(K_reg, rhs, rcond=None)
        return dq_loc

    q_p_torch = torch.zeros((1, n_primary), device=device)
    q_m_scale = (
        ann_model.input_scaler.std.detach().cpu().numpy().reshape(-1).astype(float)
    )
    if q_m_scale.size != n_primary:
        raise RuntimeError(
            f"ANN q_m scale size mismatch: got {q_m_scale.size}, expected {n_primary}."
        )
    q_m_scale = np.maximum(np.abs(q_m_scale), 1.0e-12)
    Kr_old = None
    J_full = np.zeros((n_total_dof, n_primary), dtype=float)
    q_zero_torch = torch.zeros((1, n_primary), device=device)
    e_zero_torch = torch.zeros((1, 3), device=device)
    with torch.no_grad():
        N0_const = ann_model(_build_ann_input(q_zero_torch, e_zero_torch))
    J0_const_torch = compute_manifold_jacobian(q_zero_torch, e_zero_torch).detach()
    J0_const_np = J0_const_torch.cpu().numpy()
    N0_const_np = N0_const.detach().cpu().numpy().reshape(-1)
    a_m = np.asarray(getattr(ann_model, "a_m_np", None), dtype=float)
    if a_m.shape != (n_primary, n_primary):
        raise RuntimeError(
            f"[PROM-ANN] Missing or invalid LS master map A_m: {a_m.shape}."
        )
    phi_master = phi_p @ a_m
    phi_p_eff_np = phi_master + phi_s @ J0_const_np
    w0_const_np = phi_s @ N0_const_np
    Vp_eff = torch.from_numpy(phi_p_eff_np.astype(np.float32)).to(device)
    w0_const_t = torch.from_numpy(w0_const_np.astype(np.float32)).to(device).unsqueeze(0)
    
    print(f"  [PROM-ANN] Solving for {n_steps_total} dynamic increments...")
    print(f"  [PROM-ANN] Decoder Jacobian backend: {jac_backend}")
    if bool(getattr(ann_model, "origin_anchored", False)):
        print("  [PROM-ANN] Origin-anchored ANN: q_s(q_m=0)=0 exactly.")
    else:
        print(
            "  [PROM-ANN] Legacy ANN without exact origin anchoring: "
            f"||q_s(0)||={np.linalg.norm(N0_const_np):.3e}."
        )
    print("  [PROM-ANN] LS decoder active: u = Phi_m A_m q_m + Phi_s q_s(q_m).")
    print(f"  [PROM-ANN] q_m initializer mode: {qp_init_mode}")
    if qp_init_mode in ("continuation", "mu_affine") and qp_aff is not None:
        print(f"  [PROM-ANN] Using affine initializer: {qp_aff['path']}")
    if max_its == 0:
        print("  [PROM-ANN] Direct mode: Newton correction disabled (max_its=0).")
    t_map = 0.0
    t_assembly = 0.0
    t_projection = 0.0
    t_solve = 0.0
    t_full_sync = 0.0
    step_iters = []
    
    for step in range(1, n_steps_total + 1):
        time_val = float(step) * float(dt)
        mp.CloneTimeStep(time_val)
        mp.ProcessInfo[KM.DELTA_TIME] = dt
        mp.ProcessInfo[KM.TIME] = time_val
        mp.ProcessInfo[KM.STEP] = step

        sim.time, sim.step, sim.end_time = time_val, step, end_time
        sim.InitializeSolutionStep()

        s = int(np.searchsorted(step_offsets, step, side="left") - 1)
        s = max(0, min(s, n_seg - 1))
        xi = float(step - step_offsets[s]) / float(max(seg_steps[s], 1))
        E = (1.0 - xi) * E_wp[s, :] + xi * E_wp[s + 1, :]
        E_torch = torch.from_numpy(E.astype(np.float32)).unsqueeze(0).to(device)
        u_aff_free = _compute_affine_free_displacement(E)

        sim.batch_strain = E.copy()
        if use_fast_dirichlet_bc:
            disp_base_step = np.zeros(n_total_dof, dtype=float)
            disp_base_step[dir_dofs] = _compute_affine_dirichlet_displacement(E)
        else:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()

        if step == 1 or step % 100 == 0 or step == n_steps_total:
            print(f"\n[PROM-ANN] Step {step:04d}/{n_steps_total} | E={E}")

        q_prev_np = q_p_torch.detach().cpu().numpy().reshape(-1)
        q_init_np = _initial_qp_guess(E, q_prev_np, step)
        q_p_torch = torch.from_numpy(
            np.asarray(q_init_np, dtype=np.float32)
        ).reshape(1, -1).to(device)
        
        # Reduced Newton
        it = 0
        res_norm_0 = None
        converged = bool(max_its == 0)
        nonfinite_detected = False
        Kr_last = None
        dq_norm_prev = None
        prev_res_norm = None
        prev_q_eval = None
        prev_qs_eval = None
        plateau_count = 0
        q_step_start = q_p_torch.detach().clone()
        best_q = q_step_start.detach().clone()
        best_res = np.inf
        best_rel = np.inf
        it_step_count = 0
        
        verbose_step = bool(verbose_iterations)
        while it < max_its:
            mp.ProcessInfo[KM.NL_ITERATION_NUMBER] = it + 1
            it_step_count += 1
            t0 = time.perf_counter()
            # 1. Neural Manifold Enrichment & Jacobian
            with torch.no_grad():
                q_s_raw = ann_model(_build_ann_input(q_p_torch, E_torch))
                q_s = q_s_raw - N0_const - (q_p_torch @ J0_const_torch.T)
            
            qp_np = q_p_torch.detach().cpu().numpy().flatten()
            qs_np = q_s.detach().cpu().numpy().flatten()
            if verbose_step:
                print(f"    > q_p norm: {np.linalg.norm(qp_np):.3e} | q_s norm: {np.linalg.norm(qs_np):.3e}")
            if (not _is_finite(qp_np)) or (not _is_finite(qs_np)):
                print("  [PROM-ANN] WARNING: non-finite reduced state detected.")
                nonfinite_detected = True
                break
            q_eval_delta = np.inf
            q_eval_delta_normalized = np.inf
            if prev_q_eval is not None and prev_q_eval.shape == qp_np.shape:
                q_eval_change = qp_np - prev_q_eval
                q_eval_delta = float(np.linalg.norm(q_eval_change))
                q_eval_delta_normalized = float(
                    np.linalg.norm(q_eval_change / q_m_scale)
                )
            prev_q_eval = qp_np.copy()
            q_s_delta = np.inf
            if prev_qs_eval is not None and prev_qs_eval.shape == qs_np.shape:
                q_s_delta = float(np.linalg.norm(qs_np - prev_qs_eval))
            prev_qs_eval = qs_np.copy()
            
            # Displacement reconstruction
            u_fluc_torch = w0_const_t + q_p_torch @ Vp_eff.T + q_s @ Vs.T
            u_fluc = u_fluc_torch.detach().cpu().numpy().flatten()
            if not _is_finite(u_fluc):
                print("  [PROM-ANN] WARNING: non-finite reconstructed displacement detected.")
                nonfinite_detected = True
                break
            u_free = u_aff_free + u_fluc
            
            # Jacobian: J = Vp + Vs * (dq_s/dq_p)
            J_ann_raw = compute_manifold_jacobian(q_p_torch, E_torch).detach().cpu().numpy()
            J_ann = J_ann_raw - J0_const_np
            if J_ann.shape != (n_secondary, n_primary):
                raise RuntimeError(
                    f"Invalid ANN Jacobian shape {J_ann.shape}; expected ({n_secondary}, {n_primary})."
                )
            if not _is_finite(J_ann):
                print("  [PROM-ANN] WARNING: non-finite ANN Jacobian detected.")
                nonfinite_detected = True
                break
            J_manifold = phi_p_eff_np + phi_s @ J_ann
            if not _is_finite(J_manifold):
                print("  [PROM-ANN] WARNING: non-finite manifold Jacobian detected.")
                nonfinite_detected = True
                break
            t_map += time.perf_counter() - t0
            
            # 2. Update Kratos
            u_eq_curr = _apply_total_free_displacement(u_free, base_disp_vec=disp_base_step)
            
            # 3. Assemble
            InitializeNonLinearIteration(entities, mp.ProcessInfo)
            t0 = time.perf_counter()
            K_sparse, rhs_vec = vec_assembler.Assemble(u_eq_curr)
            t_assembly += time.perf_counter() - t0
            FinalizeNonLinearIteration(entities, mp.ProcessInfo)
            
            if (not _is_finite(rhs_vec)) or (not _is_finite(K_sparse.data)):
                print("  [PROM-ANN] WARNING: non-finite full residual/stiffness detected.")
                nonfinite_detected = True
                break
            
            # 4. Galerkin Projection
            t0 = time.perf_counter()
            J_full.fill(0.0)
            J_full[free_dofs, :] = J_manifold
            KJ = K_sparse @ J_full
            r_r = J_full.T @ rhs_vec
            K_std = J_full.T @ KJ
            # Exact nonlinear-manifold Newton tangent. Since rhs_vec=-f_int,
            # d(J^T rhs)/dq = H_u:rhs - J^T K J, so the matrix multiplying
            # the additive update is J^T K J - H_u:rhs.
            curvature_weights = phi_s.T @ rhs_vec[free_dofs]
            K_curv = (
                compute_weighted_decoder_hessian(
                    q_p_torch,
                    E_torch,
                    curvature_weights,
                )
                .detach()
                .cpu()
                .numpy()
            )
            K_curv = 0.5 * (K_curv + K_curv.T)
            K_r = K_std - K_curv
            t_projection += time.perf_counter() - t0
            Kr_last = K_r
            if (not _is_finite(r_r)) or (not _is_finite(K_r)):
                print("  [PROM-ANN] WARNING: non-finite reduced residual/stiffness detected.")
                nonfinite_detected = True
                break
            if verbose_step:
                print(
                    "    > tangent norms: "
                    f"||J^T K J||={np.linalg.norm(K_std):.3e}, "
                    f"||K_curv||={np.linalg.norm(K_curv):.3e}"
                )
            
            res_norm = np.linalg.norm(r_r)
            if not np.isfinite(res_norm):
                print("  [PROM-ANN] WARNING: non-finite reduced residual norm detected.")
                nonfinite_detected = True
                break
            if res_norm_0 is None:
                res_norm_0 = max(res_norm, 1e-30)
            rel_res = res_norm / (res_norm_0 + 1e-12)
            if res_norm < best_res:
                best_res = float(res_norm)
                best_q = q_p_torch.detach().clone()
                best_rel = float(rel_res)
            update_text = ""
            if np.isfinite(q_eval_delta_normalized):
                update_text = (
                    f", ||D_q^-1 dq_m|| = {q_eval_delta_normalized:.3e}"
                )
            print(
                f"  > It {it:02d}: ||R_r|| = {res_norm:.3e}, "
                f"rel = {rel_res:.3e}{update_text}"
            )
            
            if res_norm < float(abs_res_cutoff):
                print(f"  > Converged in {it} iterations.")
                converged = True
                break
            if (
                it > 0
                and np.isfinite(q_eval_delta_normalized)
                and q_eval_delta_normalized < float(normalized_dq_cutoff)
            ):
                print(
                    f"  > Converged in {it} iterations "
                    f"(normalized q_m update={q_eval_delta_normalized:.3e} "
                    f"< {float(normalized_dq_cutoff):.3e})."
                )
                converged = True
                break
            if (
                dq_norm_prev is not None
                and dq_norm_prev < float(dq_abs_cutoff)
                and rel_res < float(relnorm_cutoff)
                and res_norm < float(max_res_for_rel_convergence)
            ):
                print(f"  > Converged in {it} iterations (small update + reduced residual).")
                converged = True
                break
            if prev_res_norm is not None:
                rel_drop = abs(prev_res_norm - res_norm) / max(prev_res_norm, 1e-30)
                q_tol = 20.0 * float(dq_abs_cutoff)
                q_is_flat = (
                    np.isfinite(q_eval_delta)
                    and np.isfinite(q_s_delta)
                    and (q_eval_delta < q_tol)
                    and (q_s_delta < q_tol)
                )
                if (
                    it >= 2
                    and q_is_flat
                    and rel_res < max(float(relnorm_cutoff), 0.5 * float(stagnation_relnorm_gate))
                ):
                    print(
                        f"  > Converged in {it} iterations (frozen q-state: "
                        f"q_p_delta={q_eval_delta:.3e}, q_s_delta={q_s_delta:.3e}, rel={rel_res:.3e})."
                    )
                    converged = True
                    break
                if rel_drop < float(min_rel_drop_stop) and q_is_flat:
                    plateau_count += 1
                else:
                    plateau_count = 0
                if (
                    rel_drop < float(min_rel_drop_stop)
                    and rel_res < float(stagnation_relnorm_gate)
                    and res_norm < float(max_res_for_rel_convergence)
                ):
                    print(
                        f"  > Converged in {it} iterations (stagnation criterion: "
                        f"rel_drop={rel_drop:.3e})."
                    )
                    converged = True
                    break
                # Flat reduced residual and essentially unchanged q-state:
                # accept as stalled fixed-point even when absolute reduced residual gate is not met.
                if (
                    plateau_count >= 3
                    and rel_res < max(3.0 * float(relnorm_cutoff), float(stagnation_relnorm_gate))
                    and (dq_norm_prev is None or dq_norm_prev < 5.0 * float(dq_abs_cutoff))
                ):
                    print(
                        f"  > Converged in {it} iterations (flat residual plateau: "
                        f"rel_drop={rel_drop:.3e}, q_p_delta={q_eval_delta:.3e}, q_s_delta={q_s_delta:.3e})."
                    )
                    converged = True
                    break
            prev_res_norm = float(res_norm)
                
            # 5. Solve (with small dampening if res high)
            K_solve = K_r
            used_old = False
            if (
                it == 0
                and use_old_stiffness_in_first_iteration
                and Kr_old is not None
                and Kr_old.shape == K_r.shape
                and res_norm < float(old_stiffness_residual_cutoff)
            ):
                K_solve = Kr_old
                used_old = True

            t0 = time.perf_counter()
            dq_p = _solve_reduced_system(K_solve, r_r)
            if used_old and not _is_finite(dq_p):
                dq_p = _solve_reduced_system(K_r, r_r)
                used_old = False
            t_solve += time.perf_counter() - t0
            if not _is_finite(dq_p):
                print("  [PROM-ANN] WARNING: non-finite reduced update detected.")
                nonfinite_detected = True
                break
            dq_norm = float(np.linalg.norm(dq_p))
            if dq_norm > float(max_dq_norm) and dq_norm > 0.0:
                scale = float(max_dq_norm) / dq_norm
                dq_p *= scale
                dq_norm = float(np.linalg.norm(dq_p))
                print(f"    > large reduced update clipped with scale={scale:.3e}")
            
            dq_torch = torch.from_numpy(dq_p.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                q_p_torch = q_p_torch + dq_torch
            dq_norm_prev = dq_norm
            if used_old:
                print("    > using previous reduced stiffness (K_old) at first iteration")
            it += 1

        if not converged:
            quasi_converged = (
                np.isfinite(best_res)
                and np.isfinite(best_rel)
                and (best_rel < float(relnorm_cutoff))
                and (best_res < float(max_res_for_rel_convergence))
            )
            plateau_quasi = (
                np.isfinite(best_res)
                and np.isfinite(best_rel)
                and plateau_count >= 3
                and (best_rel < max(3.0 * float(relnorm_cutoff), float(stagnation_relnorm_gate)))
            )
            if quasi_converged or plateau_quasi:
                q_p_torch = best_q.detach().clone()
                converged = True
                if quasi_converged:
                    print(
                        "  [PROM-ANN] Step accepted as quasi-converged: "
                        f"best ||R_r||={best_res:.3e}, rel={best_rel:.3e}."
                    )
                else:
                    print(
                        "  [PROM-ANN] Step accepted as plateau-converged: "
                        f"best ||R_r||={best_res:.3e}, rel={best_rel:.3e}, "
                        f"plateau_count={plateau_count}."
                    )
                Kr_old = None
            else:
                msg = f"[PROM-ANN] Step {step} did not converge in {max_its} iterations."
                if bool(fail_on_nonconvergence):
                    raise RuntimeError(
                        msg
                        + f" Best ||R_r||={best_res:.3e}, rel={best_rel:.3e}."
                    )
                print(f"  [PROM-ANN] WARNING: {msg}")
                if nonfinite_detected:
                    print("  [PROM-ANN] WARNING: non-finite state encountered; rolling back to best finite iterate.")
                if np.isfinite(best_res):
                    q_p_torch = best_q.detach().clone()
                    print(f"  [PROM-ANN] Using best finite iterate with ||R_r||={best_res:.3e}.")
                else:
                    q_p_torch = q_step_start.detach().clone()
                    print("  [PROM-ANN] Reverting to previous-step reduced state.")
                Kr_old = None
        elif Kr_last is not None and _is_finite(Kr_last):
            Kr_old = Kr_last.copy()
        else:
            Kr_old = None

        # Ensure the accepted reduced state is explicitly pushed to Kratos.
        with torch.no_grad():
            q_s_final_raw = ann_model(_build_ann_input(q_p_torch, E_torch))
            q_s_final = q_s_final_raw - N0_const - (q_p_torch @ J0_const_torch.T)
            u_fluc_final = (
                w0_const_t + q_p_torch @ Vp_eff.T + q_s_final @ Vs.T
            ).detach().cpu().numpy().reshape(-1)
        if not _is_finite(u_fluc_final):
            q_p_torch = q_step_start.detach().clone()
            with torch.no_grad():
                q_s_final_raw = ann_model(_build_ann_input(q_p_torch, E_torch))
                q_s_final = q_s_final_raw - N0_const - (q_p_torch @ J0_const_torch.T)
                u_fluc_final = (
                    w0_const_t + q_p_torch @ Vp_eff.T + q_s_final @ Vs.T
                ).detach().cpu().numpy().reshape(-1)
        if not _is_finite(u_fluc_final):
            raise RuntimeError("PROM-ANN accepted state is non-finite after rollback.")
        if not use_fast_dirichlet_bc:
            sim.ApplyBoundaryConditions()
            disp_base_step = _capture_current_displacement_vector()
        u_eq_final = _apply_total_free_displacement(u_aff_free + u_fluc_final, base_disp_vec=disp_base_step)

        InitializeNonLinearIteration(entities, mp.ProcessInfo)
        t0 = time.perf_counter()
        _, _ = vec_assembler.Assemble(u_eq_final)
        t_full_sync += time.perf_counter() - t0
        FinalizeNonLinearIteration(entities, mp.ProcessInfo)
        eps_h, sig_h = CalculateHomogenizedFromAssemblerWithElementWeights(
            vec_assembler,
            reference_measure=hom_reference_measure,
        )
        sim.FinalizeSolutionStep()
        step_iters.append(int(it_step_count))

        results_eps.append(eps_h)
        results_sig.append(sig_h)
        q_hist.append(q_p_torch.detach().cpu().numpy().reshape(-1).copy())

    sim.Finalize()
    if out_dir is not None:
        np.save(os.path.join(out_dir, "prom_ann_run_q_p.npy"), np.stack(q_hist))
    t_wall_total = time.perf_counter() - t_wall_total_start
    t_accounted = t_map + t_assembly + t_projection + t_solve + t_full_sync
    t_other = max(t_wall_total - t_accounted, 0.0)
    iters = np.asarray(step_iters, dtype=float)
    n_iter_total = int(np.sum(iters)) if iters.size else 0
    n_iter_mean = float(np.mean(iters)) if iters.size else 0.0
    print(
        f"\n[PROM-ANN] Timing: map={t_map:.3f}s, assembly={t_assembly:.3f}s, "
        f"projection={t_projection:.3f}s, solve={t_solve:.3f}s, full_sync={t_full_sync:.3f}s, "
        f"accounted={t_accounted:.3f}s, other={t_other:.3f}s, total={t_wall_total:.3f}s, "
        f"iters(total={n_iter_total}, mean/step={n_iter_mean:.2f})"
    )
    return np.array(results_eps), np.array(results_sig)
