import time

import numpy as np


def _ensure_2d(x, name):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
    return arr


def _ensure_3d(x, name):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape {arr.shape}.")
    return arr


def _normalize_model_shapes(model):
    x_mean = np.asarray(model["x_mean"], dtype=float).reshape(-1)
    x_std = np.asarray(model["x_std"], dtype=float).reshape(-1)
    y_mean = np.asarray(model["y_mean"], dtype=float).reshape(-1)
    y_std = np.asarray(model["y_std"], dtype=float).reshape(-1)

    input_dim = int(model["input_dim"])
    output_dim = int(model["output_dim"])
    if x_mean.size != input_dim or x_std.size != input_dim:
        raise ValueError("x_mean/x_std size mismatch with input_dim.")
    if y_mean.size != output_dim or y_std.size != output_dim:
        raise ValueError("y_mean/y_std size mismatch with output_dim.")

    inducing_points = _ensure_3d(model["inducing_points"], "inducing_points")
    alpha = _ensure_2d(model["alpha"], "alpha")
    if inducing_points.shape[0] != output_dim:
        raise ValueError("inducing_points first dimension must match output_dim.")
    if alpha.shape[0] != output_dim:
        raise ValueError("alpha first dimension must match output_dim.")
    if alpha.shape[1] != inducing_points.shape[1]:
        raise ValueError("alpha second dimension must match n_inducing.")

    lengthscales = np.asarray(model["lengthscales"], dtype=float)
    if lengthscales.ndim == 1:
        lengthscales = np.broadcast_to(lengthscales[None, :], (output_dim, input_dim)).copy()
    elif lengthscales.ndim == 2 and lengthscales.shape[0] == 1:
        lengthscales = np.broadcast_to(lengthscales, (output_dim, input_dim)).copy()
    elif lengthscales.ndim != 2 or lengthscales.shape != (output_dim, input_dim):
        raise ValueError(
            f"lengthscales must have shape ({output_dim}, {input_dim}), got {lengthscales.shape}."
        )
    if np.any(lengthscales <= 0.0):
        raise ValueError("All lengthscales must be > 0.")

    outputscales = np.asarray(model["outputscales"], dtype=float).reshape(-1)
    if outputscales.size == 1:
        outputscales = np.full(output_dim, float(outputscales[0]), dtype=float)
    elif outputscales.size != output_dim:
        raise ValueError(
            f"outputscales size mismatch: expected {output_dim}, got {outputscales.size}."
        )
    if np.any(outputscales <= 0.0):
        raise ValueError("All outputscales must be > 0.")

    model["x_mean"] = x_mean
    model["x_std"] = x_std
    model["y_mean"] = y_mean
    model["y_std"] = y_std
    model["inducing_points"] = inducing_points
    model["alpha"] = alpha
    model["lengthscales"] = lengthscales
    model["outputscales"] = outputscales
    return model


def save_sparse_gp_model(path, model_dict):
    model = dict(model_dict)
    model = _normalize_model_shapes(model)

    payload = {
        "x_mean": np.asarray(model["x_mean"], dtype=float),
        "x_std": np.asarray(model["x_std"], dtype=float),
        "y_mean": np.asarray(model["y_mean"], dtype=float),
        "y_std": np.asarray(model["y_std"], dtype=float),
        "inducing_points": np.asarray(model["inducing_points"], dtype=float),
        "alpha": np.asarray(model["alpha"], dtype=float),
        "lengthscales": np.asarray(model["lengthscales"], dtype=float),
        "outputscales": np.asarray(model["outputscales"], dtype=float),
        "input_dim": np.asarray([int(model["input_dim"])], dtype=np.int64),
        "output_dim": np.asarray([int(model["output_dim"])], dtype=np.int64),
        "n_primary": np.asarray([int(model["n_primary"])], dtype=np.int64),
        "n_secondary": np.asarray([int(model["n_secondary"])], dtype=np.int64),
        "include_macro_strain_input": np.asarray([0], dtype=np.int64),
        "kernel_name": np.asarray([str(model.get("kernel_name", "rbf"))], dtype=object),
        "model_family": np.asarray(["sparse_gp"], dtype=object),
    }

    if "noise" in model:
        payload["noise"] = np.asarray(model["noise"], dtype=float).reshape(-1)
    if "num_inducing" in model:
        payload["num_inducing"] = np.asarray([int(model["num_inducing"])], dtype=np.int64)
    if "backend" in model:
        payload["backend"] = np.asarray([str(model["backend"])], dtype=object)
    if "epochs" in model:
        payload["epochs"] = np.asarray([int(model["epochs"])], dtype=np.int64)
    if "batch_size" in model:
        payload["batch_size"] = np.asarray([int(model["batch_size"])], dtype=np.int64)
    if "train_samples_used" in model:
        payload["train_samples_used"] = np.asarray([int(model["train_samples_used"])], dtype=np.int64)
    if "loss_history" in model:
        payload["loss_history"] = np.asarray(model["loss_history"], dtype=float).reshape(-1)

    np.savez(path, **payload)


def load_sparse_gp_model(path):
    data = np.load(path, allow_pickle=True)
    model = {
        "x_mean": np.asarray(data["x_mean"], dtype=float),
        "x_std": np.asarray(data["x_std"], dtype=float),
        "y_mean": np.asarray(data["y_mean"], dtype=float),
        "y_std": np.asarray(data["y_std"], dtype=float),
        "inducing_points": np.asarray(data["inducing_points"], dtype=float),
        "alpha": np.asarray(data["alpha"], dtype=float),
        "lengthscales": np.asarray(data["lengthscales"], dtype=float),
        "outputscales": np.asarray(data["outputscales"], dtype=float),
        "input_dim": int(np.ravel(data["input_dim"])[0]),
        "output_dim": int(np.ravel(data["output_dim"])[0]),
        "n_primary": int(np.ravel(data["n_primary"])[0]),
        "n_secondary": int(np.ravel(data["n_secondary"])[0]),
        "include_macro_strain_input": bool(int(np.ravel(data["include_macro_strain_input"])[0])),
        "kernel_name": str(np.ravel(data["kernel_name"])[0]) if "kernel_name" in data else "rbf",
        "model_family": str(np.ravel(data["model_family"])[0]) if "model_family" in data else "sparse_gp",
    }
    if "noise" in data:
        model["noise"] = np.asarray(data["noise"], dtype=float).reshape(-1)
    if "num_inducing" in data:
        model["num_inducing"] = int(np.ravel(data["num_inducing"])[0])
    if "backend" in data:
        model["backend"] = str(np.ravel(data["backend"])[0])
    return _normalize_model_shapes(model)


def _evaluate_sparse_gp_kernel(x_scaled, z, lengthscales, outputscale):
    # x_scaled: (d,), z: (m,d), lengthscales: (d,)
    diff = x_scaled[None, :] - z
    inv_l2 = 1.0 / (lengthscales * lengthscales)
    scaled_sq = np.sum(diff * diff * inv_l2[None, :], axis=1)
    k = float(outputscale) * np.exp(-0.5 * scaled_sq)
    return k, diff, inv_l2


def evaluate_sparse_gp_map(input_vec, model):
    m = _normalize_model_shapes(dict(model))
    x = np.asarray(input_vec, dtype=float).reshape(-1)
    if x.size != int(m["input_dim"]):
        raise ValueError(f"Input size mismatch: got {x.size}, expected {m['input_dim']}.")

    x_scaled = (x - m["x_mean"]) / m["x_std"]
    out_scaled = np.zeros(int(m["output_dim"]), dtype=float)
    for j in range(int(m["output_dim"])):
        k, _, _ = _evaluate_sparse_gp_kernel(
            x_scaled,
            m["inducing_points"][j, :, :],
            m["lengthscales"][j, :],
            m["outputscales"][j],
        )
        out_scaled[j] = float(np.dot(k, m["alpha"][j, :]))

    return out_scaled * m["y_std"] + m["y_mean"]


def _profile_add(profile, key, value):
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + float(value)


def evaluate_sparse_gp_map_and_jacobian_qp(input_vec, model, n_primary, profile=None):
    t0 = time.perf_counter() if profile is not None else None
    m = _normalize_model_shapes(dict(model))
    if profile is not None:
        _profile_add(profile, "normalize_model", time.perf_counter() - t0)

    t0 = time.perf_counter() if profile is not None else None
    x = np.asarray(input_vec, dtype=float).reshape(-1)
    if x.size != int(m["input_dim"]):
        raise ValueError(f"Input size mismatch: got {x.size}, expected {m['input_dim']}.")
    n_p = int(n_primary)
    if n_p < 1 or n_p > x.size:
        raise ValueError(f"Invalid n_primary={n_p} for input dimension {x.size}.")

    x_scaled = (x - m["x_mean"]) / m["x_std"]
    out_scaled = np.zeros(int(m["output_dim"]), dtype=float)
    jac_scaled_xscaled = np.zeros((int(m["output_dim"]), int(m["input_dim"])), dtype=float)
    if profile is not None:
        _profile_add(profile, "input_scale_and_alloc", time.perf_counter() - t0)

    t0 = time.perf_counter() if profile is not None else None
    diff = x_scaled[None, None, :] - m["inducing_points"]
    inv_l2 = 1.0 / (m["lengthscales"] * m["lengthscales"])
    scaled_sq = np.sum(diff * diff * inv_l2[:, None, :], axis=2)
    k = m["outputscales"][:, None] * np.exp(-0.5 * scaled_sq)
    if profile is not None:
        _profile_add(profile, "kernel_eval", time.perf_counter() - t0)

    t0 = time.perf_counter() if profile is not None else None
    alpha_k = m["alpha"] * k
    out_scaled[:] = np.sum(alpha_k, axis=1)
    if profile is not None:
        _profile_add(profile, "mean_dot", time.perf_counter() - t0)

    # d k / d x_scaled = -k * (x-z)/l^2
    t0 = time.perf_counter() if profile is not None else None
    jac_scaled_xscaled[:] = -np.einsum("om,omd->od", alpha_k, diff, optimize=True) * inv_l2
    if profile is not None:
        _profile_add(profile, "jacobian_terms", time.perf_counter() - t0)

    t0 = time.perf_counter() if profile is not None else None
    out = out_scaled * m["y_std"] + m["y_mean"]
    jac_out_xscaled = m["y_std"][:, None] * jac_scaled_xscaled
    jac_out_x = jac_out_xscaled / m["x_std"][None, :]
    if profile is not None:
        _profile_add(profile, "output_unscale", time.perf_counter() - t0)
        profile["calls"] = int(profile.get("calls", 0)) + 1
    return out, jac_out_x[:, :n_p]
