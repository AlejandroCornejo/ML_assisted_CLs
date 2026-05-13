import numpy as np


def _kernel_from_dist2(dist2, kernel_name, epsilon):
    kname = str(kernel_name).strip().lower()
    eps2 = float(epsilon) * float(epsilon)

    if kname in ("gaussian", "gauss"):
        return np.exp(-eps2 * dist2)

    z = 1.0 + eps2 * dist2
    if kname in ("inverse_multiquadric", "imq"):
        return 1.0 / np.sqrt(z)
    if kname in ("multiquadric", "mq"):
        return np.sqrt(z)

    raise ValueError(f"Unsupported RBF kernel '{kernel_name}'.")


def _phi_and_grad_xscaled(x_scaled, centers_scaled, kernel_name, epsilon):
    x = np.asarray(x_scaled, dtype=float).reshape(1, -1)
    c = np.asarray(centers_scaled, dtype=float)
    if c.ndim != 2:
        raise ValueError("centers_scaled must have shape (n_centers, input_dim).")
    if x.shape[1] != c.shape[1]:
        raise ValueError(
            f"Input dim mismatch: x has {x.shape[1]}, centers have {c.shape[1]}."
        )

    diff = x - c  # (m, d)
    dist2 = np.sum(diff * diff, axis=1)
    phi = _kernel_from_dist2(dist2, kernel_name, epsilon)

    kname = str(kernel_name).strip().lower()
    eps2 = float(epsilon) * float(epsilon)

    if kname in ("gaussian", "gauss"):
        grad = (-2.0 * eps2) * phi[:, None] * diff
    elif kname in ("inverse_multiquadric", "imq"):
        z = 1.0 + eps2 * dist2
        grad = (-eps2) * (z ** (-1.5))[:, None] * diff
    elif kname in ("multiquadric", "mq"):
        z = 1.0 + eps2 * dist2
        grad = (eps2 / np.sqrt(z))[:, None] * diff
    else:
        raise ValueError(f"Unsupported RBF kernel '{kernel_name}'.")

    return phi, grad


def save_rbf_model(path, model_dict):
    payload = {
        "centers_scaled": np.asarray(model_dict["centers_scaled"], dtype=float),
        "weights_scaled": np.asarray(model_dict["weights_scaled"], dtype=float),
        "x_mean": np.asarray(model_dict["x_mean"], dtype=float),
        "x_std": np.asarray(model_dict["x_std"], dtype=float),
        "y_mean": np.asarray(model_dict["y_mean"], dtype=float),
        "y_std": np.asarray(model_dict["y_std"], dtype=float),
        "epsilon": np.asarray([float(model_dict["epsilon"])], dtype=float),
        "ridge": np.asarray([float(model_dict["ridge"])], dtype=float),
        "kernel_name": np.asarray([str(model_dict["kernel_name"])], dtype=object),
        "input_dim": np.asarray([int(model_dict["input_dim"])], dtype=np.int64),
        "output_dim": np.asarray([int(model_dict["output_dim"])], dtype=np.int64),
        "n_primary": np.asarray([int(model_dict["n_primary"])], dtype=np.int64),
        "n_secondary": np.asarray([int(model_dict["n_secondary"])], dtype=np.int64),
        "include_macro_strain_input": np.asarray(
            [1 if bool(model_dict["include_macro_strain_input"]) else 0], dtype=np.int64
        ),
    }
    if "center_indices" in model_dict and model_dict["center_indices"] is not None:
        payload["center_indices"] = np.asarray(model_dict["center_indices"], dtype=np.int64)
    if "center_selection" in model_dict:
        payload["center_selection"] = np.asarray([str(model_dict["center_selection"])], dtype=object)
    if "sparse_prune_centers" in model_dict:
        payload["sparse_prune_centers"] = np.asarray([int(model_dict["sparse_prune_centers"])], dtype=np.int64)
    np.savez(path, **payload)


def load_rbf_model(path):
    data = np.load(path, allow_pickle=True)
    return {
        "centers_scaled": np.asarray(data["centers_scaled"], dtype=float),
        "weights_scaled": np.asarray(data["weights_scaled"], dtype=float),
        "x_mean": np.asarray(data["x_mean"], dtype=float),
        "x_std": np.asarray(data["x_std"], dtype=float),
        "y_mean": np.asarray(data["y_mean"], dtype=float),
        "y_std": np.asarray(data["y_std"], dtype=float),
        "epsilon": float(np.ravel(data["epsilon"])[0]),
        "ridge": float(np.ravel(data["ridge"])[0]),
        "kernel_name": str(np.ravel(data["kernel_name"])[0]),
        "input_dim": int(np.ravel(data["input_dim"])[0]),
        "output_dim": int(np.ravel(data["output_dim"])[0]),
        "n_primary": int(np.ravel(data["n_primary"])[0]),
        "n_secondary": int(np.ravel(data["n_secondary"])[0]),
        "include_macro_strain_input": bool(int(np.ravel(data["include_macro_strain_input"])[0])),
        "center_indices": np.asarray(data["center_indices"], dtype=np.int64)
        if "center_indices" in data
        else None,
        "center_selection": str(np.ravel(data["center_selection"])[0])
        if "center_selection" in data
        else "random",
        "sparse_prune_centers": int(np.ravel(data["sparse_prune_centers"])[0])
        if "sparse_prune_centers" in data
        else 0,
    }


def evaluate_rbf_map(input_vec, model):
    x = np.asarray(input_vec, dtype=float).reshape(-1)
    if x.size != int(model["input_dim"]):
        raise ValueError(f"Input size mismatch: got {x.size}, expected {model['input_dim']}.")

    x_scaled = (x - model["x_mean"]) / model["x_std"]
    phi, _ = _phi_and_grad_xscaled(
        x_scaled,
        model["centers_scaled"],
        model["kernel_name"],
        model["epsilon"],
    )
    y_scaled = phi @ model["weights_scaled"]
    return y_scaled * model["y_std"] + model["y_mean"]


def evaluate_rbf_map_and_jacobian_qp(input_vec, model, n_primary):
    x = np.asarray(input_vec, dtype=float).reshape(-1)
    if x.size != int(model["input_dim"]):
        raise ValueError(f"Input size mismatch: got {x.size}, expected {model['input_dim']}.")

    n_p = int(n_primary)
    if n_p < 1 or n_p > x.size:
        raise ValueError(f"Invalid n_primary={n_p} for input dimension {x.size}.")

    x_scaled = (x - model["x_mean"]) / model["x_std"]
    phi, grad_xscaled = _phi_and_grad_xscaled(
        x_scaled,
        model["centers_scaled"],
        model["kernel_name"],
        model["epsilon"],
    )

    w_scaled = np.asarray(model["weights_scaled"], dtype=float)
    y_scaled = phi @ w_scaled

    # J_scaled_xscaled has shape (n_out, d)
    j_scaled_xscaled = (grad_xscaled.T @ w_scaled).T
    j_scaled_x = j_scaled_xscaled / model["x_std"][None, :]

    y = y_scaled * model["y_std"] + model["y_mean"]
    j_y_x = model["y_std"][:, None] * j_scaled_x

    return y, j_y_x[:, :n_p]
