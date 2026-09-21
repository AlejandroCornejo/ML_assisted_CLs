"""Evaluate the three preselected Free witnesses with the unchanged periodic FOM."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from protocol.run_data_stage import (BASE, DEFAULT_SPEC, atomic_json, digest,
    continuation_solve, field_stats, pf, boundary_screen, rim_indices,
    enforce_physical_screen)
from run_pilot import derivative_check
from audit_pilot import rank_one_screen

OUTPUT = BASE/'results/directed_search'


def curvature(e, stress, tangent, a, b):
    C = np.array([[1+2*e[0], e[2]], [e[2], 1+2*e[1]]])
    vals, vec = np.linalg.eigh(C)
    F = (vec*np.sqrt(vals))@vec.T
    S = np.array([[stress[0], stress[2]], [stress[2], stress[1]]])
    H = np.outer(a, b)
    edot = (F.T@H+H.T@F)/2
    ev = np.array([edot[0, 0], edot[1, 1], 2*edot[0, 1]])
    return float(ev@tangent@ev+np.sum(S*(H.T@H)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slug', required=True)
    args = parser.parse_args()
    result_path = OUTPUT/(args.slug+'_fom.json')
    if result_path.exists():
        raise FileExistsError(result_path)
    run = json.loads((OUTPUT/(args.slug+'.json')).read_text())
    w = run['nearest']
    if not w or not w['negative_verified']:
        raise ValueError('No verified witness')
    spec = json.loads(DEFAULT_SPEC.read_text())
    geometry_path = BASE/spec['domain_basis']['geometry_specification']
    if digest(geometry_path) != spec['domain_basis']['geometry_specification_sha256']:
        raise ValueError('Geometry changed')
    geometry = json.loads(geometry_path.read_text())
    report = dict(status='running', witness=w, source_sha256=digest(Path(__file__)),
                  witness_sha256=digest(OUTPUT/(args.slug+'.json')),
                  protocol_sha256=digest(DEFAULT_SPEC), meshes=[])
    atomic_json(result_path, report)
    from _material_law_guard_claude import true_neo_hookean_active
    for mesh in ('working', 'audit'):
        start = time.monotonic()
        record = dict(mesh=mesh, equilibrium_converged=False, physical_screen_passed=False)
        try:
            mesh_path = BASE/spec['mesh_policy'][mesh+'_mesh']
            coord_path = BASE/spec['mesh_policy'][mesh+'_mesh_coordinates']
            if digest(mesh_path) != spec['mesh_policy'][mesh+'_mesh_sha256']:
                raise ValueError('Mesh changed')
            if digest(coord_path) != spec['mesh_policy'][mesh+'_mesh_coordinates_sha256']:
                raise ValueError('Coordinates changed')
            record['mesh_sha256'] = digest(mesh_path)
            xy = np.load(coord_path)['xy']
            rims = rim_indices(geometry, xy)
            e = np.array(w['strain'])
            with true_neo_hookean_active():
                rve = pf.PeriodicRVE(mesh_path.with_suffix(''), cell_area=geometry['cell_side']**2)
                s, q, attempts = continuation_solve(rve, e,
                    spec['solver_policy']['maximum_engineering_strain_vector_increment'],
                    spec['solver_policy']['minimum_increment'])
                record['attempts'] = attempts
                s, d, q = rve.stress_and_tangent_consistent(e, u_ind_init=q, E_start=e, return_state=True)
                fields, u, _ = field_stats(rve, e, q)
                boundary = boundary_screen(xy, u[rve._eq_map], rims, e, geometry['cell_side'])
                record.update(equilibrium_converged=True, stress=s.tolist(), tangent=d.tolist(),
                    energy=float(rve.homogenized_energy()), fields=fields, boundary=boundary,
                    matched_curvature_Pa=curvature(e, s, d, w['a'], w['b']),
                    sampled_minimum_curvature_Pa=rank_one_screen(e, s, d))
                np.savez_compressed(OUTPUT/(args.slug+'_'+mesh+'_state.npz'), strain=e, q=q,
                                    stress=s, tangent=d, displacement=u)
                if fields['relative_reduced_residual'] > spec['solver_policy']['relative_residual_tolerance']:
                    raise RuntimeError('Residual screen failed')
                enforce_physical_screen(fields, boundary, spec)
                record['physical_screen_passed'] = True
                record['derivative_check'] = derivative_check(rve, e, q, s, d, 1e-6)
        except Exception as error:
            record['error'] = repr(error)
            record.setdefault('attempts', getattr(error, 'attempts', []))
        record['seconds'] = time.monotonic()-start
        report['meshes'].append(record)
        atomic_json(result_path, report)
        print(args.slug, mesh, 'converged', record['equilibrium_converged'],
              'screened', record['physical_screen_passed'], record.get('error', ''), flush=True)
    if all(m['physical_screen_passed'] for m in report['meshes']):
        coarse, fine = report['meshes']
        report['relative_mesh_differences'] = {
            key: float(np.linalg.norm(np.array(coarse[key])-fine[key])/max(np.linalg.norm(fine[key]),1.))
            for key in ('stress', 'tangent', 'energy', 'matched_curvature_Pa')}
    report['status'] = 'complete'
    atomic_json(result_path, report)


if __name__ == '__main__':
    main()
