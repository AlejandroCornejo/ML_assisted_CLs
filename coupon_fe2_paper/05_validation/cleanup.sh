#!/usr/bin/env bash
# Archive superseded experiments. Nothing is deleted: everything moves into
# attic/, because these scripts are the record of how each reported number was
# obtained and MAW_FINDINGS.md cites them by name.
#
# WHAT SURVIVES IN PLACE, and what each is for:
#   ecm_supports.npz       the one HPROM      (135 residual + 73 stress, 39-mode basis)
#   maw_phase2_sig.npz     MAW-10 stress rule (support + trained field)
#   maw_res_long10.npz     MAW-10 residual rule (support + trained field)
#   timing_hierarchy.npz   the final hierarchy table
#   full_integrand.npz     per-element integrand, all 4950 states (regenerable in 48 s)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p attic

KEEP_NPZ="ecm_supports.npz maw_phase2_sig.npz maw_res_long10.npz timing_hierarchy.npz full_integrand.npz"
KEEP_PY="maw_lab.py numpy_decoder.py reduced_mesh.py linear_prom.py hprom_full.py hprom_ann.py timing_hierarchy.py deploy_maw_full.py build_full_integrand.py sweep_phase2.py train_maw_fields.py cleanup.sh"

moved=0
for f in *.npz; do
  case " $KEEP_NPZ " in *" $f "*) continue;; esac
  mv -- "$f" attic/ && moved=$((moved+1))
done
for f in *.py; do
  case " $KEEP_PY " in *" $f "*) continue;; esac
  mv -- "$f" attic/ && moved=$((moved+1))
done

# Reduced meshes are pure scratch: ReducedAssembly rewrites them on every
# construction, so they carry no information worth keeping.
rm -f -- *.mdpa
echo "moved $moved files to attic/, removed the scratch .mdpa"
