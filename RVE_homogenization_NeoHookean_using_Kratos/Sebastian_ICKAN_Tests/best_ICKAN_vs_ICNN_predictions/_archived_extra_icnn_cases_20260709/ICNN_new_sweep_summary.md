# ICNN Sweep Summary After First Baseline

All runs below use the same diagnostic setting:

```text
train trajectory: 3
predict trajectory: 3
strain source: applied_strain
training samples: 1000
activation: softplus
main optimizer pattern: Adam warmup + LBFGS/refinement
```

## Results

| Case | Global L2 | Sxx L2 | Syy L2 | Sxy L2 | Comment |
|---|---:|---:|---:|---:|---|
| ICKAN best old baseline | `6.2709e-02` | `5.9046e-02` | `6.6090e-02` | `4.4380e-01` | Best ICKAN reference. |
| ICNN principal order 1 old best | `2.6145e-02` | `2.7186e-02` | `2.5043e-02` | `1.6228e-01` | First good ICNN baseline. |
| ICNN hybrid | `2.4521e-02` | `2.5958e-02` | `2.2993e-02` | `1.0633e-01` | Better Sxy, but not best global. |
| ICNN orthotropic signed | `2.8812e-02` | `3.3316e-02` | `2.3485e-02` | `5.7047e-02` | Very good Sxy, worse global. |
| ICNN principal order 2 | `2.3981e-02` | `2.3450e-02` | `2.4476e-02` | `1.5978e-01` | Order 2 improved order 1. |
| ICNN principal order 2 + low LR | `2.1252e-02` | `2.1240e-02` | `2.1247e-02` | `1.3039e-01` | Good balanced result. |
| ICNN principal order 2 + alpha 0.05 | `2.1161e-02` | `2.1156e-02` | `2.1149e-02` | `1.3080e-01` | Best balanced result. |
| ICNN principal order 2 + global loss | `1.9549e-02` | `1.9812e-02` | `1.9224e-02` | `2.3179e-01` | Better global, worse Sxy. |
| ICNN principal order 2 + deep global refine + W loss | `1.6792e-02` | `1.6295e-02` | `1.7131e-02` | `3.3576e-01` | Strong Sxx/Syy/global. |
| ICNN principal order 2 + deep global refine, no W loss | `1.6363e-02` | `1.5782e-02` | `1.6793e-02` | `3.1687e-01` | Best global stress result. |
| ICNN principal order 2 fixed powers | `3.1064e-02` | `3.0978e-02` | `3.0905e-02` | `5.9402e-01` | More conservative features, worse fit. |
| ICNN bigger principal network | `3.2082e-02` | `3.3636e-02` | `3.0433e-02` | `1.8840e-01` | Bigger was not better. |

## Practical Choice

For a stress-focused reproduction test, the current best is:

```text
ICNN_prediction_traj3_full_from1000_appliedstrain_principal_order2_width323216_global_W000_deep_refine_frombest
```

For a more balanced fit that does not sacrifice Sxy as much, use:

```text
ICNN_prediction_traj3_full_from1000_appliedstrain_principal_order2_width323216_blended005_W001_resume_lowLR_frombest
```

The global best is likely the right candidate if the next comparison is based on equivalent stress/von Mises-like quantities dominated by Sxx and Syy. The balanced one is useful if Sxy needs to be shown explicitly.
