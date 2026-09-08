# Cómo correr los 8 casos del FE² del Cruciforme

Este documento explica cómo reproducir, desde cero, los 8 modelos de la
demostración FE² del especimen cruciforme (4 tiers de PANN + HPROM-FE²
+ HPROM-ANN-FE² + D-HPROM-ANN-FE² + FOM-FE²) tal como se corrieron para
generar `cruciform_fe2_table_claude.tex` y las figuras
`cruciform_deformed_field_claude.pdf` / `cruciform_vonmises_field_claude.pdf`
del paper. Todo el trabajo vive en `fe2_extension/` (los drivers/leyes)
y `pann/anisotropic/` (las figuras/tablas para el paper).

## Los 8 modelos y sus claves en `MATERIAL_FUNCS`

| # | Modelo (nombre en el paper) | Clave en `MATERIAL_FUNCS` | Necesita registro manual |
|---|---|---|---|
| 1 | Regression (tier 1) | `pann_regression` | No, ya está en el dict base |
| 2 | Free hyperelastic (tier 2) | `pann_free` | No |
| 3 | Polyconvex ICNN (tier 3a) | `pann_certified` | No |
| 4 | Polyconvex ICKAN (tier 3b) | `pann_ickan` | No |
| 5 | Linear-HPROM-FE² (= "HPROM-FE²") | `linear_hprom_parallel_continuation` | Sí, vía `make_linear_hprom_parallel_continuation_material_func(n_workers=16)` |
| 6 | HPROM--ANN-FE² | `hprom_ann_parallel_continuation` | Sí, vía `make_hprom_ann_parallel_continuation_material_func(hprom_ann_dir, n_workers=16)` |
| 7 | D-HPROM--ANN-FE² | `dhprom_ann_parallel` | Sí, vía `make_dhprom_ann_parallel_material_func(dhpromann_dir, n_workers=16)` |
| 8 | FOM-FE² (referencia verdadera) | `fom_nested_consistent_parallel` | Sí, vía `make_fom_nested_consistent_parallel_material_func(n_workers=16)` |

Los 4 tiers PANN ya están en `run_cruciform_fe2_claude.py`'s propio
`MATERIAL_FUNCS` (son funciones directas macro-a-macro, sin RVE
anidada, así que son baratísimas). Los otros 4 necesitan registrarse
explícitamente en el diccionario antes de llamarlos, cada uno con su
propia función factory (todas viven en `run_cruciform_fe2_claude.py`).

## Malla y protocolo de carga (fijos para los 8)

- `n_body=6, n_arm_len=4` → 200 elementos Tri6, 600 puntos de Gauss
  macro (la malla elegida por un estudio de convergencia de malla con
  ICNN, ver el paper, Fig. 22).
- Los 4 puntas del cruciforme se desplazan hacia afuera, cada una en su
  propio eje, simétricamente (`delta_x_final = delta_y_final = 1.2`),
  en 20 pasos de carga (`n_steps=20`).
- Tolerancia de convergencia relativa `1e-4`, con line search secante
  activado (`use_line_search=True`) para los 8, incluyendo FOM-FE².

## Los dos scripts driver

**1. `run_cruciform_nbody6_full_suite_claude.py`** — corre los 6 modelos
rápidos (4 PANN + Linear-HPROM + HPROM-ANN + D-HPROM-ANN) uno tras otro,
en un solo proceso, secuencialmente. Toma ~10 minutos en total.

```bash
cd fe2_extension
python3 run_cruciform_nbody6_full_suite_claude.py
```

**2. `run_cruciform_fom_nbody6_full_claude.py`** — corre FOM-FE² solo,
en su propio proceso. Este es el lento: la corrida real que generó los
números del paper tardó **~11.1 horas** (40093.7 s), y el costo por paso
NO es constante, crece con la deformación acumulada (cada paso hace un
Newton macro de ~4 iteraciones, y cada iteración llama al material real
en los 600 puntos de Gauss, cada uno resolviendo desde cero la RVE
completa de 990 elementos — mientras más deformada está la RVE, más
cara es esa resolución interna, aunque la CALIDAD de convergencia del
Newton macro se mantiene perfecta y estable en los 20 pasos, sin
degradarse).

```bash
cd fe2_extension
nohup python3 run_cruciform_fom_nbody6_full_claude.py > /ruta/a/tu/log_fom.log 2>&1 &
```

**IMPORTANTE — nunca corras los dos scripts al mismo tiempo, ni corras
nada más pesado en paralelo.** Cada uno arma su propio pool de 16
workers; correr dos a la vez satura la máquina y contamina los tiempos
de pared que después se reportan en el paper. Regla de esta sesión:
serial siempre, uno termina antes de que el siguiente empiece.

## Qué queda guardado

Cada corrida guarda `cruciform_results_<clave>_claude.npz` en
`fe2_extension/`, con estos campos:

- `coords`, `tris`, `u_nodal`: malla y desplazamiento nodal final.
- `e_gp`, `s_gp`: deformación y tensión (Voigt, 3 componentes) en cada
  uno de los 600 puntos de Gauss macro, en el estado final.
- `iters_per_step`, `converged_per_step`, `status_per_step`,
  `best_rel_per_step`: historial de convergencia por paso.
- `fully_converged`, `ever_diverged`: banderas globales de la corrida.
- `reaction_px`, `reaction_mx`, `reaction_py`, `reaction_my`: fuerza de
  reacción total en cada una de las 4 puntas, en el estado final
  convergido (útil porque este problema es controlado por
  desplazamiento, no por fuerza, así que no hay un "rango de
  desplazamiento en la punta" que reportar como en Cook — la fuerza de
  reacción es el escalar de salida físicamente significativo aquí).

**Antes de confiar en un `.npz` existente**, revisa que
`fully_converged == True` y que `len(iters_per_step) == 20` — un
archivo bajo el mismo nombre canónico puede ser un leftover viejo o
incompleto (p.ej. de una corrida de un solo paso, o una interrumpida a
medias). Los scripts de figuras/tabla ya hacen esta verificación
automáticamente y saltan (no truenan) cualquier archivo que no la pase.

## Cómo generar las figuras y la tabla, una vez que los 8 `.npz` existen

```bash
cd pann/anisotropic
python3 make_cruciform_deformed_field_claude.py   # -> cruciform_deformed_field_claude.pdf
python3 make_cruciform_vonmises_field_claude.py   # -> cruciform_vonmises_field_claude.pdf

cd ../../fe2_extension
python3 compare_cruciform_speedup_accuracy_claude.py   # -> imprime la tabla + escribe
                                                        #    pann/anisotropic/cruciform_fe2_table_claude.tex
```

**Advertencia sobre `compare_cruciform_speedup_accuracy_claude.py`:**
este script lee los tiempos de pared parseando los *logs* de las
corridas (no están guardados dentro de los `.npz`), buscando líneas con
el patrón `[<clave>] ... wall=X.Xs`. Las rutas de esos logs están
hard-codeadas en el script (`FAST_BATCH_LOG`, `FOM_LOG`, ambas bajo un
directorio de scratchpad específico de la sesión en la que se escribió
este script). **Si vas a re-correr esto en el futuro, redirige la
salida de tus propias corridas a esas mismas rutas (o edita esas dos
constantes al inicio del script)** — si no, el script sigue calculando
bien los errores/speedup relativo a FOM, pero mostrará `?`/`--` en las
columnas de tiempo de pared y speedup.

## Herramientas de depuración disponibles (opcionales, todas por defecto apagadas)

`run_newton_fe2_cruciform` (en `run_cruciform_fe2_claude.py`) tiene
varios parámetros opcionales, todos con default que preserva el
comportamiento exacto de antes de que existieran (ningún caller
existente cambia de comportamiento a menos que los pases
explícitamente):

- `max_steps_to_run=N`: corta la corrida después de N pasos (cada paso
  sigue usando `frac=step/n_steps`, el mismo incremento que una corrida
  completa) — útil para probar solo el primer paso sin pagar los 20.
- `iter_callback(step, it, u, res_norm, assembler)`: se llama después
  de cada `Assemble()`, con acceso a `assembler._E_voigt`/`_S_voigt`
  (el campo completo de deformación/tensión macro en ese iterado
  exacto) — útil para instrumentar y comparar trayectorias de Newton
  punto por punto entre dos leyes.
- `alpha_callback(step, it, alpha, du_free, free_dofs)`: se llama justo
  después de elegir el alpha del line search (o 1.0 si no hay line
  search) — útil para ver si el line search está siendo demasiado
  agresivo.
- `enforce_residual_decrease=True`: agrega una red de seguridad que el
  line search por sí solo no da (es una búsqueda secante sobre la
  derivada direccional, no una garantía de que `||res||` de verdad
  baje) — si el alpha elegido no baja el residuo, lo va reduciendo a la
  mitad hasta que sí lo haga, capturando incluso una excepción de
  "estado de deformación inválido" durante ese proceso en vez de
  dejarla propagarse. Verificado que es un no-op exacto (mismos alphas,
  misma convergencia) en los casos donde ya converge bien; cuesta una
  llamada extra a `Assemble()` por iteración en el caso común.

Estas herramientas quedan disponibles como infraestructura general de
depuración para el Newton macro, pero no son necesarias para reproducir
los 8 casos de este documento a `n_body=6` — los 8 convergen limpio con
los parámetros por defecto, sin necesitar ninguna de ellas.
