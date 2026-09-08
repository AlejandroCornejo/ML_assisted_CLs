# Memoria de cálculo: los cuatro tiers PANN sobre la RVE del cupón

Documento de traspaso. Está escrito para que otra persona — o otro agente — pueda
retomar esto sin repetir lo ya medido, y sobre todo sin repetir los errores.
Todos los números son medidos, no estimados, y donde algo es inferencia y no
medida está dicho explícitamente.

Fecha del estado descrito: 2026-09-04.
Directorio: `coupon_fe2_paper/06_pann/`.

**Resultado más reciente, 2026-09-05:** ambos modelos ya entrenan con buena
precisión en esta RVE: **ICNN 0.649% e ICKAN 0.655% de error relativo de tensión
en test**, con ~0.215% de error de energía. Son variantes nuevas con features
entrenables, normalización afín en J y barrera independiente; la ICKAN usa un
spline C2 local corregido. Conservan la construcción policonvexa y los pesos
seleccionados pasan también un certificado suficiente de energía no negativa.
El error de extrapolación `probe` sigue en ~7%; todavía falta el ensayo macro
FE2. Ver [FLEXIBLE_TRAINING.md](FLEXIBLE_TRAINING.md) para resultados,
derivaciones, comandos y checkpoints. No confundir estos resultados con los
checkpoints originales de este documento histórico.

**Diagnóstico inicial revisado, 2026-09-05:** [ENRICHMENT_AUDIT.md](ENRICHMENT_AUDIT.md).
Las mediciones históricas de este documento se conservan, pero la interpretación
y las propuestas de las secciones 7–9 quedan parcialmente superadas. Se ha
demostrado una restricción de la arquitectura original (`D11=D22`,
`D13+D23=0`) que esta RVE no satisface. Añadir más sistemas balanceados con
las mismas potencias no la elimina. La extensión local con potencias distintas
redujo inicialmente el error de tensión de test a ~6.46%, conservando una
construcción policonvexa; esa primera etapa quedó superada por el resultado
~0.65% descrito arriba. También se documenta
un problema independiente en las colas del spline ICKAN original. No se han
modificado los modelos compartidos, los manuscritos ni los checkpoints previos.
No se debe rechazar `J<1` automáticamente a partir del diagnóstico anterior.

---

## 1. Qué se está intentando

Ajustar una **ley constitutiva macroscópica** que sustituya a la RVE en un cálculo
FE². La RVE es una celda con un agujero elíptico rotado (aspecto 2, 30°, 20% de
porosidad), material matriz neo-Hookeano (E = 1.628 GPa, ν = 0.4), deformación
plana, condiciones periódicas.

La ley toma la deformación macroscópica de Green-Lagrange y devuelve tensión (y,
en tres de los cuatro tiers, energía):

```
E = [E11, E22, gamma12]  ->  W(E),  S(E) = dW/dE
```

Se comparan cuatro arquitecturas, en orden creciente de estructura impuesta:

| tier | clase | potencial | policonvexo | módulo |
|---|---|---|---|---|
| `regression` | MLP directo E -> S | **no** | no | `anisotropic_pann_model_regression_claude.py` |
| `free` | MLP potencial sin restricciones | sí | **no** | `anisotropic_pann_model.py` |
| `icnn` | ICNN + invariantes de minores direccionales | sí | **sí** | `anisotropic_pann_model.py` |
| `ickan` | ídem con núcleo KAN | sí | **sí** | `anisotropic_pann_model_ickan_claude.py` |

Las clases de modelo se **importan sin modificar** del proyecto anterior
(`RVE_NeoHookean_Homogenization/pann/anisotropic/`). Sólo el bucle de
entrenamiento es nuevo (`train_pann.py`), porque los entrenadores originales
están atados a datos estructurados en diez trayectorias y lo guardan
explícitamente:

```python
if set(np.unique(trajectory_ids)) != set(range(1, 11)):
    raise RuntimeError("The final anisotropic models must use all ten Stage-1 trajectories.")
```

Los datos del cupón vienen de una caja de muestreo, no de trayectorias.
Rellenar ids sintéticos habría satisfecho esa comprobación dejándola sin
sentido, así que se escribió un bucle nuevo en lugar de falsificar la entrada.

---

## 2. Los datos

Fichero: `coupon_fe2_paper/03_data/data.npz`.

### 2.1 Conjunto de entrenamiento

**4950 estados** en una rejilla regular de paso `0.0115` sobre la caja

```
blo = [ 0.00000, -0.09865, -0.16807]
bhi = [ 0.19550,  0.01635,  0.10793]
```

Por estado se resuelve la RVE completa (1546 elementos T6, 6320 gdl) y se
almacena:

- `E_train` (4950, 3) — deformación macroscópica impuesta
- `S_train` (4950, 3) — segundo Piola-Kirchhoff homogeneizado, `F̄⁻¹⟨P⟩`
- `W_train` (4950,) — densidad de energía homogeneizada

Magnitudes: `W ∈ [1.624e+04, 2.704e+07]` (rango **1665×**),
`max|S| = 2.350e+08`.

### 2.2 De dónde sale la caja

De la **prepasada macro** (`01_macro_prepass/`): se resuelve el cupón ASTM D638
con una ley SVK aproximada, se recogen los estados de deformación en **todos los
puntos de Gauss y todos los pasos de carga** (60480 estados, `prepass_cloud_svk.npz`),
se toma su envolvente rectangular y se ensancha con `ENVELOPE_MARGIN = 0.40`,
mitad del margen por lado.

**Este paso tiene un defecto de diseño, documentado en la sección 6.**

### 2.3 Qué es `test` y qué es `probe`

Definidos en `02_sampling/build_eval_sets.py`. Son los dos conjuntos de
evaluación y **ninguno de los dos se usa para entrenar ni para seleccionar**.

**`test` — 400 estados, DENTRO del sobre.** Muestreo aleatorio dentro de la
caja, en posiciones que no caen en la rejilla de entrenamiento. Mide
interpolación: qué hace el modelo entre los puntos que vio.

**`probe` — 350 estados, FUERA del sobre.** Construido deliberadamente para
salir de la caja, en **anillos** de sobrepaso crecientes

```
RINGS    = (1.05, 1.1, 1.25, 1.5, 2.0)
PATTERNS = ((0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2))
N_PER_PATTERN = 10
```

O sea: se escala la distancia al centro de la caja por 1.05, 1.1, 1.25, 1.5 y
2.0, en las siete combinaciones de direcciones (una sola componente, pares,
y las tres a la vez). 5 anillos × 7 patrones × 10 estados = 350. Mide
**extrapolación**, que es la mitad del mensaje del paper.

De los 350, cinco no tienen solución convergida en el FOM y se excluyen por el
filtro `isfinite`; quedan 345. Están en la esquina fuertemente compresiva,
declarada fuera de alcance desde el diseño.

### 2.4 La convención de Voigt, verificada

Los modelos esperan `[E11, E22, gamma12]` con `gamma12 = 2·E12`, y devuelven la
tensión como `dW/d(esa parametrización)`. Como

```
S : dE = S11 dE11 + S22 dE22 + S12 dgamma12
```

la derivada respecto a `gamma12` es `S12` tensorial, que es exactamente lo que
`homogenized_stress` almacena en la tercera componente. Eso es el álgebra; un
desajuste ahí sería **silencioso** (sólo números peores), así que se comprobó
numéricamente aprovechando que los estados están en rejilla:

| componente | pares | error rel. máx | error rel. mediano |
|---|---:|---:|---:|
| E11 | 4400 | 1.331e−03 | 5.434e−04 |
| E22 | 4050 | 1.325e−03 | 7.797e−04 |
| **gamma12** | 4554 | 2.191e−04 | **7.068e−05** |

Diferencias centradas de `W_train` contra `S_train`. Los errores son
truncamiento de segundo orden con paso 0.0115, no desajuste — un factor 2 mal
daría ~100%. Y la tercera componente es la **más** precisa, que es lo que
confirma que `S_train[:,2]` es el conjugado de `gamma12`.

Implementado en `train_pann.py::check_voigt_convention`, se ejecuta antes de
cada entrenamiento y aborta si falla.

---

## 3. El escalado, que es donde se cometieron dos errores

Los modelos derivan la energía respecto a la deformación **normalizada**, así
que emiten cantidades normalizadas y los objetivos deben normalizarse igual.
La receta correcta, copiada del entrenador validado
(`train_anisotropic_pann_claude.py`, líneas 212-216):

```python
strain_scale  = max|E|                      # 0.1955
energy_scale  = max|W|                      # 2.7043e+07
x             = E / strain_scale
energy_target = W / energy_scale
stress_target = S * (strain_scale / energy_scale)
```

Los denominadores de las dos pérdidas se toman de los objetivos normalizados,
`mean(target²)`, de modo que ambos términos son relativos y comparables sin
pesos ajustados a mano.

Al evaluar hay que volver a unidades físicas: `S_fis = S_modelo · (energy_scale / strain_scale)`.

**Error 1 (corregido).** La primera versión comparó la salida del modelo contra
`W` y `S` **físicos**. Los objetivos quedaron desfasados por factores de
`energy_scale` y `energy_scale/strain_scale`, nada podía ajustar, y los tiers
`regression` y `free` reportaron error relativo exactamente `1.0000e+00` — la
firma de una predicción esencialmente nula. Los tiers `icnn` e `ickan` no
llegaron a entrenar: reventaron con `feature_scale must contain 15 positive
entries` porque se les pasó el vector de 4 del modelo `free`.

**Error 2 (corregido, y es el importante).** Ver sección 5.

---

## 4. Los hiperparámetros

Tomados del entrenador validado por tier, **no elegidos aquí**:

```python
BUDGET = {
    "regression": dict(epochs=350, batch=2048, lr=4.0e-4),
    "free":       dict(epochs=350, batch=2048, lr=4.0e-4),
    "icnn":       dict(epochs=700, batch=4096, lr=2.0e-3),
    "ickan":      dict(epochs=700, batch=4096, lr=2.0e-3),
}
WEIGHT_DECAY = 1.0e-9   # AdamW
GRAD_CLIP    = 50.0
```

La primera versión usó 20000 épocas a lote completo, `lr = 2e-3` para todos y
Adam desnudo sin recorte. Tres errores simultáneos: ~14× más pasos de los
necesarios, learning rate 5× demasiado alto para `free` (que divergía — mejor
época 698 de 3699), y sin recorte de gradiente. Con la receta validada
`regression` entrena en **4 segundos** en lugar de 184.

**Selección de modelo:** el mejor estado se guarda por la pérdida de validación
sobre una fracción retenida del conjunto de **entrenamiento** (15%,
`SPLIT_SEED = 5`). Los conjuntos `test` y `probe` no intervienen ni en la
selección ni en el scheduler.

---

## 5. El error de raíz: broadcasting en la pérdida de energía

Este es el hallazgo central del documento y explica casi todo lo observado
antes de corregirlo.

`model.energy_and_stress(x)` devuelve `wp` con forma **(n, 1)**. El objetivo
`Wt[idx]` tiene forma **(n,)**. Entonces:

```python
torch.mean((wp - Wt[idx]) ** 2) / w_den      # <-- MAL
```

`(n,1) − (n,)` **no resta elemento a elemento**: hace broadcasting a **(n,n)**.
Se promedian las `n²` diferencias cruzadas entre la energía predicha de cada
estado y la energía objetivo de *todos los demás*.

Comprobación mínima:

```python
wp = torch.tensor([[1.],[2.],[3.]])   # (3,1)
wt = torch.tensor([1., 2., 3.])       # (3,)
torch.mean((wp - wt)**2)              # -> 1.3333   (deberia ser 0.0)
torch.mean((wp.reshape(-1) - wt)**2)  # -> 0.0
```

El arreglo es `wp.reshape(-1)`.

### 5.1 Qué consecuencias tuvo

**La pérdida de energía quedaba clavada en ~0.5 para cualquier modelo**, bueno o
malo, porque medía la dispersión de `W` y no el error. Y su gradiente, que es
basura, envenenaba el ajuste de tensión — que es el término que sí importa.

Efecto medido, tier `free`, misma arquitectura y presupuesto:

| configuración | pérdida tensión | pérdida energía | error en `test` |
|---|---:|---:|---:|
| stress + energy, **con el bug** | 3.79e−02 | 4.98e−01 | **1.97e−01** |
| stress only (término eliminado) | 7.47e−05 | 6.04e−01 † | 7.52e−03 |
| stress + energy, **corregido** | 2.13e−04 | 1.19e−05 | **1.25e−02** |

† la columna de energía sigue mostrando el valor con bug en esa fila porque el
término no participaba en la pérdida; el modelo era el mismo.

**Factor 16 en el error, factor 42000 en la pérdida de energía.**

### 5.2 Cómo se localizó, y por qué tardó

El síntoma que lo delató es una **contradicción física**. Con la pérdida mal:

- el modelo ajustaba la tensión al 0.86% (`stress only`),
- su tensión por autograd coincide con la diferencia finita de su **propia**
  energía a **7e−10** (control incluido en `diag_energy_path.py`),
- `W(0) = 0` está impuesto por construcción y los datos cumplen `W_zero = 0`,
- los datos cumplen `S = dW/dE` a 5e−04 (sección 2.4).

Si las cuatro cosas son ciertas, integrar la tensión da la energía, y la
energía **tiene** que ajustar a ~1%. Reportaba 0.60. Una de las premisas era
falsa, y no era ninguna de las cuatro: era la medida.

**Lo que costó tiempo fue no usar la evidencia que ya estaba en la tabla.** El
tier `free` no tiene policonvexidad, ni monotonía, ni ICNN, y también fallaba
(1.74e−01, 21× peor que la regresión). Eso ya descartaba la arquitectura como
causa desde el primer resultado. En vez de leerlo, se persiguieron tres
hipótesis sobre la arquitectura y los datos. Las tres se midieron **con el
gradiente envenenado**, así que ninguna de esas medidas dice lo que pretendía:

- barrido de anchura (79× parámetros): inválido, repetido en 7.2
- "déficit de amplitud del 19%" (ratio 0.82): era mayormente el bug; con la
  pérdida correcta el ratio es **0.965**
- restringir el entrenamiento a J ≥ 1: inválido

También se atribuyó a "ruido de minitanda de 111 muestras" una discrepancia
entre el log (5.59e−01) y la medida real (4.12e−02). **Falso**: era este bug.
La ruta de reporte usaba `.ravel()` y estaba bien; la de entrenamiento no.

---

## 6. El defecto de la caja de muestreo (real, pero no era la causa)

Independiente del bug anterior y sin corregir todavía.

| magnitud | nube real del cupón (60480 pts de Gauss) | caja de muestreo |
|---|---|---|
| E11 | +0.0096 .. +0.1505, **0% negativo** | 0.0000 .. +0.1955 |
| E22 | −0.0711 .. −0.0046, **100% negativo** | −0.0987 .. +0.0163 |
| gamma12 | −0.0767 .. +0.0190 | −0.1681 .. +0.1079 |
| **J** | **1.0049 .. 1.0577** | **0.9037 .. 1.1844** |
| **J < 1** | **0.0% de los estados** | **28.3% de los estados** |

**El cupón nunca entra en compresión volumétrica.** La contracción de Poisson en
`E22` queda siempre más que compensada por la extensión en `E11`, así que
`det F ≥ 1.0049`.

Ensanchar una envolvente **rectangular** componente a componente alcanza
esquinas que la nube nunca visita — `E11` bajo con `E22` muy negativo y cortante
grande — y esas esquinas **cruzan J = 1**, que no es una versión más ancha del
mismo régimen sino otro régimen: cierre de poro en lugar de apertura.

**Alcance del defecto:**

- **Benigno para el ROM.** La base POD, el decodificador y los campos de pesos
  MAW cubren más de lo necesario; es desperdicio, no error. Los tres modelos
  reducidos se validaron contra `test` y `probe`, no contra la caja.
- **Relevante para los tiers de energía**, por la sección 7.3.
- **No era la causa del fallo de los PANN**: restringir el entrenamiento a J ≥ 1
  lo **empeoró** un 21% (2.63e−01 -> 3.18e−01), porque se pierden 1400 estados y
  la evaluación sigue siendo sobre el conjunto completo.

Recomendación: rehacer la caja rechazando puntos con J < 1 (o con J por debajo
del mínimo de la nube menos margen), manteniendo el margen del 40% en las
direcciones que no cambian de régimen. Eso obliga a regenerar la etapa 03 y todo
lo entrenado sobre ella.

---

## 7. Resultados con la pérdida corregida

### 7.1 Los cuatro tiers

Presupuesto validado, selección sobre retenido de entrenamiento, evaluación
sobre `test` (400 estados dentro del sobre) y `probe` (345 fuera). Error
relativo de Frobenius de la tensión contra el FOM.

| tier | params | dentro del sobre | fuera del sobre | degradación |
|---|---:|---:|---:|---:|
| `regression` | 25475 | **8.0945e−03** | 2.7072e−02 | 3.3× |
| `free` | 25473 | **1.2643e−02** | 5.5024e−02 | 4.4× |
| `icnn` | 1385 | 2.2781e−01 | 2.8868e−01 | **1.3×** |
| `ickan` | 2512 | 1.8326e−01 | 2.2885e−01 | **1.2×** |

Comparación con los valores previos al arreglo: `free` mejoró **14×**
(1.74e−01 -> 1.26e−02); `icnn` un 1.2× y `ickan` un 1.3×.

**Lectura.** El arreglo rescató por completo el potencial sin restricciones:
`free` es termodinámicamente consistente (`W` bien definida, `S = ∂W/∂E` exacta
por construcción) y alcanza casi la precisión del ajuste directo. Las dos
policonvexas siguen 18× y 14× por encima de la regresión.

Nótese además la inversión en la última columna: los tiers precisos degradan
3-4× fuera del sobre y los policonvexos **1.2-1.3×**. Es poca base para
concluir, porque parten de un error mucho peor, pero apunta en la dirección del
mensaje del paper.

### 7.2 Barrido de anchura de la ICNN (repetido con la pérdida correcta)

| anchuras | params | pérdida tensión | pérdida energía | ratio amplitud | `test` |
|---|---:|---:|---:|---:|---:|
| (24, 24) | 1385 | 5.3641e−02 | 1.8756e−02 | 0.9654 | 2.2781e−01 |
| (64, 64) | 6225 | 5.3209e−02 | 1.8459e−02 | 0.9589 | 2.2663e−01 |
| (128, 128) | 20625 | 5.3214e−02 | 1.8470e−02 | 0.9528 | 2.2658e−01 |
| (128, 128, 64) | 29777 | 5.2614e−02 | 1.8405e−02 | 0.9485 | 2.2536e−01 |
| (256, 256, 128) | **108689** | 4.9324e−02 | 1.7378e−02 | 0.9513 | **2.1852e−01** |

**79× más parámetros, 4% menos error.** La capacidad queda descartada, y esta
vez con una medida válida. A 108689 parámetros la ICNN tiene **4× más** que la
regresión, que da 8.09e−03.

El ratio de amplitud es ahora **0.95-0.97** (era 0.82 con el bug), así que la
magnitud de la energía es casi correcta y lo que queda es error de **forma**.

### 7.3 El mecanismo candidato: monotonía

La ICNN es convexa y **no decreciente** en sus 15 features, que son, en el orden
en que las concatena `structural_features`:

| # | feature | expresión | convexa en |
|---|---|---|---|
| 1 | `tr(C)` | \|F\|² | F |
| 2-4 | `direct_quartic[k]` | Σᵢ wₖᵢ \|F dₖᵢ\|⁴ | F |
| 5-7 | `direct_sixth[k]` | Σᵢ wₖᵢ \|F dₖᵢ\|⁶ | F |
| 8-10 | `cofactor_quartic[k]` | Σᵢ wₖᵢ \|cof(F) dₖᵢ\|⁴ | cof F |
| 11-13 | `cofactor_sixth[k]` | Σᵢ wₖᵢ \|cof(F) dₖᵢ\|⁶ | cof F |
| 14 | `J` | det F | J |
| 15 | `J²` | | J |

con k = 1,2,3 sistemas estructurales de tres direcciones cada uno (0°/50°/130°,
20°/83°/151°, 12°/76°/143°) y pesos positivos que cumplen
`Σᵢ wₖᵢ dₖᵢ⊗dₖᵢ = I` exactamente. Ese segundo momento balanceado hace que la
primera derivada de cada invariante en `F = I` sea isótropa, de modo que la ICNN
puede pesar los sistemas libremente sin generar tensión de referencia
anisótropa; la presión isótropa común se cancela con `−r·log(J)`.

**Test de falsabilidad sobre los datos**, sin entrenar nada: si existen dos
estados con `features(A) ≤ features(B)` componente a componente pero
`W(A) > W(B)`, ninguna función no decreciente puede ajustarlos.

| recorte | n | violaciones | % de pares | peor exceso |
|---|---:|---:|---:|---:|
| toda la caja | 4950 | 2,337,075 | 9.54% | **1093.7×** |
| J ≥ 0.98 | 3973 | 979,666 | 6.21% | 252.6× |
| J ≥ 1.00 | 3551 | 719,737 | 5.71% | 12.3× |
| J ≥ 1.0049 (mín. nube) | 3406 | 662,784 | 5.71% | 11.4× |

Peor caso de toda la caja:

```
estado i:  E = [0, -0.0987, -0.1681]   W = 1.778e+07
estado j:  E = [0, +0.0048, -0.0071]   W = 1.624e+04
min(features_j - features_i) = +8.541e-02      (>= 0 en las 15)
```

El estado i tiene features **menores en las quince componentes** y **1094× más
energía**. Mecanismo: las quince features son medidas monótonas de
estiramiento; bajo compresión todas **bajan** mientras la energía **sube** —
cierre de poro con 20% de porosidad.

**Limitación importante de este test: es parcial.** El modelo completo no es sólo
una función no decreciente de las features:

```python
energy = structural(features) - structural_zero - pressure*log(J) + 0.5*c*(J-1)**2
```

Los dos términos analíticos pueden absorber parte de lo que el test cuenta como
violación. `pressure` no es libre — lo fija la condición de tensión de
referencia — y `c` es un escalar, así que la libertad es poca pero no nula.
**El test correcto sería: ¿existe c tal que `W + p·log(J) − ½c(J−1)²` sea no
decreciente en las features?** No se ha hecho.

---

## 8. Estado: qué está establecido y qué no

### Establecido por medida

1. Los datos son consistentes: `S = dW/dE` a 5e−04, `W_zero = 0`, `S_zero = 0`,
   convención de Voigt verificada componente a componente.
2. Los datos son **aprendibles**: la regresión directa alcanza 8.09e−03 con
   25475 parámetros.
3. La ruta de energía es sana: autograd contra diferencia finita de la propia
   energía del modelo a **7e−10**.
4. Un potencial **sin restricciones** alcanza 1.26e−02, casi la precisión de la
   regresión directa.
5. La capacidad **no** limita la ICNN: 79× parámetros -> 4% de mejora.
6. La caja de muestreo incluye un 28.3% de estados con J < 1 que el cupón nunca
   visita.
7. Los datos de la caja violan la monotonía del conjunto de 15 features en 9.54%
   de los pares comparables, con un caso extremo de 1094×.

### No establecido

1. **Que la policonvexidad sea la causa** del techo de la ICNN en ~2.2e−01. Es
   la hipótesis viva y la única que queda en pie, pero el único test que la
   apoya es parcial (sección 7.3).
2. Que el `feature_scale` sea adecuado. Divide por `max|feature|` y deja las
   entradas en [0.39, 1.0]; en el proyecto anterior el rango era [0.004, 1.0].
   **Nota:** añadir un desplazamiento no ayudaría, porque la primera capa de la
   ICNN es lineal y absorbe cualquier shift en su sesgo. Un cambio de escala
   por componente sí cambia el condicionamiento.
3. Que `ickan` mejore sobre `icnn` de forma significativa. Da 1.83e−01 contra
   2.28e−01 con menos parámetros, pero es un solo ajuste por arquitectura y sin
   control de semilla.
4. Nada sobre estabilidad estructural (Cook, cruciforme). No se ha desplegado
   ningún tier en un cálculo macro.

---

## 9. Qué haría a continuación, en orden

1. **El test de monotonía completo**, incluyendo `p·log(J)` y `½c(J−1)²` con `c`
   ajustado. Es barato y decide si la sección 7.3 es la causa o una correlación.
2. **Control de semilla** en los cuatro tiers. Todos los números de la sección
   7.1 son un solo ajuste. En el trabajo de MAW-ECM de esta misma sesión, un
   efecto de 3.4× que se iba a reportar resultó ser varianza de semilla.
3. **Arreglar la caja** rechazando J < 1, y regenerar la etapa 03. Independiente
   de los PANN y necesario de todos modos.
4. Si 1 confirma la monotonía: **añadir features que crezcan en compresión**,
   por ejemplo `1/J` y `1/J²`, que son convexas en J (`d²(1/J)/dJ² = 2/J³ > 0`)
   y aumentan al comprimir. Como subclase, sin tocar el modelo compartido, y
   preservando el certificado.

---

## 10. Ficheros

| fichero | qué hace |
|---|---|
| `train_pann.py` | entrena un tier; contiene la comprobación de Voigt y los presupuestos por tier |
| `sweep_icnn_width.py` | barrido de anchura de la ICNN con métricas sobre el conjunto de ajuste completo |
| `diag_energy_path.py` | aísla el término de energía contra la ruta de autograd; incluye el control autograd-vs-diferencia-finita |
| `icnn_j1.py` | comparación de tres regiones de entrenamiento (caja completa, J ≥ 1, J ≥ 1.0049) |
| `pann_regression.pt` | tier de regresión entrenado — se conserva porque es la **prueba de que los datos son aprendibles** |
| `pann_icnn.pt` | línea base de la ICNN |
| `attic/` | checkpoints del barrido inválido y de la primera tanda |

### Advertencias para quien siga

- La pérdida de energía **debe** usar `wp.reshape(-1)`. Está arreglado en los
  cuatro scripts, pero es el error que costó más tiempo aquí.
- `AnisotropicFreeEnergy.energy()` llama a `torch.autograd.grad` internamente
  para imponer `S(I) = 0`. **No se puede llamar bajo `torch.no_grad()`** —
  lanza `element 0 of tensors does not require grad`. Usar `.detach()` sobre el
  resultado.
- `feature_scale` tiene **4** entradas para `free` (features materiales de C) y
  **15** para los policonvexos (`1 + 4·3 + 2`). Derivarlas con los helpers del
  proyecto anterior: `derive_free_feature_scale` y
  `derive_polyconvex_feature_scale`.
- Los modelos emiten cantidades **normalizadas**. Volver a físicas con
  `S_fis = S_modelo · energy_scale / strain_scale`.
- El log de entrenamiento imprime la pérdida de la **última minitanda**, que con
  lote 4096 sobre 4207 estados tiene 111 muestras. No leer tendencias de ahí;
  usar las métricas sobre el conjunto completo.
