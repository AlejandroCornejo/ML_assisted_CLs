# Resumen: mejores predicciones ICKAN vs ICNN

Este directorio junta las mejores predicciones que tenemos hasta ahora para el test de reproduccion de la trayectoria 3:

- `ICKAN_best_prediction/`: mejor resultado con ICKAN.
- `ICNN_best_prediction/`: mejor resultado con ICNN.

No se modificaron los scripts originales de Alejandro en la raiz del repositorio. Todo se hizo dentro de `Sebastian_ICKAN_Tests`.

## Que problema probamos

Usamos datos FOM de:

```text
stage_1_training_set_fom
```

El test fue deliberadamente simple:

```text
entrenar con trayectoria 3
predecir trayectoria 3
```

La idea fue comprobar primero si el modelo constitutivo puede reproducir una trayectoria conocida antes de exigirle generalizacion a trayectorias nuevas.

## Que usamos como input

El mejor resultado usa:

```text
strain source: applied_strain
input mode: principal
```

El dato macro inicial es:

```text
[E_xx, E_yy, gamma_xy]
```

Con eso reconstruimos:

```text
C = I + 2E
```

Luego calculamos los principal stretches:

```text
lambda_i = sqrt(eigenvalue_i(C))
```

Para `order_stretches = 1`, la red recibe aproximadamente:

```text
[lambda_bar_1, lambda_bar_2, log(J)]
```

donde:

```text
J = sqrt(det(C))
lambda_bar_i = J^(-1/3) lambda_i
```

Es decir: la red no recibe directamente `E_xx, E_yy, gamma_xy`, sino variables derivadas que separan deformacion isocorica y cambio volumetrico.

## Mejor ICKAN

Directorio original:

```text
Sebastian_ICKAN_Tests/ICKAN_prediction_traj3_3000_appliedstrain_principal_grid30_width541_blended010_W001_adam600_poststep
```

Configuracion:

- Modelo: ICKAN.
- Input: principal features desde `applied_strain`.
- Trayectoria: 3.
- Muestras de entrenamiento: 3000.
- Arquitectura: `5,4,1`.
- Grid size: `30`.
- Loss de tensiones: blended.
- Peso de loss por componente: `0.10`.
- Peso de energia: `0.01`.
- Optimizacion: Adam warmup fue clave; LBFGS puro era mas inestable.

Metricas:

| Metrica | Valor |
|---|---:|
| Global relative L2 | `6.2709e-02` |
| Sxx relative L2 | `5.9046e-02` |
| Syy relative L2 | `6.6090e-02` |
| Sxy relative L2 | `4.4380e-01` |

## Mejor ICNN

Directorio original:

```text
Sebastian_ICKAN_Tests/ICNN_prediction_traj3_full_from1000resume_appliedstrain_principal_width323216_blended010_W001_lowLR
```

Configuracion:

- Modelo: ICNN.
- Input: principal features desde `applied_strain`.
- Trayectoria: 3.
- Muestras de entrenamiento: 1000.
- Arquitectura: `32,32,16`.
- Activacion: `softplus`.
- Convexidad: pesos hidden-to-hidden no negativos.
- Loss de tensiones: blended.
- Peso de loss por componente: `0.10`.
- Peso de energia: `0.01`.
- Optimizacion:
  - Adam warmup.
  - Refinamiento con LBFGS.
  - Segundo refinamiento con learning rate menor.

Metricas:

| Metrica | Valor |
|---|---:|
| Global relative L2 | `2.6145e-02` |
| Sxx relative L2 | `2.7186e-02` |
| Syy relative L2 | `2.5043e-02` |
| Sxy relative L2 | `1.6228e-01` |

## Que cambiamos respecto a los primeros intentos

Los primeros intentos con ICKAN eran muy sensibles al grid y a LBFGS. Algunos grid updates disparaban la loss o daban NaNs.

Los cambios importantes fueron:

- Usar `applied_strain` como input controlado.
- Generar plots contra `applied_strain` y contra `strain`, porque no son lo mismo.
- Usar principal stretches como input.
- Usar Adam warmup antes de LBFGS.
- Usar una loss blended:

```text
L_stress = (1 - alpha) L_global + alpha L_component
alpha = 0.10
```

Esto evita que Sxy quede completamente ignorado, pero sin sacrificar demasiado Sxx y Syy.

- Agregar una loss pequena de energia:

```text
L_total = L_stress + beta L_energy
beta = 0.01
```

La energia ayuda a que el potencial aprendido sea mas razonable, pero no domina el ajuste de tensiones.

## Conclusion principal

Para este test de reproduccion de la trayectoria 3:

```text
ICKAN global L2: 6.27e-02
ICNN global L2 : 2.61e-02
```

La ICNN mejora claramente Sxx, Syy y Sxy. Sxy sigue siendo la componente mas dificil porque su escala fisica es mucho menor que Sxx y Syy; por eso un error absoluto pequeno se convierte en un error relativo grande.

Nota teorica: la ICNN garantiza convexidad respecto a los inputs que le damos. Como aqui usamos features basadas en principal stretches, eso no equivale automaticamente a una prueba completa de policonvexidad fisica en F. Aun asi, como modelo de energia convexa controlada, funciono mejor que la ICKAN en este test.
