# Resumen: mejores predicciones ICKAN vs ICNN

Este directorio contiene solamente las predicciones que queremos ensenar:

- `ICKAN_best_prediction/`: mejor resultado obtenido con ICKAN.
- `ICNN_best_prediction/`: mejor resultado obtenido con ICNN.
- `checkpoints/`: checkpoints de esos dos modelos.

Los casos extra de ICNN que usamos como diagnostico fueron archivados en:

```text
_archived_extra_icnn_cases_20260709/
```

No se modificaron los scripts originales de Alejandro en la raiz del repositorio. Todo se hizo dentro de `Sebastian_ICKAN_Tests`.

## Test

El test sigue siendo deliberadamente simple:

```text
entrenar con trayectoria 3
predecir trayectoria 3
```

La idea es comprobar primero si el modelo constitutivo reproduce una trayectoria conocida antes de pedir generalizacion.

## Input usado por los mejores modelos

Usamos:

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
E = [[E_xx, gamma_xy/2],
     [gamma_xy/2, E_yy]]
```

```text
C = I + 2E
```

Despues calculamos los principal stretches:

```text
lambda_i = sqrt(eigenvalue_i(C))
```

y los stretches isocoricos:

```text
J = sqrt(det(C))
lambda_bar_i = J^(-1/3) lambda_i
```

La mejor ICKAN recibe:

```text
[lambda_bar_1, lambda_bar_2, log(J)]
```

La mejor ICNN recibe principal features de orden 2:

```text
[lambda_bar_1, lambda_bar_2, lambda_bar_1^2, lambda_bar_2^2, log(J)]
```

Esto no es exactamente igual a usar los invariantes `K1,K2,K3` del driver original de ICKAN. Esos invariantes se grafican para diagnostico, pero no son el input de los mejores modelos finales.

## ICKAN

Configuracion principal:

- Modelo: ICKAN.
- Input: principal features desde `applied_strain`.
- Trayectoria: 3.
- Muestras de entrenamiento: 3000.
- Arquitectura: `5,4,1`.
- Grid size: `30`.
- Loss de tensiones: blended.
- Peso de loss por componente: `0.10`.
- Peso de energia: `0.01`.
- Optimizacion: Adam warmup; esto fue clave para quitar los saltos raros.

Metricas:

| Metrica | Valor |
|---|---:|
| Global relative L2 | `6.2709e-02` |
| Sxx relative L2 | `5.9046e-02` |
| Syy relative L2 | `6.6090e-02` |
| Sxy relative L2 | `4.4380e-01` |

## ICNN

Configuracion principal:

- Modelo: ICNN.
- Input: principal features desde `applied_strain`.
- Trayectoria: 3.
- Muestras de entrenamiento usadas en el ajuste: 1000.
- Evaluacion/prediccion: trayectoria completa.
- Arquitectura: `32,32,16`.
- Principal stretches de orden 2.
- Activacion: `softplus`.
- Convexidad: pesos hidden-to-hidden no negativos.
- Loss final: stress global.
- Peso de loss de energia en el refinamiento final: `0.00`.
- Optimizacion:
  - Adam warmup.
  - Refinamiento con LBFGS.
  - Refinamientos con learning rate menor.
  - Early stopping mas exigente para no parar demasiado pronto.

Metricas:

| Metrica | Valor |
|---|---:|
| Global relative L2 | `1.6363e-02` |
| Sxx relative L2 | `1.5782e-02` |
| Syy relative L2 | `1.6793e-02` |
| Sxy relative L2 | `3.1687e-01` |

## Colores de los plots

```text
Reference : negro
ICKAN     : rojo
ICNN      : azul
```

Si en algun diagnostico futuro aparece un cuarto caso, usamos verde oscuro, y luego naranja.

Los PNG usan transparencia para ver mejor los solapes. Los EPS tambien se guardan, pero el backend PostScript no soporta transparencia y los renderiza opacos.

## Plots contra invariantes

Para revisar la pregunta de Riccardo, se generan plots de tensiones y energia contra invariantes tipo C en:

```text
invariant_plots_applied_strain/
invariant_plots_homogenized_strain/
```

El primero calcula los invariantes desde `applied_strain`, que es la carga macro impuesta. El segundo los calcula desde la `strain` homogenizada guardada por el FOM.

Los plots contra invariantes se hacen como scatter para evitar crear saltos visuales artificiales al conectar ramas distintas de la trayectoria.

Las figuras usan LaTeX y muestran las formulas completas:

```text
C = I + 2E
I1 = tr(C)
I3 = det(C)
J = sqrt(I3)
I1_bar = I1 I3^(-1/3)
I2_bar = (I1 + I3 - 1) I3^(-2/3)
K1 = I1_bar - 3 = I1 I3^(-1/3) - 3
K2 = I2_bar^(3/2) - 3 sqrt(3)
K3 = (J - 1)^2
```

Nota importante: `K1` no es simplemente `I1 - 3`; es `I1_bar - 3`.

## Conclusion

Para este test de reproduccion de la trayectoria 3:

```text
ICKAN global L2: 6.27e-02
ICNN global L2 : 1.64e-02
```

La ICNN es la mejor opcion actual en este test. La convexidad de la ICNN es con respecto a los inputs que le damos; como usamos principal features derivadas de `C`, esto no es automaticamente una demostracion completa de policonvexidad fisica en `F`.
