# Manuscript revision plan

## Plan vigente: dos materiales y un despliegue FE² — 16 September 2026

Estado: esqueleto editorial incorporado; piloto FOM y preflight de referencia/dominio
del material B ejecutados el 15 September 2026. La campaña constitutiva FOM
de B está completa y aprobada bajo sus criterios numéricos congelados;
no hay modelos entrenados;
la comparación de características fijas/aprendidas sigue pendiente.
La receta exacta de entrenamiento y la tabla compartida de 32 características
ya están congeladas. Las 15 inicializaciones y los seis pares fijo/aprendido
aprueban sus comprobaciones; el ejecutor B reanudable pasó su smoke test.
La campaña de entrenamiento se pausó para revisar el criterio de parada:
ocho corridas completas bajo receta v1, tres reanudables y cuatro sin iniciar;
ninguna evaluación de test/trayectorias se ha abierto.
El preflight de la caja original aprueba su comprobación finita de malla.
La posterior ampliación conjunta de tracción y cortante no pasó: dos esquinas
incumplen criterios y una presenta cruces de contorno. Una candidata separada,
que amplía solo la tracción normal positiva a 0.20 y conserva el cortante
original, sí pasó su preflight finito; esto no valida todo el dominio continuo.
La malla de 4621 elementos se recomienda como referencia de trabajo con
auditorías adicionales en la campaña. Los 5777 estados solicitados están
disponibles y el ensamblador aprueba 20/20 comprobaciones, incluyendo 64 pares
de mallas y 24 pares templado/frío. No es una garantía sobre el dominio continuo.
Los registros fechados al final describen revisiones anteriores.

Integración editorial realizada:

- Secciones 5--6 reorganizadas en LaTeX, con tareas pendientes visibles y
  resultados actuales identificados como material A.
- Representación reducida de A reubicada en 6.2.
- Suplemento independiente `supplementary.tex`: probe y auditoría detallada
  de curvatura preservados; resultados desfavorables y exclusiones conservados.
- Tabla principal limitada al test independiente; el generador mantiene
  separadas las tablas del cuerpo y del suplemento.
- En esa integración editorial no se generaron datos ni se ejecutaron
  entrenamientos o simulaciones FE². El piloto posterior se registra abajo.

Piloto B realizado el 15 September 2026:

- Geometría reproducible de cuatro cavidades, porosidad del 20%, matriz y
  deformación plana iguales a A; verificación geométrica y bordes periódicos.
- Ocho trayectorias predeclaradas, cuatro estados no nulos por trayectoria;
  mallas de 1088, 2340 y 4621 triángulos cuadráticos.
- Fallos de continuación preservados y trayectorias repetidas con incrementos
  menores, sin cambiar geometría ni estados físicos. Verificaciones de
  derivadas, ensamblaje independiente y elementos nativos de Kratos.
- Diferencias máximas fine/finer: tensión homogenizada 0.0774%, tangente 0.0514%,
  norma L2 microscópica 0.0486%, máximo microscópico muestreado 5.0764%.
  Este último **no cumple** el criterio declarado del 5%; no se cambió el umbral.
- Recomendación: conservar la geometría candidata, resolver la aprobación de
  la referencia y realizar un preflight biaxial/del dominio antes de congelar
  datos, particiones, métricas y presupuestos. No comenzar entrenamiento todavía.
- A, sus datos/checkpoints y el manuscrito LaTeX no se modificaron en el piloto.
  Informe y registros: [PILOT_REPORT.md](../07_material_b/reports/PILOT_REPORT.md).

Preflight posterior de B realizado el 15 September 2026:

- Geometría y FOM compartido sin cambios; predictor afín y continuación
  adaptativa usados únicamente por el controlador de B. Doce comparaciones
  contra las soluciones previas, incluidos desplazamientos microscópicos,
  aprobaron el umbral de concordancia de 1e-7.
- Nueva malla de 8961 triángulos y 18 estados predeclarados en esquinas,
  caras y carga biaxial/combinada. Todos los 50 pares de estados comparables
  aprobaron los criterios, sin fallos de estado ni incrementos rechazados.
- Diferencias máximas 4621/8961: tensión 0.0423%, tangente 0.0890%, energía
  0.0290%, estadístico de norma microscópica 0.0364% y máximo muestreado 4.0645%.
  Umbrales predeclarados: 0.1% para las tres salidas homogenizadas y
  los 2% / 5% originales para las estadísticas microscópicas.
- La tangente queda cerca del límite de 0.1%; no interpretar la diferencia
  entre mallas como cota del error exacto. El muestreo de curvatura y de
  polígonos deformados no demuestra estabilidad ni ausencia de contacto.
- Siguiente paso: congelar el protocolo de datos/comparación y su política
  de reintentos y auditoría de malla antes de generar la campaña completa.
  No se entrenó ni se modificaron A o el manuscrito LaTeX.
- Informe: [PREFLIGHT_REPORT.md](../07_material_b/reports/PREFLIGHT_REPORT.md).
  Los registros de fallos anteriores se conservan sin cambiar sus banderas.

Exploración posterior de amplitudes mayores — 15 September 2026:

- Seis nuevos estados: Eii = 0.15/0.20 en tracción X/Y y
  2 E12 = 0.12/0.16 en cortante positivo, en las mismas mallas de 4621/8961.
  Todos los pares aprobaron los criterios numéricos declarados.
- Comparación con la linealización D0 e de cada misma malla: desviaciones
  vectoriales de tensión de 24.4/25.3% en las tracciones originales y
  50.1/51.8% en los nuevos extremos. La apariencia casi lineal de la figura
  previa no implica que el rango original carezca de no linealidad.
- No se amplió la caja de entrenamiento. Si se incorporan límites mayores,
  comprobar antes sus nuevas cargas combinadas/de frontera y cortante negativo;
  tres rayas no validan la caja ampliada. El protocolo sigue sin congelar.
- Un primer error de construcción de metadatos del controlador se conserva
  como intento incompleto, no como fallo mecánico. FOM/A/LaTeX sin cambios.
- Informe: [NONLINEAR_EXPLORATION.md](../07_material_b/reports/NONLINEAR_EXPLORATION.md).

Preflight combinado de la caja ampliada — 15 September 2026:

- Candidata E11/E22 en [-0.04, 0.20], 2 E12 en [-0.16, 0.16]. Veinte
  pares en las mismas mallas; 36 estados nuevos, cuatro reutilizados con
  procedencia, 12 comprobaciones FD y cuatro repeticiones desde cero.
  Todos los objetivos se alcanzaron; las FD y repeticiones aprobaron.
- La caja completa **no se adopta**. Dos esquinas comprimidas con cortante
  máximo fallan entre mallas. En la negativa: tensión 1.0944%, tangente
  2.2531%, energía 0.1366% (límite 0.1%) y máximo PK1 muestreado 24.8497%
  (límite 5%). Los otros 18 pares aprobaron. Sin modificar umbrales.
- La malla fina tiene cruces del contorno de una cavidad en esa esquina.
  Orden de nodos verificado por conectividad FE; la subdivisión de los bordes
  cuadráticos retiene cruces. Determinantes y curvaturas muestreados positivos
  no garantizan inyectividad global. No afirmar plasticidad, pandeo ni contacto
  exacto a partir de este diagnóstico. Un incremento rechazado fue recuperado
  por reducción de paso y se conserva en el registro.
- Siguiente candidata recomendada, no aprobada: ampliar solo los límites
  normales positivos a 0.20, conservando compresión -0.04 y cortante ±0.08.
  Requiere su propio preflight combinado. El generador del proyecto anterior
  usa seis límites independientes; no es obligatorio un cubo ni amplitudes
  simétricas. No copiar sus valores a B sin verificarlos.
- Para demostrar no linealidad, incorporar referencias lineales y curvas
  FOM, no solo invocar la ley Neo-Hookean. Auditoría de datos guardados de A:
  error agregado de tensión en sus 400 test de 42.64% con C0 guardado y
  11.25% con matriz lineal ajustada solo en los 4208 fit. Esta última es un
  diagnóstico sin simetría/estabilidad impuestas, no la Regression neuronal.
  B alcanza 50–52% de desviación en los extremos axiales 0.20 y tangentes
  axiales de aproximadamente 0.40 respecto al reposo. Las métricas miden
  cosas distintas; no compararlas como si fueran errores equivalentes.
- Dominio/protocolo sin congelar; no campaña B ni entrenamiento neuronal.
  A/FOM compartido/LaTeX intactos. Informe y registros:
  [EXPANDED_BOX_REPORT.md](../07_material_b/reports/EXPANDED_BOX_REPORT.md).

Preflight de la caja asimétrica — 15 September 2026:

- Candidata E11/E22 en [-0.04, 0.20], 2 E12 en [-0.08, 0.08]: se amplió
  solamente el límite normal positivo. Dieciocho pares de estados cubren las
  ocho esquinas, caras y cargas biaxiales/mixtas; 30 soluciones fueron nuevas
  y seis extremos exactos se reutilizaron con procedencia.
- Todos los estados, 12 controles FD y cuatro repeticiones desde cero pasaron.
  Máximos 4621/8961: tensión 0.0423%, tangente 0.0890%, energía 0.0290%,
  PK1 RMS 0.0364% y máximo PK1 muestreado 4.0645%. Los peores siguen siendo
  estados antiguos; los nuevos no introducen un máximo global.
- No hubo incrementos rechazados. Los controles de residuo, J microscópico,
  saltos periódicos, polígonos y curvatura muestreada pasaron. La tangente
  continúa cerca del umbral de 0.1%; mantener auditorías de malla durante la
  campaña. No convertir el muestreo en prueba de estabilidad o unicidad.
- La evidencia permite adoptar estos límites al congelar el protocolo de B.
  No rehabilita la caja fallida con cortante ±0.16 ni demuestra todos sus
  puntos intermedios. Todavía no se generó campaña ni se entrenaron modelos.
  Informe: [ASYMMETRIC_BOX_REPORT.md](../07_material_b/reports/ASYMMETRIC_BOX_REPORT.md).

### Pregunta y alcance

La contribución central es la energía policonvexa con características
direccionales emparejadas. La evidencia se organizará para responder:

- ¿Representa respuestas anisótropas de dos geometrías periódicas distintas?
- ¿Qué aporta aprender las características frente a fijarlas de antemano?
- ¿Cómo se comporta la aproximación dentro de una estructura, para el material
  cuyo despliegue ya se ha evaluado?

Material A: celda actual con una cavidad elíptica; conservar sus datos,
checkpoints, auditorías y comparación PROM/HPROM/FE².
Material B: nueva celda periódica con cuatro cavidades elípticas diferenciadas;
evaluación FOM y constitutiva, sin nueva campaña PROM/HPROM ni FE² por ahora.
Cada material tendrá modelos entrenados por separado. Esto no demostrará
transferencia entre geometrías, representatividad estadística de un material
aleatorio ni universalidad de la familia de energías.

### Esqueleto propuesto del manuscrito

**5. Constitutive assessment on two periodic microstructures**

| Bloque | Función y evidencia prevista | Situación actual |
|---|---|---|
| 5.1 Microstructures and reference-model verification | Geometrías A/B, ley del sólido, periodicidad, mallas y comprobaciones FOM | A disponible; preflight y auditoría de campaña B aprobados bajo sus criterios finitos |
| 5.2 Data domains and learning protocol | Cómo se muestrea, particiones, trayectorias de test y selección de modelos; figura de cobertura | B: protocolo, etiquetas, receta exacta y tabla común de 32 características cerrados; ejecutor reanudable y entrenamiento pendientes |
| 5.3 Constitutive response and predictive accuracy | Curvas FOM/modelos en trayectorias declaradas y errores independientes para A/B | Métricas A y diagnósticos lineales disponibles; campaña B completa, modelos y curvas predictivas pendientes |
| 5.4 Contribution of learned paired features | Comparación controlada de características fijas/aprendidas, preferiblemente en A/B | Pendiente en ambos materiales |
| 5.5 Mechanical consistency and scope | Integrabilidad, referencia, derivadas y condición suficiente de energía no negativa; límites de las conclusiones | Auditorías A disponibles; B pendiente |

Estos bloques son un esquema de lectura, no una obligación de producir cinco
figuras ni cinco subsecciones si la evidencia final permite una organización
más compacta. Los títulos finales se decidirán con los resultados.

**6. Structural deployment for material A: accuracy and online cost**

Mantener el cupón existente y su referencia FOM--FE². Reubicar aquí la
representación reducida específica de A (POD, cierre y coordenadas primarias),
antes de los soportes de cubatura y de las comparaciones estructurales.
Conservar las limitaciones actuales de tiempos, recursos e iteraciones;
el segundo RVE no subsana esas limitaciones.

**Material suplementario**

Trasladar de forma coordinada la figura de probe rings, sus métricas y el
detalle de extrapolación. Mantener en el cuerpo una referencia y la conclusión
de que la admisibilidad no garantiza precisión fuera del dominio, incluida
la ventaja observada de Free en el error agregado de probe.
La búsqueda extrapolativa de curvatura negativa sin FOM convergido no será
una evidencia central de ventaja física; preservar el resultado y su alcance.
No eliminar etiquetas no finitas ni sus exclusiones del registro.

### Secuencia de trabajo y criterios para avanzar

**1. Fijar el diseño del material B y realizar un piloto FOM.**

Piloto y preflight ejecutados; comprobación finita de referencia aprobada.
Dominio/protocolo congelados y auditorías de campaña aprobadas; véase
[CAMPAIGN_REPORT.md](../07_material_b/reports/CAMPAIGN_REPORT.md).
La validación predictiva del segundo material aún requiere entrenamientos.

- Especificar centros, semiejes y ángulos de cuatro cavidades distintas en una
  celda periódica. No reutilizar una mera repetición 2 x 2 del material A.
- Mantener inicialmente la ley del sólido, deformación plana y porosidad
  total del 20 %. Verificar separación de cavidades y sus imágenes periódicas,
  conectividad del sólido, calidad de malla y correspondencia de bordes.
- Declarar unas pocas trayectorias axiales, de cortante de ambos signos y
  combinadas. Comprobar respuesta homogenizada, campos, derivadas y
  sensibilidad a malla y continuación antes de producir el conjunto completo.
- La geometría es candidata, no un caso validado. La complejidad visual no
  demuestra dificultad constitutiva y convergencia de Newton no certifica
  estabilidad. Registrar fallos y revisiones del piloto; no escoger la
  geometría por favorecer a un modelo aprendido.

Entregable: geometría reproducible, diagnóstico FOM y dominio candidato.
Si aparecen contacto o bifurcaciones no tratados por el modelo, detener la
campaña masiva y decidir el alcance; no interpretar no convergencia como
prueba de inestabilidad física.

**2. Congelar el protocolo de datos y comparación.**

Muestreo, particiones, modelos, semillas, métricas y presupuestos máximos
congelados. La receta exacta se completó después del ensamblaje FOM y antes
de seleccionar características o entrenar; no afirmar que todos sus detalles
numéricos quedaron fijados antes de generar etiquetas.

- Justificar el dominio de B a partir del piloto y del propósito constitutivo;
  no copiar automáticamente la caja derivada del cupón de A.
- Fijar muestreo, número de estados, particiones y trayectorias independientes
  antes de consultar los errores de test. Documentar estados fallidos y
  máscaras comunes entre modelos. El test final no selecciona checkpoints.
- Comparación principal en B: Free, ICNN e ICKAN. Regression puede conservarse
  como testigo de integrabilidad en A; no es necesario repetir toda la
  comparación reducida para B.
- Definir errores de energía, tensiones y tangentes, normalización y tratamiento
  de referencias próximas a cero; además del agregado, informar dispersión
  y errores por componentes cuando aporten información.
- Acordar presupuesto de optimización y varias inicializaciones antes de los
  entrenamientos. Las comparaciones históricas de A seguirán identificadas
  como no equiparadas hasta disponer de nuevas ejecuciones controladas.

Entregable: protocolo con tamaños, semillas, métricas y reglas de selección.
Los valores concretos están en DATA_PROTOCOL.md y su JSON congelado.

**3. Generar datos y ejecutar la comparación fija/aprendida.**

Datos completos y aprobados; tabla común seleccionada con fit + referencia;
comparación neuronal pendiente. No confundir aprobación de etiquetas o
inicializaciones con precisión de modelos.

- Mantener mismo núcleo, número de características, datos, objetivo y
  presupuesto entre ambas variantes. Fijar previamente una configuración
  admisible y documentada de orientaciones/exponentes para la variante fija.
  No recuperar una representación histórica interna como supuesto referente
  publicado ni elegir la configuración fija usando el test.
- Aplicar a ambas variantes la normalización y los términos de crecimiento
  correspondientes. Registrar diferencias de parámetros entrenables; mismo
  ancho y mismo presupuesto no significan idéntico número de parámetros.
- La comparación conjunta fija/aprendida mide el efecto conjunto de aprender
  orientaciones y exponentes; no atribuye por separado el beneficio a cada uno.
- Ejecutar preferentemente en A y B. Si el coste obliga a limitarla a B,
  restringir expresamente la conclusión. No prometer una mejora antes de medirla.
- Conservar geometrías, particiones, configuraciones, checkpoints, fallos y
  métricas por semilla separados de los artefactos existentes.

**4. Construir las figuras que respondan las preguntas.**

- Geometrías y verificación: presentación conjunta A/B, sin duplicar la
  descripción de la ley de matriz.
- Cobertura: dominio, muestreo y test; declarar cortes/submuestreo y evitar que
  la superposición de proyecciones oculte la estructura tridimensional.
- Respuesta: trayectorias multiaxiales predefinidas y curvas FOM/modelos.
  Inventariar primero datos disponibles; no unir estados dispersos como si
  fueran una trayectoria ni inventar etiquetas para completarla.
- Comparación fija/aprendida: tabla o figura compacta con variabilidad entre
  inicializaciones. Conservar resultados neutros o desfavorables.
- Propiedades mecánicas: conservar la distinción entre demostración,
  comprobación numérica de implementación y cota suficiente para pesos
  guardados. Una auditoría de A no valida el checkpoint de B.

**5. Integrar el esqueleto en LaTeX y cerrar la coherencia global.**

La primera integración puede hacerse antes de terminar la campaña mediante
marcas inequívocas de contenido pendiente en el borrador. No escribir en pasado
ensayos pendientes, rellenar tablas con números ilustrativos ni modificar
conclusiones como si B estuviera validado. Mantener siempre una copia legible
con la evidencia disponible.

Cuando se apruebe esa edición:

- Reorganizar primero 5--6 con movimientos locales y conservar sus etiquetas
  cuando sea posible; actualizar las referencias a datos/hiperparámetros de
  la sección 4 y a las dimensiones reducidas de la sección 3.
- Crear el material suplementario y trasladar allí la evidencia de probe
  antes de retirar sus figuras/tablas del cuerpo. Actualizar generadores
  para que no restauren automáticamente el formato anterior.
- Con los resultados cerrados, revisar resumen, introducción, alcance de la
  sección 2, comparaciones de la sección 3, instancias de la sección 4 y
  conclusiones. No rehacer las demostraciones por añadir otro material.
- Auditar referencias cruzadas, orden de apéndices, procedencia de métricas,
  concordancia figura/texto y afirmaciones de generalidad; compilar y revisar
  visualmente antes de actualizar la copia de lectura.

Paso operativo actual: decidir explícitamente si se conserva el presupuesto
de optimización de la receta v1 o se formaliza una enmienda de parada por
convergencia, antes de reanudar las siete corridas incompletas/no iniciadas.
Ocho ya terminaron bajo v1; seis de esas ocho eligieron la última llamada
LBFGS como mejor checkpoint, por lo que no se debe afirmar convergencia.
Véase [TRAINING_CAMPAIGN_PAUSE.md](../07_material_b/reports/TRAINING_CAMPAIGN_PAUSE.md).
No inspeccionar test ni trayectorias antes de bloquear los 15 checkpoints.
El ejecutor B reanudable ya se comprobó con un entrenamiento
acotado: la corrida continua y la reanudada coincidieron exactamente en el
paso 10, incluida la primera validación programada. Véase
[TRAINING_RUNNER_SMOKE.md](../07_material_b/reports/TRAINING_RUNNER_SMOKE.md).
La campaña FOM, receta exacta y tabla compartida de 32 características
ya están cerradas. La selección usó fit + referencia solamente; 12 entradas
proceden de soportes NNLS y 20 de diversidad QR. Los seis pares parten de
respuestas iguales a precisión numérica. Véase
[TRAINING_PREPARATION.md](../07_material_b/reports/TRAINING_PREPARATION.md).
La ampliación conjunta con cortante ±0.16 falló y no se adopta.
El protocolo y sus coordenadas quedaron congelados: 4200 fit (4096 Sobol de
volumen, 96 de caras y ocho esquinas), 512 validation, 512 test intocado,
diez trayectorias y una auditoría de 64 estados en la malla de 8961 elementos.
Véase [DATA_PROTOCOL.md](../07_material_b/protocol/DATA_PROTOCOL.md). El test y
las trayectorias no seleccionan modelos; la comparación principal de B es
Free, ICNN/ICKAN con características fijas y sus variantes aprendidas, tres
semillas cada una. Regression conserva en A su función de testigo de
integrabilidad. Los smoke tests de escritura/reanudación y de las ramas
sin tangente, malla fina y arranque en frío ya pasaron. Se conserva un fallo
de serialización JSON de tangente ausente, corregido sin inventar etiquetas.
La campaña terminó con cuatro procesos, 103 chunks y 5777 soluciones
contando repeticiones de auditoría; sin estados fallidos ni incrementos rechazados.
Los 64 pares de mallas, los 24 arranques en frío y los controles globales
pasaron: 20/20 comprobaciones del ensamblador. El error máximo de tangente
entre mallas sigue en 0.0889966% contra 0.1%; no ocultar el margen estrecho.
La reconstrucción independiente produce el mismo NPZ byte por byte.
Informe: [CAMPAIGN_REPORT.md](../07_material_b/reports/CAMPAIGN_REPORT.md).
No FE²/POD ni entrenamientos nuevos. La receta exacta de optimización y tasas
de aprendizaje está fijada en un añadido sin alterar el JSON de datos original.
Las comprobaciones de inicialización hicieron cero pasos de optimizador;
81/81 pruebas de implementación pasan. Las trayectorias/predicciones
reservadas sólo se abren después de bloquear los 15 checkpoints; el script
de respuesta está preparado, no se ha utilizado para abrir esas curvas.

---

## Registro histórico: Revision v0.2 — 7 September 2026

Requested scope: identify and read the newly supplied primary references;
rename their PDFs to bibliographic keys; substantially rebuild the manuscript,
especially its introduction and projection-based methodology, using the
nomenclature of Ares De Parga et al. (2026).

## Work sequence

1. Identify each new PDF by its contents, record its version and hash, and
   rename without overwriting another file. Preserve an explicit undo map.
2. Read the new primary texts; distinguish complete reading, targeted reading,
   and any OCR/access limitation. Boehler remains incomplete and is not needed
   for an unsupported theorem attribution.
3. Recover the 2026 latent-closure notation and check the actual periodic
   implementation: HDM, PROM, HPROM; N, N_s, V_tot, V, barred V, q, barred q,
   n, barred n, n_tot, n_tra, and the closure map N. Explain deviations forced
   by the strain-driven RVE lift explicitly rather than asserting an orthogonal
   POD decomposition that the implementation does not use.
4. Rewrite the introduction as a connected argument through homogenization,
   affine and nonlinear PMOR, empirical cubature, constitutive learning, and
   the precise contribution. Restore relevant primary literature, not a target
   citation count. Keep application geometry in the numerical examples.
5. Develop the projection and hyperreduction equations, consistent derivatives,
   direct versus equilibrated closure, and online complexity. Strengthen the
   constitutive comparison and separate properties from sampled evidence.
6. Preserve result/checkpoint files and the former manuscript. Compile the new
   reading copy, check references and layout, and document unresolved tasks.

No new FE2 timing or training campaign is part of this editorial revision.
No second-RVE evidence will be invented. The manuscript remains a working
draft until the full-source and experimental submission gates are satisfied.

## Delivery record

- Nine newly supplied PDFs renamed and hash-verified. Eight article texts
  and both supplied Ciarlet chapters read; the full 2026 nomenclature source
  read as well. Barnett 2023, As'ad 2022, Thakolkaran 2025 and Klein 2022
  subsequently read in full, with further formulation-specific refinements.
  The full-book and full-earlier-corpus limitations are explicit.
- Master source split into maintained section files. Introduction rebuilt,
  PMOR formulation expanded, actual closure/weight training described, and
  mechanical requirements separated from numerical observations.
- Eight figures and eight tables, including two new method/POD figures.
  Bibliography expanded to 42 relevant cited works and ordered by first
  appearance. Boehler excluded; source versions recorded.
- Preceding v0.1 archived; no training, FE2 result, timing or checkpoint
  modified. Validation and reading notes accompany the 27-page PDF.

## Introduction integration — completed 7 September 2026

The user requested that the former Section 7, "Position relative to prior
work and limitations", be part of the introduction. That standalone section
has been removed; its formulation-specific comparison is now Table 1 in
Section 1.3, and its scope and limitations are incorporated into Section 1.5.
Section 1.4 states the contribution after the literature positioning.
Conclusions is now Section 7. This is a structural integration, not merely a
rewrite that leaves the original final positioning section in place.

The introductions of Ares De Parga et al. (2026), As'ad et al. (2022), and the
current MAW–ECM manuscript informed the problem-to-gap progression, the
distinction between fitting and mechanical constraints, and the explanation
of each numerical example's purpose. The feature-order implication supplies
a concrete explanation of the representation issue before the construction
is introduced. The primary/secondary nomenclature is retained.

All 42 bibliography entries are retained and reordered by first citation.
The pre-integration sources are in `archive_pre_intro_integration/`.
Methods, numerical assets and timing evidence are unchanged. Source checks,
compilation and visual inspection of the integrated introduction pass;
these checks do not close the outstanding full-corpus reading or experimental
submission requirements.

## PANN/PROM framing and front matter — completed 7 September 2026

- Restored the exact original PANN manuscript title and its author order.
  Used the official affiliation wording in the supplied MAW–ECM source and
  its CIMNE note for S. Ares de Parga; A. Cornejo has CIMNE and UPC–DECA.
- Rebuilt the introductory progression around FE2 cost, offline/online
  surrogates, non-intrusive PANNs, and intrusive PROMs. Explained why physics
  augmentation is compatible with non-intrusiveness and ANN enrichment with
  intrusive projection. Kept the primary/secondary nomenclature and the
  direct decoder's distinct status.
- Updated the abstract, section heading and method diagram consistently.
  Literature positioning remains inside Section 1. Memory demand motivates
  acceleration, but no unmeasured memory saving is claimed.
- Preserved the method equations, numerical results, timing evidence and all
  42 bibliography entries. Pre-revision sources are archived in
  `archive_pre_pann_prom_revision/`; the original Claude source is untouched.
  The updated reading copy has 28 pages and passes compilation and source
  validation without warnings.

## Argument-led introduction and native figure — 8 September 2026

- Rewrote the abstract to introduce non-intrusive/direct and intrusive/projected
  approximations, retaining the PANN construction as the main contribution.
  Explained ECM hyperreduction, affine HPROM, nonlinear HPROM–ANN and MAW–ECM.
- Reorganized the introductory literature by the questions each construction
  addresses: constitutive state, energy structure, representation dimension,
  projected-operator cost, point placement, and state-dependent weights.
- Added the n-width motivation without asserting a barrier for this RVE;
  distinguished approximate-then-project (DEIM) from project-then-approximate
  hyperreduction. The bibliography now includes 43 works; the new DEIM
  reference's targeted-check/full-reading distinction is in the audit.
- Replaced the literature table with explanatory prose; preserved its source
  and the preceding introduction/master in `archive_pre_prose_revision/`.
- Replaced the Matplotlib hierarchy with native TikZ, using the manuscript's
  text and math fonts. Updated the preview generator and validator accordingly.
- The reading copy has 27 pages, eight figures and seven tables. Numerical
  formulations, data and timing evidence are unchanged.
