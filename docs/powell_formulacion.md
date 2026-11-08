# Formulación de los filtros como problemas de decisión secuencial (Powell)

Sección 3.5 de los lineamientos. Cada filtro se describe con el proceso de modelado de 6 pasos y con los
5 elementos del modelo matemático; se incluyen el cálculo y la interpretación de la ganancia de Kalman, la
selección de Q y R y un ejemplo numérico de varios días con datos reales.

Los números del ejemplo los genera `python main.py` (módulo `reporting/worked_example.py`) en
`artifacts/worked_example_filter1.csv` y `artifacts/worked_example_filter2.csv`. Los días elegidos
(22 al 28 de septiembre de 2022, partición de prueba) son aquellos en que el z-score cruza −2 y se activa una
señal de entrada larga, durante la caída de las utilities por las subidas de tasas.

**Notación.** ln Pᵢ,ₜ es el log-precio ajustado del activo i. El vector de Johansen normalizado sobre DUK es
β = (1, 0.181, −0.614, −0.099) para (DUK, SO, AEP, XEL). El prior de una variable es su estimación antes de
ver el dato del día t (subíndice t|t−1); el posterior, después (t|t). Como en el pizarrón, en las ecuaciones de
Kalman Sₜ denota la varianza de la innovación; cuando se habla del spread se dice explícitamente "spread Sₜ".

---

## 1. Filtro de Kalman 1: pesos dinámicos del portafolio

### Proceso de modelado de 6 pasos

1. **Definir el problema.** Cada día hay que decidir con qué pesos se combina el portafolio para formar el
   spread. Johansen da un vector fijo estimado con 2011–2020, pero la relación entre las utilities cambia con
   el tiempo (con el vector fijo el spread deja de ser estacionario en validación, ADF p = 0.054). El filtro
   actualiza los pesos día a día usando solo la información disponible hasta ese día.
2. **Modelar el estado.** zₜ = (cₜ, β_SO,ₜ, β_AEP,ₜ, β_XEL,ₜ). El peso de DUK queda fijo en +1 por la
   normalización (fija la escala y el signo); cₜ es el nivel de equilibrio del spread.
3. **Modelar la transición.** Caminata aleatoria (A = I): zₜ = zₜ₋₁ + wₜ, con wₜ ~ N(0, Q) y Q = 10⁻⁸·I.
   Los pesos pueden cambiar, pero despacio.
4. **Modelar la observación.** yₜ = ln P_DUK,ₜ = Cₜ zₜ + vₜ, con
   Cₜ = (1, −ln P_SO,ₜ, −ln P_AEP,ₜ, −ln P_XEL,ₜ) y vₜ ~ N(0, R), R = 1.136 × 10⁻³.
   Reordenando, el spread es Sₜ = βₜᵀ ln Pₜ = cₜ + vₜ: el ruido de observación es justamente la desviación
   del equilibrio que se quiere operar.
5. **Definir la regla de decisión.** La decisión es la estimación de los pesos que se usa cada día. La
   política es la actualización de Kalman, una política de función analítica (PFA) con parámetros θ = (Q, R).
   El spread de la señal se calcula con los pesos **prior**, Sₜ = β̂ₜ|ₜ₋₁ᵀ ln Pₜ, para que los pesos con los
   que se mide la desviación de hoy no dependan del precio de hoy.
6. **Mecanismo de aprendizaje.** Ecuaciones de predicción y actualización de Kalman:

   | Fase | Ecuación |
   |---|---|
   | Predict (media) | ẑₜ|ₜ₋₁ = A ẑₜ₋₁|ₜ₋₁ = ẑₜ₋₁|ₜ₋₁ |
   | Predict (incertidumbre) | Pₜ|ₜ₋₁ = A Pₜ₋₁|ₜ₋₁ Aᵀ + Q |
   | Innovación | ỹₜ = yₜ − Cₜ ẑₜ|ₜ₋₁ |
   | Varianza de la innovación | Sₜ = Cₜ Pₜ|ₜ₋₁ Cₜᵀ + R |
   | Ganancia de Kalman | Kₜ = Pₜ|ₜ₋₁ Cₜᵀ Sₜ⁻¹ |
   | Estimador ajustado | ẑₜ|ₜ = ẑₜ|ₜ₋₁ + Kₜ ỹₜ (nuevo estado = estado anterior + ganancia × sorpresa) |
   | Incertidumbre reducida | Pₜ|ₜ = (I − Kₜ Cₜ) Pₜ|ₜ₋₁ |

### Los 5 elementos del modelo

| Elemento | Filtro 1 |
|---|---|
| **Variable de estado** | Creencia sobre los pesos: (ẑₜ₋₁|ₜ₋₁, Pₜ₋₁|ₜ₋₁) |
| **Variable de decisión** xₜ | Pesos estimados ẑₜ|ₜ (y el spread Sₜ calculado con los pesos prior) |
| **Información exógena** Wₜ₊₁ | Log-precios del día siguiente: ln Pₜ₊₁ |
| **Función de transición** S^M | Ecuaciones de predict y update de la tabla anterior |
| **Función objetivo** | Estadísticamente, minimizar el error cuadrático de predicción E[Σ ỹₜ²]. En el sistema completo, Q se evalúa por su efecto en el desempeño de la estrategia (ver la justificación de Q) |

### Ganancia de Kalman: cálculo e interpretación

En el ejemplo, la varianza de la innovación es Sₜ ≈ 1.161 × 10⁻³, de la cual R = 1.136 × 10⁻³ es ruido de
observación y Cₜ Pₜ|ₜ₋₁ Cₜᵀ ≈ 2.5 × 10⁻⁵ es incertidumbre sobre los pesos. Es decir, el filtro atribuye
aproximadamente **2%** de cada sorpresa a un cambio en los pesos y **98%** a una desviación del spread.

El 26 de septiembre de 2022 la innovación fue ỹ = −0.0421: DUK cerró 4.2% por debajo de lo que predecían los
pesos. Con K(β_SO) = −0.00177, el peso de SO cambió solo −0.00177 × (−0.0421) = +0.000075. **Esa es la
interpretación de una ganancia pequeña:** una desviación grande del spread se trata como oportunidad de
arbitraje, no como un cambio en la relación. Las ganancias de los pesos son negativas porque en Cₜ los
log-precios entran con signo menos; su magnitud (≈ 0.0015 a 0.0019) es lo que importa (figura 5).

### Selección de Q y R

- **R = 1.136 × 10⁻³**, la varianza del spread de Johansen en entrenamiento. Es el "price noise": cuánto se
  desvía normalmente el portafolio de su equilibrio.
- **Q = 10⁻⁸ · I**, fijada por diseño. Se intentó elegir Q por máxima verosimilitud, pero la verosimilitud
  siempre prefería Q grandes: con pesos que se mueven rápido el filtro "explica" cada desviación moviendo los
  pesos, las innovaciones se vuelven ruido blanco y la reversión desaparece (vida media del spread de 98 a
  293 días). La causa es que el modelo supone vₜ ruido blanco, mientras que el spread real es autocorrelado
  (revierte en ~15 días). Con Q = 10⁻⁸ los pesos solo siguen cambios estructurales lentos: β_SO pasa de
  0.181 a 0.177 en todo el entrenamiento y a 0.141 al final de la muestra (figura 3). La razón Q/R ≈ 8.8 × 10⁻⁶.
  En el sistema completo Q queda **fuera del grid search**: optimizada junto con los umbrales elige 10⁻⁹,
  que maximiza el Sharpe de entrenamiento pero deja un spread no estacionario fuera de muestra
  (sobreajuste). La comparación 10⁻⁹ / 10⁻⁸ / 10⁻⁷ se reporta ex post en `artifacts/q_sensitivity.csv`.
- **Inicialización:** ẑ₀ = (media del spread de Johansen en entrenamiento, 0.181, −0.614, −0.099) y
  P₀ = 10⁻⁶ · I (confianza alta en el vector de Johansen).
- **Burn-in:** 60 días hábiles (unas cuatro vidas medias), hasta el 30 de enero de 2012; no se opera antes.

### Ejemplo numérico (5 días)

| Fecha | yₜ = ln P_DUK | Cₜ ẑₜ|ₜ₋₁ | Innovación ỹₜ | Sₜ | K(β_SO) | β_SO prior → posterior | β_AEP posterior | β_XEL posterior | Spread Sₜ |
|---|---|---|---|---|---|---|---|---|---|
| 22-sep-2022 | 4.49609 | 4.52109 | −0.02499 | 0.0011617 | −0.00188 | 0.155445 → 0.155492 | −0.628004 | −0.112247 | 1.88743 |
| 23-sep-2022 | 4.48678 | 4.51383 | −0.02705 | 0.0011616 | −0.00185 | 0.155492 → 0.155542 | −0.627959 | −0.112208 | 1.88537 |
| 26-sep-2022 | 4.45872 | 4.50085 | −0.04213 | 0.0011613 | −0.00177 | 0.155542 → 0.155616 | −0.627886 | −0.112147 | 1.87028 |
| 27-sep-2022 | 4.43341 | 4.48200 | −0.04859 | 0.0011610 | −0.00184 | 0.155616 → 0.155706 | −0.627808 | −0.112077 | 1.86379 |
| 28-sep-2022 | 4.44645 | 4.48484 | −0.03839 | 0.0011612 | −0.00191 | 0.155706 → 0.155779 | −0.627748 | −0.112022 | 1.87396 |

Comprobación: en todos los días y componentes, posterior = prior + K × ỹ (error máximo de 10⁻¹⁰, redondeo).
La innovación del filtro 1 coincide con la desviación del spread respecto a su nivel de equilibrio
(1.8703 − 1.9124 = −0.0421 el 26 de septiembre): DUK cayó más de lo que justificaban SO, AEP y XEL.

---

## 2. Filtro de Kalman 2: media y varianza latentes del spread

### Proceso de modelado de 6 pasos

1. **Definir el problema.** Para decidir si el spread está "lejos" de su equilibrio hay que estimar, sin
   look-ahead, su nivel de equilibrio μₜ y su dispersión σₜ, y normalizar la desviación en un z-score.
2. **Modelar el estado.** La media latente μₜ (estimada con Kalman) y la varianza σₜ² (actualizada en
   paralelo con un EWMA de las innovaciones).
3. **Modelar la transición.** μₜ = μₜ₋₁ + wₜ, con wₜ ~ N(0, Q), Q = 10⁻¹⁰.
   σₑ,ₜ² = λ σₑ,ₜ₋₁² + (1 − λ) eₜ², con λ = 0.97.
4. **Modelar la observación.** El spread revierte como un AR(1): Sₜ − μₜ = φ (Sₜ₋₁ − μₜ₋₁) + eₜ. Con
   μₜ ≈ μₜ₋₁, la observación queda lineal en el estado: yₜ = Sₜ − φ Sₜ₋₁ = (1 − φ) μₜ + eₜ, con
   eₜ ~ N(0, R), φ = 0.9563 (vida media de 15.5 días) y R = 6.515 × 10⁻⁵.
5. **Definir la regla de decisión.** zₜ = (Sₜ − μ̂ₜ|ₜ₋₁) / σ̂ₜ|ₜ₋₁, con σ̂ = √(σₑ² / (1 − φ²)), la
   desviación estacionaria del spread. Se usan los estadísticos **prior**: normalizar con la media
   posterior usaría el dato de hoy para juzgarse a sí mismo y sesgaría el z-score hacia cero. Las reglas
   de entrada y salida (sección 3.6) se aplican sobre este zₜ.
6. **Mecanismo de aprendizaje.** Predict: Pₜ|ₜ₋₁ = Pₜ₋₁|ₜ₋₁ + Q. Innovación: eₜ = yₜ − (1 − φ) μ̂ₜ|ₜ₋₁.
   Varianza: Sₜ = (1 − φ)² Pₜ|ₜ₋₁ + R. Ganancia: Kₜ = Pₜ|ₜ₋₁ (1 − φ) / Sₜ.
   Update: μ̂ₜ|ₜ = μ̂ₜ|ₜ₋₁ + Kₜ eₜ y Pₜ|ₜ = (1 − Kₜ (1 − φ)) Pₜ|ₜ₋₁. Después se actualiza σₑ² con el EWMA.

### Los 5 elementos del modelo

| Elemento | Filtro 2 |
|---|---|
| **Variable de estado** | Creencia: (μ̂ₜ₋₁|ₜ₋₁, Pₜ₋₁|ₜ₋₁, σ̂ₑ,ₜ₋₁²) |
| **Variable de decisión** xₜ | El z-score zₜ con estadísticos prior, que alimenta la regla de trading |
| **Información exógena** Wₜ₊₁ | El spread del día siguiente, Sₜ₊₁, que llega del filtro 1 |
| **Función de transición** S^M | Predict y update de μ̂ y P, y EWMA de σₑ² |
| **Función objetivo** | Máxima verosimilitud de las innovaciones en entrenamiento, con la que se elige Q; las innovaciones normalizadas deben ser ruido blanco con varianza 1 |

### Ganancia de Kalman: cálculo e interpretación

El 26 de septiembre de 2022: Pₜ|ₜ₋₁ = 1.1167 × 10⁻⁵ y 1 − φ = 0.0437, así que
Sₜ = 0.0437² × 1.1167 × 10⁻⁵ + 6.515 × 10⁻⁵ = 6.517 × 10⁻⁵ y
**Kₜ = 1.1167 × 10⁻⁵ × 0.0437 / 6.517 × 10⁻⁵ = 0.00749**.

La innovación de ese día fue eₜ = −0.0164, de modo que la media latente bajó apenas
0.00749 × (−0.0164) = −0.00012 (de 1.91624 a 1.91612). **Interpretación:** el filtro trata la caída del spread
como una desviación del equilibrio, no como un cambio del equilibrio; por eso el z-score cruzó −2 y se generó
la señal de entrada larga. La ganancia también decrece con el tiempo (figura 5): con Q tan pequeña, cada día
el filtro confía más en lo que ya sabe sobre μ.

### Selección de Q y R

- **φ y R:** AR(1) ajustado por OLS sobre el spread del filtro 1 en entrenamiento, después del burn-in:
  φ = 0.9563 y R = varianza de los residuales = 6.515 × 10⁻⁵.
- **Q por máxima verosimilitud** en entrenamiento. Aquí sí es válido porque el modelo incluye la reversión
  AR(1) del spread:

  | Q | 10⁻¹⁰ | 10⁻⁹ | 10⁻⁸ | 10⁻⁷ | 10⁻⁶ | 10⁻⁵ |
  |---|---|---|---|---|---|---|
  | Log-verosimilitud | **7483.5** | 7483.5 | 7483.4 | 7482.7 | 7480.1 | 7474.4 |

  La verosimilitud es prácticamente plana para Q ≤ 10⁻⁸ y cae para Q mayores: los datos indican que el nivel
  de equilibrio del spread de Kalman casi no se mueve. Se elige el máximo, Q = 10⁻¹⁰.
- **λ = 0.97** en el EWMA de la varianza: vida media de unas 23 sesiones, comparable a la del spread.
- **Diagnóstico:** fuera de muestra las innovaciones normalizadas tienen varianza 0.99 (prueba) y 1.08
  (validación) y Ljung-Box p = 0.975 y 0.374: ruido blanco, como exige el modelo. En entrenamiento Ljung-Box
  rechaza (p = 0.008) por la volatilidad de marzo de 2020.

### Ejemplo numérico (5 días)

| Fecha | Sₜ | μ̂ prior | σ̂ prior | **zₜ** | Innovación eₜ | Kₜ | μ̂ posterior |
|---|---|---|---|---|---|---|---|
| 22-sep-2022 | 1.88743 | 1.91626 | 0.02155 | −1.34 | 0.00068 | 0.007494 | 1.91627 |
| 23-sep-2022 | 1.88537 | 1.91627 | 0.02123 | −1.46 | −0.00332 | 0.007492 | 1.91624 |
| 26-sep-2022 | 1.87028 | 1.91624 | 0.02100 | **−2.19** | −0.01644 | 0.007490 | 1.91612 |
| 27-sep-2022 | 1.86379 | 1.91612 | 0.02286 | **−2.29** | −0.00849 | 0.007487 | 1.91606 |
| 28-sep-2022 | 1.87396 | 1.91606 | 0.02307 | −1.82 | 0.00788 | 0.007485 | 1.91611 |

Cálculo del z-score del 26 de septiembre: zₜ = (1.87028 − 1.91624) / 0.02100 = −2.19. Con z_entrada = 2.0 se
activa la entrada larga en el spread: comprar DUK y SO, vender AEP y XEL. El 27 de septiembre la desviación
crece y la dispersión estimada σ̂ también, porque el EWMA incorpora la innovación del día anterior. La ganancia
calculada a mano coincide con la del filtro en todos los días (error de 10⁻¹⁰).

---

## 3. Sistema completo como un solo proceso de decisión secuencial

El sistema completo une los dos filtros, el VECM y la lógica de trading en un solo problema: cada día, al cierre,
decidir qué posición tener en el spread y cuántas acciones de cada activo, sabiendo que esa decisión se ejecuta al
cierre del día siguiente y que cada movimiento paga comisiones.

### Proceso de modelado de 6 pasos

1. **Definir el problema.** Maximizar la ganancia neta de costos ajustada por riesgo de un portafolio de
   $1,000,000 que opera la reversión a la media del spread de DUK, SO, AEP y XEL, desplegando 80% del capital y
   pagando 0.125% por pata en cada operación y 0.25% anual sobre los cortos. Toda decisión usa solo información
   disponible al cierre del día en que se toma.
2. **Modelar el estado.** Sₜ = (Rₜ, Iₜ, Bₜ), separado en tres componentes (tabla de abajo): lo que el
   portafolio tiene (físico), lo que se observó (información) y lo que se cree sobre variables no observables
   (creencia).
3. **Modelar la transición.** Tres bloques, en este orden cada día:
   (a) *contabilidad*: se ejecuta la orden de ayer al precio de cierre de hoy, se cobran comisiones, se devenga el
   préstamo y se valúa el portafolio; (b) *filtro 1*: actualiza los pesos β̂ₜ con ln Pₜ; (c) *filtro 2*: calcula zₜ
   con la media y la dispersión prior y después actualiza μ̂ₜ y σ̂ₜ².
4. **Modelar la observación.** Los precios de cierre ajustados Pₜ, de los que se derivan el spread
   Sₜ = β̂ₜ|ₜ₋₁ᵀ ln Pₜ y el z-score zₜ = (Sₜ − μ̂ₜ|ₜ₋₁) / σ̂ₜ|ₜ₋₁.
5. **Definir la regla de decisión.** Política de función analítica (PFA) con parámetros
   θ = (z_entrada, z_salida, z_stop, tenencia máxima):
   - Sin posición: entrar largo si −z_stop ≤ zₜ < −z_entrada; entrar corto si z_entrada < zₜ ≤ z_stop
     (solo si la estrategia está "re-armada": después de un stop o de un cierre por tiempo, |z| debe regresar
     antes a la banda de entrada).
   - Con posición d ∈ {+1, −1}: salir si d·zₜ > −z_salida (el spread regresó a la media o la cruzó); stop si
     d·zₜ < −z_stop; salida por tiempo si los días en posición llegan a la tenencia máxima.
   - Tamaño: Nᵢ,ₜ = d · 0.8 · Vₜ · β̂ᵢ,ₜ / Σⱼ |β̂ⱼ,ₜ|, con rebalanceo diario hacia ese objetivo mientras la
     posición está abierta.
6. **Mecanismo de aprendizaje.** En línea, los dos filtros de Kalman (secciones 1 y 2). Fuera de línea, θ se
   calibra una sola vez con grid search sobre entrenamiento (72 combinaciones, Sharpe neto, mínimo 15
   operaciones) y queda congelado para prueba y validación. Resultado: θ = (2.0, 0.0, 4.0, 63 días).

### Los 5 elementos del modelo

| Elemento | Sistema completo |
|---|---|
| **Variable de estado** Sₜ = (Rₜ, Iₜ, Bₜ) | Ver tabla de componentes abajo |
| **Variable de decisión** xₜ | Dirección de la posición dₜ ∈ {−1, 0, +1} y acciones objetivo de cada activo Nᵢ,ₜ / Pᵢ,ₜ₊₁ (se ejecutan en t + 1) |
| **Información exógena** Wₜ₊₁ | Precios de cierre del día siguiente Pₜ₊₁ (y con ellos el nuevo spread y el nuevo z-score) |
| **Función de transición** S^M | Contabilidad multi-pata (efectivo, acciones, comisiones, préstamo) + ecuaciones de Kalman de ambos filtros + contador de días en posición |
| **Función objetivo** | max_θ Sharpe anualizado de los rendimientos diarios netos de costos en entrenamiento, sujeto a al menos 15 operaciones; equivalente a max E[Σₜ C(Sₜ, xₜ)] ajustado por riesgo, con C = cambio diario del valor neto del portafolio |

| Componente | Contenido |
|---|---|
| **Física** Rₜ | Efectivo, acciones de cada activo (positivas = largo, negativas = corto), dirección de la posición, días en posición y si la estrategia está re-armada |
| **Información** Iₜ | Precios Pₜ, spread Sₜ, z-score zₜ y la orden decidida ayer (pendiente de ejecutar hoy) |
| **Creencia** Bₜ | Filtro 1 (ẑₜ, Pₜ), filtro 2 (μ̂ₜ, Pₜ^μ, σ̂ₜ²), vector de Johansen y parámetros del VECM (α, vida media) |

**Ganancia de Kalman en el sistema completo.** Las ganancias de los filtros determinan qué tan rápido se mueven
el spread y el z-score con los que se decide. La ganancia del filtro 1 es pequeña (≈ 0.002): una desviación grande
se interpreta como oportunidad de arbitraje y no como cambio de pesos; si fuera grande, los pesos absorberían la
desviación y la señal desaparecería. La del filtro 2 converge a valores muy pequeños con Q = 10⁻¹⁰: la media de
equilibrio se mueve muy despacio, así que una desviación del spread se traduce casi completa en z-score. Q y R se
justifican en las secciones 1 y 2; Q del filtro 1 no entra al grid porque elegirla con la estrategia sobreajusta
(`artifacts/q_sensitivity.csv`: con 10⁻⁹ el Sharpe de entrenamiento sube a 0.70 pero cae a 0.20 en prueba).

### Ejemplo numérico: entrada larga del 26 al 29 de septiembre de 2022

Mismos días que el ejemplo de los filtros (partición de prueba). Valores de `artifacts/equity.csv`,
`artifacts/signals.csv` y `artifacts/filter1.csv`.

**26 de septiembre (decisión).** El filtro 2 da μ̂ = 1.91624 y σ̂ = 0.02100 (prior). El spread es 1.87028, así que
z = (1.87028 − 1.91624) / 0.02100 = **−2.19 < −2.0**: el portafolio está por debajo del equilibrio y la política
decide **entrar largo en el spread**. Todavía no hay posición; el valor es V = $1,139,358.42 (todo en efectivo).

**27 de septiembre (ejecución).** La orden se ejecuta al cierre de hoy con los pesos del 26,
β̂ = (1, 0.1556, −0.6279, −0.1121), Σ|β̂| = 1.8956, y con 0.8 · V = $911,486.74:

| Activo | β̂ | Nocional objetivo | Precio de cierre | Acciones |
|---|---|---|---|---|
| DUK | +1.0000 | +$480,830.93 | $84.22 | +5,709.33 |
| SO | +0.1556 | +$74,825.10 | $62.10 | +1,204.88 |
| AEP | −0.6279 | −$301,907.00 | $80.23 | −3,762.82 |
| XEL | −0.1121 | −$53,923.71 | $59.84 | −901.08 |

- Comisión: 0.125% × $911,486.74 = **$1,139.36** (las cuatro patas).
- Préstamo de la primera noche: 0.25% / 252 × $355,830.71 de nocional corto = **$3.53**.
- Efectivo: 1,139,358.42 − 555,656.03 (compras) + 355,830.71 (ventas en corto) − 1,139.36 − 3.53 = **$938,390.21**.
- Valor: 938,390.21 + 555,656.03 − 355,830.71 = **$1,138,215.53**, que es el valor anterior menos los costos
  ($1,142.89). La identidad valor = efectivo + patas es la que verifica la prueba 3.
- z = −2.29: sigue lejos del equilibrio, se mantiene la posición.

**28 de septiembre.** El spread sube 0.0102 (de 1.8638 a 1.8740) y la posición larga gana:

| Pata | Acciones × cambio de precio |
|---|---|
| DUK | +5,709.33 × (+$1.105) = +$6,308.66 |
| SO | +1,204.88 × (+$0.693) = +$834.64 |
| AEP | −3,762.82 × (+$0.605) = −$2,275.83 |
| XEL | −901.08 × (+$0.473) = −$425.90 |
| **Total** | **+$4,441.57** |

Menos el rebalanceo diario hacia los nuevos pesos (comisión $9.01) y el préstamo ($3.54), el valor sube
**$4,429.02**, a $1,142,644.56. z = −1.82: como la salida es en z > 0, la posición sigue abierta.

**29 de septiembre.** Las utilities caen otra vez y el spread baja a 1.8583 (z = −2.49): el valor baja a
$1,134,824.12. La posición se mantiene porque z no ha llegado ni a la salida (0) ni al stop (−4).

**Cierre.** El spread regresa a la media y la posición se cierra el 14 de diciembre de 2022, 55 días después:
P&L bruto de $7,532.49, comisiones de $3,056.71 (entrada, salida y rebalanceos), préstamo de $190.95 y
**P&L neto de $4,284.84**. El ejemplo muestra el mecanismo completo y también la principal debilidad de la
estrategia: las comisiones se llevaron 41% de la ganancia bruta de esta operación (26% en toda la muestra).
