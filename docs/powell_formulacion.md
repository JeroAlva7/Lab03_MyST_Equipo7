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
  En el sistema completo, Q es un parámetro θ de la política que el grid search puede mover (10⁻⁹ a 10⁻⁷).
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

*Pendiente: se completa con la lógica de trading y la contabilidad del backtest (sección 3.6), incluyendo
su ejemplo numérico de varios días.* Estructura acordada del estado:

| Componente | Contenido |
|---|---|
| **Física** Rₜ | Efectivo, acciones de cada activo, dirección de la posición y días en posición |
| **Información** Iₜ | Precios Pₜ, spread Sₜ y z-score zₜ |
| **Creencia** Bₜ | Filtro 1 (ẑₜ, Pₜ), filtro 2 (μ̂ₜ, Pₜ^μ, σ̂ₜ²), vector de Johansen y parámetros del VECM |
