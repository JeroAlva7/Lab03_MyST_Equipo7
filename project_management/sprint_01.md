# Sprint 01 — 3 al 8 de noviembre de 2026

## Objetivo

Construir de punta a punta la estrategia de arbitraje estadístico sobre un grupo de 4 activos cointegrados,
desde los datos hasta el dashboard. Al cierre del sprint el repositorio debe correr con `python main.py`,
pasar las pruebas con `pytest` y contener el reporte en `docs/reporte.pdf`.

## Tareas

| # | Tarea | Responsable | Estado |
|---|---|---|---|
| 1 | Repositorio, entorno, estructura, charter y plan de trabajo | Jerónimo | Hecho |
| 2 | Universo de activos y datos (10-15 años), validación y partición | Jerónimo | Hecho |
| 3 | Orden de integración, correlación, Engle-Granger y Johansen | Jerónimo | Hecho |
| 4 | Filtro de Kalman 1 (pesos dinámicos) | Jerónimo | Hecho |
| 5 | Filtro de Kalman 2 (media y varianza del spread) y Johansen móvil | Jerónimo | Hecho |
| 6 | Pruebas de causalidad, normalización e innovaciones | Jerónimo | Hecho |
| 7 | VECM, reglas de señal y sizing | Raúl | Hecho |
| 8 | Backtest multi-pata con costos y bitácora | Raúl | Hecho |
| 9 | Prueba de contabilidad multi-pata | Raúl | Hecho |
| 10 | Grid search, walk-forward y rebalanceo diario contra banda | Raúl | Hecho |
| 11 | Métricas, integración en main.py y artifacts | Raúl | Hecho |
| 12 | Dashboard (vista ejecutiva y técnica) | Raúl | Hecho |
| 13 | Figuras obligatorias (1, 3-6 Jerónimo; 2, 7, 8 Raúl) | Ambos | Hecho: 1 a 8, más A1 (robustez del grid) |
| 14 | Reporte ejecutivo, presentación y post-mortem | Ambos | En curso |

El detalle de commits por integrante está en `plan_de_trabajo.md`.

## Bloqueos y desviaciones

- **4 nov — Cambio de universo.** El universo inicial (petroleras: XOM, CVX, COP, EOG) no superó el embudo:
  ningún grupo rechazó la hipótesis de no cointegración (Engle-Granger p > 0.44, traza de Johansen por debajo
  de su valor crítico) y las vidas medias fueron de 77 a 205 días. Se evaluaron con el mismo embudo 5 universos
  más de un solo sector (66 grupos en total, `artifacts/universe_search.csv`). Solo las utilities eléctricas
  (DUK, SO, AEP, XEL) aprobaron: 5 de 11 grupos. Se eligió el grupo de 4 activos (vida media de 24 días) y se
  compara contra el mejor par, DUK-AEP. Impacto: medio día de retraso en el hito H3.
- **4 nov — Calibración del filtro 1.** Elegir Q por máxima verosimilitud hacía que los pesos absorbieran las
  desviaciones del spread (vida media de 98 a 293 días). Se fijó Q = 1e-8 por diseño y Q del filtro 2 sí se
  eligió por máxima verosimilitud, porque ese modelo incluye la reversión AR(1) del spread.
- **4 nov — Hallazgo de estabilidad.** Con pesos fijos de Johansen el spread deja de ser estacionario en
  validación (ADF p = 0.054); con pesos de Kalman sigue siéndolo (p = 0.002).
- **4 nov — Entorno de Python.** VS Code activaba un entorno virtual de otro proyecto; se corrigió seleccionando
  el intérprete de `.venv` del repositorio.
- **4 nov — Retraso en la segunda mitad.** El 4 de noviembre se terminaron R1 a R3; R4 (sizing), planeado para ese
  día, pasó al jueves 5. Se recuperó el mismo jueves: R4 a R13 quedaron terminados un día antes de lo planeado.
- **4 nov — Señal sin duplicar el filtro 2.** La primera versión del VECM recalculaba el z-score, que ya es la
  columna `z` del filtro 2 (diferencia máxima de 7·10⁻⁸). Se eliminó la función duplicada: las señales leen
  directamente la salida del filtro 2 y el VECM se usa para reportar las velocidades de ajuste α.
- **5 nov — Tenencia máxima con la vida media equivocada.** Se calculaba con la vida media del spread estático de
  Johansen (24 días), pero las señales operan el spread de Kalman, cuya vida media es de 15.5 días (AR(1) del
  filtro 2). Se corrigió; con el múltiplo inicial de 3, la tenencia máxima bajó de 72 a 47 días.
- **5 nov — Regla de salida incompleta.** "Salir si |z| < z_salida" dejaba posiciones abiertas cuando el z-score
  cruzaba la media de un día para otro sin caer en la banda (se mueve hasta 2.16 en un día). Pasó 3 días con los
  umbrales iniciales y 17 con z_entrada = 1.0; con z_salida = 0 la salida era imposible. Se corrigió: un largo
  sale si z > −z_salida y un corto si z < +z_salida. También se impidió entrar si |z| ya rebasó el stop.
- **5 nov — Exposición fuera del 80%.** Sin rebalanceo, la exposición bruta derivaba entre 66% y 96.5% del
  capital mientras la posición estaba abierta. El rebalanceo diario la mantiene en 80% a un costo de $10,670 en
  comisiones; una banda de 5% ahorraría 85% de ese costo con exposición entre 73% y 87% (`rebalancing.csv`).
- **5 nov — Pocas operaciones fuera de muestra.** Con los umbrales elegidos hay 5 operaciones en prueba y 5 en
  validación. El grid search mostró que bajar z_entrada aumenta las operaciones pero reduce el Sharpe en
  entrenamiento (0.24 con z = 1.0 contra 0.41 con z = 2.0) porque cada operación paga ~$2,000 de comisiones. Se
  aceptó la limitación y se exige un mínimo de 15 operaciones en entrenamiento para que una combinación compita.
- **5 nov — Diferencias de versión de Python.** Correr `main.py` con Python 3.13 en lugar de 3.12 cambia los
  últimos decimales de algunos artifacts; esos cambios se descartan antes de cada commit para no generar
  conflictos.
- **6 nov — Q del filtro 1 fuera del grid.** Se probó agregar Q del filtro 1 al grid search de umbrales. El grid
  elige Q = 1e-9 (Sharpe de entrenamiento 0.70 contra 0.41 con 1e-8), pero con esa Q el spread de Kalman deja de
  ser estacionario fuera de muestra y el Sharpe en prueba y validación baja (0.20 y 0.27 contra 0.32 y 0.30).
  Se mantuvo Q = 1e-8 fija por diseño y la comparación se reporta como sensibilidad ex post
  (`artifacts/q_sensitivity.csv`). Se agregó la figura A1 con el Sharpe de entrenamiento de todo el grid para
  mostrar que la combinación elegida está en una zona estable.
- **8 nov — Dashboard con pandas 3.** Al correrlo en el entorno del equipo falló porque en pandas 3 los arreglos de
  `to_numpy()` son de solo lectura; se corrigió pidiendo una copia (`to_numpy(copy=True)`). El dashboard y las
  figuras 2, 7 y 8, planeados para el sábado 7, quedaron el domingo 8.
