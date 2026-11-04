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
| 4 | Filtro de Kalman 1 (pesos dinámicos) | Jerónimo | Pendiente |
| 5 | Filtro de Kalman 2 (media y varianza del spread) y Johansen móvil | Jerónimo | Pendiente |
| 6 | Pruebas de causalidad, normalización e innovaciones | Jerónimo | Pendiente |
| 7 | VECM, reglas de señal y sizing | Raúl | Pendiente |
| 8 | Backtest multi-pata con costos y bitácora | Raúl | Pendiente |
| 9 | Prueba de contabilidad multi-pata | Raúl | Pendiente |
| 10 | Grid search, walk-forward y rebalanceo diario contra banda | Raúl | Pendiente |
| 11 | Métricas, integración en main.py y artifacts | Raúl | Pendiente |
| 12 | Dashboard (vista ejecutiva y técnica) | Raúl | Pendiente |
| 13 | Figuras obligatorias (1, 3-6 Jerónimo; 2, 7, 8 Raúl) | Ambos | Pendiente |
| 14 | Reporte ejecutivo, presentación y post-mortem | Ambos | Pendiente |

El detalle de commits por integrante está en `plan_de_trabajo.md`.

## Bloqueos y desviaciones

- **4 nov — Cambio de universo.** El universo inicial (petroleras: XOM, CVX, COP, EOG) no superó el embudo:
  ningún grupo rechazó la hipótesis de no cointegración (Engle-Granger p > 0.44, traza de Johansen por debajo
  de su valor crítico) y las vidas medias fueron de 77 a 205 días. Se evaluaron con el mismo embudo 5 universos
  más de un solo sector (66 grupos en total, `artifacts/universe_search.csv`). Solo las utilities eléctricas
  (DUK, SO, AEP, XEL) aprobaron: 5 de 11 grupos. Se eligió el grupo de 4 activos (vida media de 24 días) y se
  compara contra el mejor par, DUK-AEP. Impacto: medio día de retraso en el hito H3.
- **4 nov — Entorno de Python.** VS Code activaba un entorno virtual de otro proyecto; se corrigió seleccionando
  el intérprete de `.venv` del repositorio.
