# Post-mortem — Lab03_MyST_Equipo7

Cada integrante redacta la parte de su mitad del pipeline; las secciones se completan en conjunto.

## Hitos cumplidos a tiempo

**De la señal al dinero (Raúl).** H5 (VECM, señales y backtest completo con costos) estaba planeado para el
viernes 6 y quedó el jueves 5, junto con el grid search, el walk-forward, el rebalanceo contra banda, las métricas
y la integración en `main.py` (R4 a R13 en un solo día). La prueba 3 (contabilidad multi-pata) quedó el mismo
jueves.

## Hitos retrasados y causas

**De la señal al dinero (Raúl).**
- **R4 (sizing):** planeado para el miércoles 4, se hizo el jueves 5. El miércoles se terminaron R1 a R3 en la
  noche, después de incorporarse al repositorio y configurar el entorno.
- **H6 (dashboard y figuras 2, 7 y 8):** planeado para el sábado 7, se terminó el domingo 8 por la mañana. Al
  probarlo en el entorno real falló por un cambio de pandas 3 (los arreglos que devuelve `to_numpy()` son de solo
  lectura) que no apareció al probarlo con pandas 2.
- **Secciones 7 a 10 del reporte:** dependían de los resultados finales del backtest, por lo que se redactaron al
  final, el domingo 8.

## Cuellos de botella técnicos y organizacionales

**De la señal al dinero (Raúl).**
- **Errores de lógica encontrados en revisión, no en las pruebas.** Tres errores pasaron las pruebas iniciales y
  se encontraron al revisar el código contra los datos: (1) la tenencia máxima usaba la vida media del spread de
  Johansen (24 días) y no la del spread de Kalman que se opera (15.5 días); (2) la regla de salida |z| < z_salida
  dejaba posiciones abiertas cuando el z-score cruzaba la media de un día a otro sin caer en la banda (se mueve
  hasta 2.16 en un día), y con z_salida = 0 la salida era imposible; (3) sin rebalanceo la exposición derivaba
  entre 66% y 96.5% del capital. Aprendizaje: además de pruebas de contabilidad, hacen falta pruebas de invarianza
  (causalidad, alterar precios futuros) y revisar casos extremos de cada regla.
- **Código duplicado.** La primera versión del VECM recalculaba el z-score que ya entrega el filtro 2; la cadena
  señales → backtest → bitácora se repetía en el grid search y en la prueba de contabilidad. Se resolvió con una
  sola función (`run_strategy`) y un solo lugar para los parámetros de ejecución (`execution_settings`).
- **Entornos distintos.** Correr el pipeline con Python 3.13 en lugar de 3.12 cambiaba los últimos decimales de
  los artifacts del otro integrante; esos cambios se descartaban antes de cada commit para evitar conflictos.
- **Organizacional.** El contrato de archivos en `artifacts/` (definido en `plan_de_trabajo.md`) permitió
  trabajar la segunda mitad sin esperar a la primera. Los archivos compartidos (`main.py`, `README.md`,
  `sprint_01.md`) obligaron a coordinar quién los editaba y a hacer `git pull` antes de cada cambio.

## Qué haríamos distinto en producción

**De la señal al dinero (Raúl).**
- **Costos realistas.** El backtest ejecuta al precio de cierre, sin diferencial compra-venta, sin impacto de
  mercado y suponiendo que siempre hay acciones para vender en corto. Las comisiones ya consumen 26% de la ganancia
  bruta; con esos costos adicionales el margen sería menor. Habría que modelarlos antes de operar.
- **Rebalanceo con banda.** El rebalanceo diario cuesta $10,670; una banda de 5% ahorra 85% de ese costo con una
  exposición entre 73% y 87%. En producción el límite de 80% se definiría como un rango.
- **Salir cuando se pierde la cointegración, no solo por tiempo.** Las 8 pérdidas vinieron de posiciones que
  dejaron de revertir. Un filtro que cierre o bloquee posiciones cuando el Johansen móvil o el ADF del spread
  dejan de detectar la relación cortaría esas pérdidas antes de los 63 días.
- **Más operaciones independientes.** Con 5 operaciones en prueba y 5 en validación los resultados no son
  estadísticamente concluyentes. Operar varios grupos (Nivel B) daría más operaciones y diversificación.
- **Acciones enteras y monitoreo.** Redondeo a acciones enteras, datos en tiempo real, alertas cuando la
  innovación normalizada del filtro se sale de su rango y conciliación diaria de efectivo y posiciones.

## Materialización de los riesgos del registro

**De la señal al dinero (Raúl).**
- **R1, pérdida de cointegración fuera de muestra: se materializó y la mitigación funcionó.** Las tres pérdidas
  fuera de muestra (−$14,598, −$21,939 y −$15,282) fueron posiciones que dejaron de revertir y se cerraron por la
  tenencia máxima. Sin ese límite habrían seguido abiertas; la peor pérdida de una operación fue de −2.5% del
  portafolio (el único stop, en enero de 2018).
- **R4, sesgo de look-ahead: no se materializó.** La señal de t se ejecuta en t + 1 con los pesos de t, el
  z-score usa estadísticos prior y los umbrales se eligen solo con entrenamiento. Se verificó que cortar los datos
  en cualquier fecha, o alterar los precios posteriores, no cambia el valor del portafolio antes de esa fecha.
- **R5, pruebas múltiples: mitigado en la calibración.** El grid search exige al menos 15 operaciones en
  entrenamiento (la combinación de mayor Sharpe, con 9 operaciones, quedó fuera) y Q del filtro 1 no entra al
  grid. El Sharpe baja de 0.41 a 0.32 y 0.30 fuera de muestra, sin colapsar.
- **R6, plazo de una semana y dependencia entre integrantes: se materializó parcialmente.** Hubo un día de
  retraso en R4 y en el dashboard; el contrato de artifacts evitó que una mitad bloqueara a la otra.
