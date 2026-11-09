# Post-mortem — Lab03_MyST_Equipo7

Cada integrante redacta su mitad: Jerónimo, del dato a la señal; Raúl, de la señal al dinero.

## Hitos cumplidos a tiempo

**De la señal al dinero (Raúl).** H5 (VECM, señales y backtest completo con costos) estaba planeado para el
viernes 6 y quedó el jueves 5, junto con el grid search, el walk-forward, el rebalanceo contra banda, las métricas
y la integración en `main.py` (R4 a R13 en un solo día). La prueba 3 (contabilidad multi-pata) quedó el mismo
jueves.

**Del dato a la señal (Jerónimo).** H1 (martes 3) y H2 (miércoles 4) se cumplieron en la fecha planeada. H4
(filtros 1 y 2, Johansen móvil y pruebas 1, 2 y 4) quedó el miércoles 4, un día antes de lo planeado, y la
formulación de Powell con el ejemplo numérico y los supuestos del README quedaron el jueves 5, uno y dos días antes.

## Hitos retrasados y causas

**De la señal al dinero (Raúl).**
- **R4 (sizing):** planeado para el miércoles 4, se hizo el jueves 5. El miércoles se terminaron R1 a R3 en la
  noche, después de incorporarse al repositorio y configurar el entorno.
- **H6 (dashboard y figuras 2, 7 y 8):** planeado para el sábado 7, se terminó el domingo 8 por la mañana. Al
  probarlo en el entorno real falló por un cambio de pandas 3 (los arreglos que devuelve `to_numpy()` son de solo
  lectura) que no apareció al probarlo con pandas 2.
- **Secciones 7 a 10 del reporte:** dependían de los resultados finales del backtest, por lo que se redactaron al
  final, el domingo 8.

**Del dato a la señal (Jerónimo).**
- **H3 (grupo seleccionado):** medio día de retraso. Las petroleras no cointegraron (Engle-Granger p > 0.44, vidas
  medias de 77 a 205 días) y hubo que evaluar cinco sectores más con el mismo embudo antes de elegir utilities.
- **Figura 6:** se hizo al final porque necesitaba los umbrales del grid search; fue la única dependencia de la
  primera mitad respecto a la segunda.

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

**Del dato a la señal (Jerónimo).**
- **Máxima verosimilitud con un modelo mal especificado.** Para Q del filtro 1, la verosimilitud siempre
  favorecía Q grandes: los pesos absorbían las desviaciones y la vida media del spread subía a 98-293 días, porque
  el filtro supone ruido blanco y el spread es autocorrelado. Se fijó Q por diseño. Optimizarla junto con los
  umbrales tampoco sirvió: elegía Q = 10⁻⁹, con el doble de Sharpe en entrenamiento y spread no estacionario fuera
  de muestra. El Sharpe por sí solo no basta para calibrar un filtro.
- **Reproducibilidad.** Dos descargas de Yahoo Finance diferían en el sexto decimal; se resolvió guardando los
  precios en `data/prices.csv`. El entorno de VS Code activaba el entorno virtual de otro proyecto.

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

**Del dato a la señal (Jerónimo).**
- **Datos de un proveedor profesional** en lugar de Yahoo Finance, cuyos precios ajustados cambian entre descargas.
- **Recalibración y monitoreo de los filtros:** reestimar el vector y R con ventana móvil y alertar cuando las
  innovaciones normalizadas salgan de su banda o los pesos se muevan más de lo que permite Q.
- **Un solo modelo de estado con el spread AR(1) dentro,** para estimar Q del filtro 1 por máxima verosimilitud
  sin que los pesos absorban la señal.

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

**Del dato a la señal (Jerónimo).**
- **R1, pérdida de cointegración: se materializó con pesos fijos.** El spread de Johansen deja de ser estacionario
  en validación (p = 0.054); el de Kalman no (p = 0.002). El filtro 1 fue la mitigación que funcionó, aunque el
  Johansen móvil solo detecta la relación en 19% de las ventanas de tres años.
- **R2, divergencia del filtro: solo en desarrollo.** La Q por máxima verosimilitud se detectó antes del backtest;
  fuera de muestra las innovaciones tienen varianza 0.99 y 1.08 y pasan Ljung-Box (p = 0.975 y 0.374).
- **R3, calidad de datos: no se materializó.** 3,771 días sin faltantes ni duplicados.
- **R5, pruebas múltiples: declarado.** De 66 grupos (3.3 falsos positivos esperados) los 5 aprobados son de un
  solo sector, pero con Bonferroni dentro de utilities (0.05/11) solo DUK-AEP pasa Engle-Granger, y sobre los 66
  grupos (0.05/66) no pasaría ninguno de utilities. Además, las dos operaciones del COVID aportan 61% de la
  ganancia de entrenamiento: la calibración depende de un solo evento.

## Conclusión conjunta

Todos los hitos se cumplieron con a lo sumo un día de retraso, porque el contrato de `artifacts/` permitió trabajar
las dos mitades en paralelo. Los errores importantes (Q del filtro 1, regla de salida, vida media de la tenencia
máxima) no los encontraron las pruebas sino la revisión de cada número contra su interpretación. La relación
resultó real pero débil e intermitente: antes de operar haría falta detectar cuándo se pierde la cointegración.
