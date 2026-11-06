"""Reglas de operación sobre el z-score del filtro 2 (sección 3.6).

Dirección de la posición en el spread:
    +1  spread largo: z_t < -z_entrada (el portafolio está por debajo del equilibrio)
    -1  spread corto: z_t > +z_entrada (el portafolio está por encima del equilibrio)
     0  sin posición

Una posición abierta se cierra por la primera de estas razones:
    "salida"   el spread regresó al equilibrio: z_t > -z_salida en un largo, z_t < +z_salida en
               un corto. Incluye |z_t| < z_salida (la regla del enunciado) y también el caso en
               que z cruza la media de un día para otro sin caer dentro de la banda: el z-score
               llega a moverse más de 2 unidades en un día, más que el ancho de la banda (2 x z_salida).
    "stop"     el spread se alejó todavía más: z_t < -z_stop en un largo, z_t > +z_stop en un corto
    "tiempo"   se cumplió la tenencia máxima, ligada a la vida media del spread
    "burn-in"  el filtro aún no es confiable o no hay z-score

El stop y la tenencia máxima existen porque esta estrategia pierde dinero con una posición
que deja de revertir, no con muchas pérdidas pequeñas. Después de un stop o de un cierre
por tiempo no se vuelve a entrar hasta que |z| regrese a la banda de entrada; si no, se
reabriría al día siguiente la misma apuesta que se acaba de cortar. Tampoco se entra si el
z-score ya está más allá del stop: la posición se cerraría por stop al día siguiente.

La señal de t usa solo z_t, conocido al cierre de t; el backtest la ejecuta a partir de t+1.
"""
import numpy as np
import pandas as pd


def max_holding_days(half_life: float, multiple: float) -> int:
    """Tenencia máxima = múltiplo de la vida media (redondeado hacia arriba)."""
    return int(np.ceil(multiple * half_life))


def generate_signals(z: pd.Series, tradable: pd.Series, z_entry: float, z_exit: float,
                     z_stop: float, max_holding: int) -> pd.DataFrame:
    """Recorre el z-score día por día y decide la dirección de la posición al cierre.

    Args:
        z: z-score del filtro 2 (columna "z" de filter2.csv).
        tradable: True después del burn-in del filtro.
        z_entry: umbral de entrada (z_entry > z_exit).
        z_exit: umbral de salida.
        z_stop: umbral de stop (z_stop > z_entry).
        max_holding: días máximos con la posición abierta.

    Returns:
        Por fecha: z, direction (-1, 0, +1), days_held (días con la posición abierta al
        cierre) y event ("entrada larga", "entrada corta", "salida", "stop", "tiempo",
        "burn-in" o vacío).
    """
    if not 0 <= z_exit < z_entry < z_stop:
        raise ValueError("Se requiere 0 <= z_salida < z_entrada < z_stop")

    direction, days_held, rearmed = 0, 0, True
    rows = []
    for z_t, ok in zip(z.to_numpy(), tradable.to_numpy()):
        event = ""
        if not ok or np.isnan(z_t):
            if direction != 0:
                event = "burn-in"
            direction, days_held = 0, 0
        elif direction != 0:
            days_held += 1
            if direction * z_t > -z_exit:            # regresó a la banda o ya la cruzó
                event = "salida"
            elif direction * z_t < -z_stop:
                event, rearmed = "stop", False
            elif days_held >= max_holding:
                event, rearmed = "tiempo", False
            if event:
                direction, days_held = 0, 0
        else:
            rearmed = rearmed or abs(z_t) < z_entry
            in_band = z_entry < abs(z_t) <= z_stop    # nunca se entra ya más allá del stop
            if rearmed and in_band and z_t < 0:
                direction, event = 1, "entrada larga"
            elif rearmed and in_band and z_t > 0:
                direction, event = -1, "entrada corta"
        rows.append((z_t, direction, days_held, event))

    return pd.DataFrame(rows, index=z.index, columns=["z", "direction", "days_held", "event"])
