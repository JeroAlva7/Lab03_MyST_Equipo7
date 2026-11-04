"""Configuración de pytest: agrega la raíz del repositorio al path de importación."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
