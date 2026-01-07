# Proyecto RNFE

## Descripción

Este proyecto implementa una arquitectura de simulación y experimentación para escenarios de mundos artificiales, aprendizaje y razonamiento, con módulos para agentes, control, métricas, experimentos y más. El código está organizado en submódulos bajo `src/rnfe` y cuenta con scripts, tests y documentación adicional.

## Estructura del proyecto

- `src/rnfe/` — Código fuente principal, organizado en:
  - `agents/` — Agentes y estrategias.
  - `core/` — Núcleo: control, geometría, métricas, etc.
  - `fmse/` — Modelos de evolución, semántica y mundo.
  - `infra/` — Infraestructura: logging, almacenamiento, utilidades.
  - `pmv/` — Módulos de experimentos y fases.
  - `reasoning/` — Modos y pipelines de razonamiento.
- `scripts/` — Scripts de entrenamiento, pruebas y utilidades.
- `tests/` — Pruebas unitarias y de integración.
- `configs/` — Configuraciones y presets de experimentos.
- `data/` — Datos y logs generados.
- `docs/` — Documentación, diagramas y especificaciones.
- `experiments/` — Resultados y configuraciones de experimentos.

## Instalación

1. Clona el repositorio y entra al directorio del proyecto.
2. Crea y activa un entorno virtual (recomendado):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   O usa tu entorno gestionado por pyenv, por ejemplo:
   ```bash
   pyenv activate rnfe-lab
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución de tests

Para ejecutar los tests, asegúrate de tener el entorno virtual activo y ejecuta:
```bash
PYTHONPATH=src pytest --maxfail=5 --disable-warnings -q
```

## Notas
- Si tienes problemas de importación, revisa que `PYTHONPATH` incluya la carpeta `src`.
- El archivo `requirements.txt` se puede regenerar con `pip freeze > requirements.txt` tras instalar nuevos paquetes.

## Contacto
Para dudas o contribuciones, contacta al responsable del repositorio.
