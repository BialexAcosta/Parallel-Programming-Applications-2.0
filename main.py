#!/usr/bin/env python3
"""Runner principal para orquestar todos los scripts del repositorio.

Uso:
  python main.py [--run-mpi] [--include-colab] [--dry-run]

- Por seguridad, los scripts que contienen "mpi" en su nombre se omiten
  a menos que se pase `--run-mpi` (requieren mpirun/mpiexec).
- `--include-colab` permitirá ejecutar `setup_colab.py` si existe.
- `--dry-run` solo mostrará el orden sin ejecutar nada.

El runner ejecuta cada script con el intérprete de Python activo
(`sys.executable`) usando como cwd el directorio del script.
No modifica archivos del repositorio.
"""

from __future__ import annotations
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("main-runner")

TOP_ORDER = ["download_data.py", "download_firms.py"]
COLAB_NAME = "setup_colab.py"
EXCLUDE_NAMES = {"main.py"}

# Root of the repository (this file lives in repo root)
REPO_ROOT = Path(__file__).resolve().parent


def discover_scripts(include_colab: bool, run_mpi: bool) -> List[Path]:
    root = REPO_ROOT
    run_list: List[Path] = []

    # Add top-level ordered scripts if present
    for name in TOP_ORDER:
        p = root / name
        if p.exists() and p.suffix == ".py":
            run_list.append(p)

    # Optionally include setup_colab
    if include_colab:
        p = root / COLAB_NAME
        if p.exists() and p.suffix == ".py":
            run_list.append(p)

    # Walk exercise folders (sorted) and collect .py files
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("exercise")):
        for script in sorted(folder.glob("*.py")):
            if script.name in EXCLUDE_NAMES:
                continue
            if script.suffix != ".py":
                continue
            # Skip compiled / cache / private files
            if script.name.startswith("_"):
                continue
            # Skip MPI scripts unless allowed
            if ("mpi" in script.name.lower()) and (not run_mpi):
                logger.debug("Skipping MPI script (use --run-mpi to enable): %s", script)
                continue
            run_list.append(script)

    # Also include top-level scripts not in TOP_ORDER (except excluded)
    for script in sorted(p for p in root.glob("*.py") if p.name not in TOP_ORDER and p.name not in EXCLUDE_NAMES and p.name != COLAB_NAME):
        # Already added earlier
        if script in run_list:
            continue
        if ("mpi" in script.name.lower()) and (not run_mpi):
            continue
        run_list.append(script)

    return run_list


def run_script(path: Path, dry_run: bool) -> int:
    logger.info("-> %s", path)
    if dry_run:
        return 0
    try:
        # Run each script with repository root as cwd so scripts that
        # use repo-relative paths (e.g. "exercise_3/firms_data.csv")
        # resolve correctly instead of duplicating folder names.
        completed = subprocess.run([sys.executable, str(path)], cwd=str(REPO_ROOT))
        return completed.returncode
    except Exception as e:
        logger.error("Error al ejecutar %s: %s", path, e)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Orquesta ejecución de scripts del repositorio")
    parser.add_argument("--run-mpi", action="store_true", help="Permitir ejecutar scripts MPI (requieren mpirun/mpiexec)")
    parser.add_argument("--include-colab", action="store_true", help="Incluir setup_colab.py si existe")
    parser.add_argument("--dry-run", action="store_true", help="Mostrar qué se ejecutaría sin ejecutar")
    args = parser.parse_args(argv)

    logger.info("Descubriendo scripts (run_mpi=%s, include_colab=%s)", args.run_mpi, args.include_colab)
    scripts = discover_scripts(args.include_colab, args.run_mpi)

    if not scripts:
        logger.warning("No se encontraron scripts para ejecutar.")
        return 0

    logger.info("Se encontraron %d scripts. Orden de ejecución:", len(scripts))
    for s in scripts:
        logger.info("  - %s", s)

    failures = 0
    for s in scripts:
        rc = run_script(s, args.dry_run)
        if rc != 0:
            logger.warning("Script %s terminó con código %d", s, rc)
            failures += 1

    if args.dry_run:
        logger.info("Dry-run completado. No se ejecutaron scripts.")
        return 0

    if failures:
        logger.warning("Ejecución completada con %d fallo(s)", failures)
        return 2

    logger.info("Ejecución completada correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
