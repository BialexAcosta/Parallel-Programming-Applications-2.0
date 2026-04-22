# HPC Unit 3 — Final Assignment

## Objetivo del Proyecto

Este repositorio contiene la tarea final para el curso de High Performance Computing (HPC) Unidad 3. El objetivo es implementar y comparar soluciones **seriales vs paralelas** para cuatro problemas distintos de cómputo científico, ciencia de datos e inteligencia artificial, utilizando `multiprocessing` y `mpi4py` en Python. 

El enfoque principal es demostrar el impacto de la paralelización midiendo tiempos de ejecución, calculando el *speedup* y la eficiencia, y analizando los cuellos de botella (como el *overhead* de creación de procesos o el costo de comunicación).

## Estructura del Repositorio

El proyecto está organizado en las siguientes carpetas y archivos:

```text
hpc-unit3/
├── README.md               ← Este archivo de documentación
├── requirements.txt        ← Dependencias de Python necesarias
├── setup_colab.py          ← Script de configuración si se usa Google Colab
├── download_data.py        ← Script para descargar imágenes HeLa (Ej. 2)
├── download_firms.py       ← Script para descargar datos de la NASA (Ej. 3)
│
├── exercise_1/             ← Multiplicación de Matrices
│   ├── serial_matmul.py
│   ├── parallel_row.py
│   ├── parallel_col.py
│   ├── parallel_block.py
│   ├── mpi_matmul.py
│   ├── strassen.py
│   └── sparse_matmul.py
│
├── exercise_2/             ← Procesamiento de Imágenes Celulares
│   ├── serial_pipeline.py
│   ├── parallel_pipeline.py
│   ├── cellpose_pipeline.py
│   └── results/            ← Contiene resultados CSV e imágenes
│
├── exercise_3/             ← Autómata Celular (Incendios Forestales)
│   ├── serial_automaton.py
│   ├── mpi_automaton.py
│   ├── firms_data.csv      ← (Se genera tras ejecutar download_firms.py)
│   └── frames/             ← Gifs y estados finales
│
├── exercise_4/             ← Clustering K-Means
│   ├── serial_kmeans.py
│   └── mpi_kmeans.py
│
└── docs/                   ← Documentación
    ├── report_base.md      ← Borrador/Base del reporte
    └── report.pdf          ← Reporte final (PENDIENTE DE GENERAR)
```

## Requisitos de Software

Para asegurar la reproducibilidad de los experimentos, instala las dependencias utilizando el archivo `requirements.txt`. 

1. **Instalación local (Windows/Linux/Mac):**
   Asegúrate de tener instalado Python 3.8+ y una implementación de MPI en tu sistema (ej. OpenMPI en Linux o MS-MPI en Windows).
   ```bash
   pip install -r requirements.txt
   ```

2. **Instalación en Google Colab:**
   Usa el archivo `setup_colab.py` para configurar el entorno automáticamente.

---

## Instrucciones de Ejecución

### 1. Preparación de Datos (Solo una vez)
Algunos ejercicios requieren descargar datasets pesados o datos externos:
```bash
python download_data.py   # Descarga imágenes HeLa para Ejercicio 2
python download_firms.py  # Descarga datos NASA para Ejercicio 3
```

### 2. Cómo correr los Ejercicios
Dependiendo de tu sistema operativo, los comandos de MPI pueden variar:

#### En Linux / Google Colab (OpenMPI)
*   **Scripts Seriales/Multiprocessing:** `python exercise_X/script.py`
*   **Scripts MPI:** `mpirun --allow-run-as-root -n 4 python exercise_X/mpi_script.py`

#### En Windows (MS-MPI)
*   **Scripts Seriales/Multiprocessing:** `python exercise_X/script.py`
*   **Scripts MPI:** `mpiexec -n 4 python exercise_X/mpi_script.py`

---

## Autores y Tareas Pendientes (Equipo de 5)

Este repositorio fue construido colaborativamente. A continuación se asignan las tareas finales para la entrega:

*   **Rivaldo:** Coordinación del repositorio, extracción de scripts desde el notebook original y validación técnica del código.
*   **Persona 1 (Bianca):** Exportar `docs/report_base.md` a PDF (`docs/report.pdf`) asegurándose de incluir las gráficas generadas (`final_state.png`, etc.) y revisar la redacción de los Ejercicios 1 y 2.
*   **Persona 2 (Damian):** Completar el análisis y discusión en el reporte sobre los resultados de MPI en el Ejercicio 3 (Autómata Celular) y Ejercicio 4 (K-Means).
*   **Persona 3 (Russel):** Revisar que todas las "Evidencias de completitud" (capturas, tablas de *speedup*) estén insertadas correctamente en el PDF final según las instrucciones.
*   **Persona 4 (Jonathan):** Subir la rama final al repositorio y gestionar la entrega oficial asegurándose de que el PDF esté en la carpeta `docs/`.

---
*Nota: Para que las comparaciones de *speedup* en el Ejercicio 1 sean justas, los scripts desactivan el multithreading interno de NumPy (BLAS) asignando `OMP_NUM_THREADS=1`.*
