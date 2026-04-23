# ==============================================================
# HPC Unit 3 - Setup inicial para Google Colab
# Corre este script al inicio de cada sesión
# ==============================================================

# ── 1. Instalar MPI en el sistema (necesario para mpi4py) ─────
!apt-get install -y -q libopenmpi-dev openmpi-bin




# ── 2. Instalar todas las dependencias Python ─────────────────
!pip install -q \
    numpy scipy matplotlib pandas seaborn \
    mpi4py \
    cellpose scikit-image opencv-python-headless Pillow tqdm \
    requests geopandas shapely pyproj imageio \
    scikit-learn ucimlrepo \
    psutil joblib ipywidgets