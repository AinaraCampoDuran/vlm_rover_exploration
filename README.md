# VLM Rover Exploration

Este repositorio contiene el código y los recursos necesarios para desplegar y evaluar la exploración autónoma de un rover lunar utilizando Modelos de Lenguaje Visual (VLMs) en ROS 2 y una Maquina de Estados Finita en YASMIN. 

## Prerrequisitos del Sistema

Para ejecutar este proyecto, el sistema debe cumplir con los siguientes requisitos:

- **Sistema Operativo**: Ubuntu 24.04.
- **Framework Robótico**: ROS 2 Jazzy.
- **Hardware de Aceleración**: Tarjeta gráfica NVIDIA compatible con CUDA (necesario para compilar y ejecutar los modelos VLM con aceleración GPU mediante `GGML_CUDA`).

## Instalación

Sigue estos pasos para configurar tu entorno de trabajo (workspace) y compilar el proyecto:

1. **Crear el workspace de ROS 2**:
   ```bash
   mkdir -p ~/ros2_ws
   cd ~/ros2_ws
   ```

2. **Clonar este repositorio** en el workspace:
   ```bash
   git clone git@github.com:AinaraCampoDuran/vlm_rover_exploration.git
   ```

3. **Instalar dependencias de repositorios**:
   El proyecto hace uso de repositorios externos que están especificados en el archivo `dependencies.repos`. Estas dependencias incluyen versiones fijadas (en este caso, la rama `main`) para asegurar la compatibilidad:
   - `yasmin` (versión: `main`)

   Para importarlos al workspace, ejecuta:
   ```bash
   vcs import src < src/vlm_rover_exploration/dependencies.repos
   ```

4. **Instalar dependencias del sistema y de ROS**:
   ```bash
   sudo apt update
   sudo apt upgrade
   sudo apt install ros-jazzy-gz-ros2-control ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-dev-tools python3-vcstool
   
   rosdep update
   rosdep install --from-paths src --ignore-src -y
   ```

5. **Compilar el proyecto**:
   Es indispensable compilar el workspace activando la bandera `-DGGML_CUDA=ON` para habilitar la aceleración por hardware de los modelos de lenguaje:
   ```bash
   colcon build --cmake-args -DGGML_CUDA=ON
   ```

6. **Cargar el entorno**:
   ```bash
   source install/setup.bash
   ```

## Uso y Ejecución

Una vez instalado y compilado, puedes hacer uso del sistema mediante los scripts provistos. Asegúrate de estar siempre en la raíz del workspace y haber hecho el `source` del entorno (`source install/setup.bash`).

### Iniciar el Simulador
Para iniciar únicamente el entorno de simulación en Gazebo y probar el rover, puedes usar el script de inicio de simulación:

```bash
ros2 launch test_sim.launch.py world_script:=low_moon.launch.py
```

### Ejecutar Experimentos y Evaluaciones con VLMs
Para lanzar de forma automatizada las evaluaciones y repeticiones de los modelos VLM (MiniCPM, Qwen3-VL, InternVL3), utiliza el script de experimentos:

```bash
./run_experiments.sh
```

Este script se encargará de iterar por los diferentes modelos, ejecutarlos bajo un límite de tiempo preestablecido, y limpiar los nodos de ROS 2 y procesos de Gazebo residuales entre cada ejecución utilizando el script `cleanup_ros.sh`.

### Ejecutar un modelo de forma manual
Si deseas lanzar la exploración con un modelo específico manualmente en lugar de utilizar el script automatizado de experimentos, puedes usar el *launch file* directamente:

```bash
ros2 launch vlm_rover_exploration_bringup vlm_rover_exploration.launch.py vlm_model:=MiniCPM.yaml
```

### Análisis y Visualización de Resultados
Tras finalizar las ejecuciones de los experimentos, puedes procesar los datos recolectados y generar gráficos comparativos utilizando los scripts proporcionados en el paquete `vlm_rover_exploration`. 

Asegúrate de ejecutar ambos scripts desde la **raíz del workspace** (`~/ros2_ws`):

1. **Procesar métricas (Benchmark)**:
   Este script busca todos los archivos de resultados brutos (`raw_metrics_*.json`), calcula estadísticas detalladas y genera resúmenes para cada modelo.
   ```bash
   python3 src/vlm_rover_exploration/vlm_rover_exploration/scripts/benchmark.py
   ```

2. **Generar gráficos y visualizaciones**:
   Una vez procesadas las métricas, utiliza este script para generar gráficas de barras, diagramas de caja (boxplots) y frecuencias de estrategias de los modelos. Las imágenes resultantes se guardarán en un nuevo directorio llamado `benchmark_visualizations/`.
   ```bash
   python3 src/vlm_rover_exploration/vlm_rover_exploration/scripts/visualize_results.py
   ```