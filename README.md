# Modelo de Iris en FPGAs

Este proyecto demuestra cómo entrenar un modelo de inteligencia artificial utilizando el conjunto de datos Iris y desplegarlo en una FPGA usando `hls4ml`. El modelo está cuantizado y podado para optimizar su implementación en una FPGA.

## Set up

Para configurar el proyecto, utiliza el siguiente comando para instalar las dependencias:

```bash
poetry install
```

**Nota:** El proceso de instalación tomará un tiempo considerable, ya que incluye bibliotecas de aprendizaje automático como TensorFlow y PyTorch.

## Cómo ejecutar

Para ejecutar el proyecto y construir el modelo, utiliza:

```bash
poetry run buildmodel
```

Este comando ejecutará la clase `BuildModel`, que gestiona todo el proceso desde la creación del conjunto de datos, el entrenamiento del modelo, la generación del bitstream y la preparación de los archivos para el despliegue en la FPGA.

## Estructura del proyecto

- **`.gitignore`**: Especifica los archivos y directorios que git debe ignorar.
- **`pyproject.toml`**: Contiene metadatos del proyecto y las dependencias gestionadas con Poetry.
- **`utilities/on_target.py`**: Script para ejecutarse en la FPGA objetivo, que carga los datos de prueba, el bitfile y realiza predicciones.
- **`iris_model_on_fpgas/BuildModel.py`**: Script principal que maneja la creación del conjunto de datos, el entrenamiento del modelo, la generación del bitstream y la preparación de los archivos.
- **`iris_model_on_fpgas/__init__.py`**: Inicializa el paquete `iris_model_on_fpgas`.

## Pasos realizados por `BuildModel`

1. **Crear conjunto de datos**: Carga y preprocesa el conjunto de datos Iris, dividiéndolo en conjuntos de entrenamiento y prueba, y guarda los datos de prueba.
2. **Entrenar el modelo**: Define y entrena una red neuronal cuantizada en el conjunto de datos Iris, incorporando poda para reducir el tamaño del modelo.
3. **Construir bitstream**: Convierte el modelo entrenado de Keras a un modelo HLS utilizando `hls4ml`, lo compila y genera el bitstream para la implementación en la FPGA.
4. **Preparar archivos:**: Copia los archivos necesarios, incluyendo el bitstream, los archivos de transferencia de hardware y los scripts de control al directorio `package`.
5. **Copiar archivos:**: (Opcional) Transfiere los archivos preparados a la FPGA objetivo mediante SCP.

## Ejecución en la placa objetivo

Una vez que los archivos se copian a la FPGA objetivo, puedes ejecutar el script `on_target.py` en la placa para realizar predicciones utilizando el modelo desplegado:

```bash
python on_target.py
```

Este script:
- Carga los datos de prueba.
- Carga el bitfile en la FPGA.
- Realiza predicciones y guarda los resultados.

## Validación
La validación del modelo (es decir, la verificación de precisión) se realiza antes de sintetizar y construir el bitfile de la FPGA. En la salida aparecerá un resumen comparativo entre el modelo original y el modelo HLS, como el siguiente:
```
Accuracy of the original pruned and quantized model: 93.33%
Accuracy of the HLS model: 93.33%
```

## Dependencias

El proyecto requiere varias dependencias que son gestionadas con Poetry:

- `tensorflow`, `torch`, `hls4ml`, `qkeras`, `scikit-learn`, y más, para aprendizaje automático y optimización de modelos.
- `pynq` para operaciones relacionadas con FPGA.
- Varias bibliotecas de Jupyter para desarrollo interactivo.

Consulta el archivo `pyproject.toml` para obtener la lista completa de dependencias.

## Autores

Javier Fernández Trespando

Juan Rafael Sánchez Martínez
