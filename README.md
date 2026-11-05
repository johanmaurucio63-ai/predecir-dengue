# ComplicaSense

Proyecto académico de analítica de datos para estimar el riesgo de dengue complicado en población pediátrica. Incluye un notebook reproducible, dos modelos y una interfaz web clínica con predicción individual y por lote.

## Estructura

```text
├── notebooks/01_exploracion_modelado_complicasense.ipynb
├── data_pipeline.py       # limpieza y variables derivadas compartidas
├── train.py               # entrenamiento automatizado para Render
├── app.py                 # API Flask y servidor web
├── templates/             # frontend
├── static/                # estilos y JavaScript
├── MASTER_CHART...xlsx    # dataset original
└── render.yaml
```

## Opciones de la página

- **Predicción individual:** diligencia el perfil de un paciente y compara la probabilidad estimada por regresión logística y red neuronal.
- **Predicción por lote:** carga un archivo `.csv` o `.xlsx` de hasta 5.000 filas. La aplicación acepta el Excel original o la plantilla disponible en **Descargar plantilla CSV**. Si el archivo incluye `target`, muestra accuracy, precision, recall, F1, ROC-AUC y las matrices de confusión de ambos modelos. También permite descargar las predicciones.

El endpoint de lote es `POST /api/predict-batch` y recibe el archivo en el campo multipart `file`.

## Ejecutar el notebook

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_exploracion_modelado_complicasense.ipynb
```

Ejecuta las celdas en orden. El notebook documenta la exploración, los valores faltantes, la conversión de códigos, las variables derivadas, las métricas y la exportación de `artifacts/models.joblib`.

## Ejecutar localmente

```bash
python train.py
python app.py
```

Abre `http://localhost:5000`.

## Despliegue en Render

1. Sube esta carpeta a un repositorio de GitHub.
2. En Render selecciona **New > Blueprint** y conecta el repositorio.
3. Render leerá `render.yaml`, instalará dependencias, ejecutará el notebook completo para entrenar los modelos y después iniciará `gunicorn app:app`.
4. Copia el enlace `https://complicasense.onrender.com` (o el dominio que Render asigne) para la entrega.

El proyecto no guarda datos enviados por los usuarios. La interfaz es una demostración educativa y no reemplaza la valoración de un profesional de la salud.
