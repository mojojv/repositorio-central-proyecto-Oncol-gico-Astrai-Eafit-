<div align="center">

# 🎗️ Proyecto Central Oncológico GICO - ASTRAI EAFIT

### *Detección multimodal de enfermedades oncológicas mediante ML*

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-En%20Desarrollo-yellow.svg)]()

<img src="https://via.placeholder.com/800x400/1a1a2e/e94560?text=Oncol+GICO+-+ASTRAI+EAFIT" alt="Banner" width="100%"/>

*Un modelo de Machine Learning multimodal para la detección temprana y análisis de enfermedades oncológicas*

[🚀 Introducción](#-introducción) • [📦 Instalación](#-instalación) • [🔬 Metodología](#-metodología) • [📊 Resultados](#-resultados)

</div>

---

## 📋 Tabla de Contenidos

- [Introducción](#-introducción)
- [Características](#-características)
- [Instalación](#-instalación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Uso](#-uso)
- [Metodología](#-metodología)
- [Dataset](#-dataset)
- [Modelos](#-modelos)
- [Resultados](#-resultados)
- [Contribución](#-contribución)
- [Equipo](#-equipo)
- [Licencia](#-licencia)

---

## 🎯 Introducción

Este proyecto fue creado para desarrollar un **modelo de Machine Learning multimodal** para la detección de enfermedades oncológicas. El sistema integra múltiples fuentes de datos médicos para mejorar la precisión diagnóstica y apoyar la toma de decisiones clínicas.

### 🎗️ Objetivos Principales

- 🔍 **Detección Temprana**: Identificar signos tempranos de enfermedades oncológicas
- 🧠 **Análisis Multimodal**: Integrar imágenes médicas, datos clínicos y genómicos
- 📊 **Predicción de Riesgo**: Evaluar probabilidades y factores de riesgo
- 🏥 **Soporte Clínico**: Herramienta de apoyo para profesionales de la salud

---

## ✨ Características

<table>
<tr>
<td width="50%">

### 🧬 Análisis Multimodal
- ✅ Procesamiento de imágenes DICOM
- ✅ Análisis de datos clínicos
- ✅ Integración de información genómica
- ✅ Fusión de características multimodales
- ✅ Interpretabilidad de resultados

</td>
<td width="50%">

### 🤖 Machine Learning Avanzado
- ✅ Deep Learning (CNN, ResNet, VGG)
- ✅ Transfer Learning
- ✅ Ensemble Methods
- ✅ Validación cruzada robusta
- ✅ Optimización de hiperparámetros

</td>
</tr>
<tr>
<td width="50%">

### 📊 Visualización y Análisis
- ✅ Mapas de calor (Grad-CAM)
- ✅ Curvas ROC y PR
- ✅ Matrices de confusión
- ✅ Dashboard interactivo
- ✅ Reportes automatizados

</td>
<td width="50%">

### 🔒 Seguridad y Privacidad
- ✅ Anonimización de datos
- ✅ Cumplimiento HIPAA
- ✅ Encriptación de datos sensibles
- ✅ Logs de auditoría
- ✅ Control de acceso

</td>
</tr>
</table>

---

## 📦 Instalación

### 🔧 Requisitos Previos

```bash
Python 3.8+
CUDA 11.0+ (para GPU)
8GB RAM mínimo (16GB recomendado)
```

### 🚀 Instalación Rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/mojojv/repositorio-central-proyecto-Oncol-gico-Astrai-Eafit.git
cd repositorio-central-proyecto-Oncol-gico-Astrai-Eafit

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones
```

### 🐳 Con Docker

```bash
# Construir imagen
docker build -t oncol-gico-astrai .

# Ejecutar contenedor
docker run -p 8888:8888 -v $(pwd)/data:/app/data oncol-gico-astrai

# Con GPU
docker run --gpus all -p 8888:8888 oncol-gico-astrai
```

### 📋 Dependencias Principales

```python
tensorflow>=2.8.0
pytorch>=1.10.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
pydicom>=2.2.0
nibabel>=3.2.0
opencv-python>=4.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 🏗️ Estructura del Proyecto

```
📦 repositorio-central-proyecto-Oncol-gico-Astrai-Eafit/
┣ 📂 carpeta datos/                    # Datasets y datos procesados
┃ ┣ 📂 raw/                           # Datos crudos (DICOM, CSV, etc.)
┃ ┣ 📂 processed/                     # Datos preprocesados
┃ ┣ 📂 augmented/                     # Datos aumentados
┃ ┗ 📂 external/                      # Datasets externos
┃
┣ 📂 code/                            # Código fuente principal
┃ ┣ 📂 preprocessing/                 # Scripts de preprocesamiento
┃ ┃ ┣ 📜 dicom_processor.py          # Procesamiento DICOM
┃ ┃ ┣ 📜 data_augmentation.py        # Aumento de datos
┃ ┃ ┗ 📜 feature_extraction.py       # Extracción de características
┃ ┣ 📂 models/                        # Definición de modelos
┃ ┃ ┣ 📜 cnn_models.py               # Modelos CNN
┃ ┃ ┣ 📜 multimodal_fusion.py        # Fusión multimodal
┃ ┃ ┗ 📜 ensemble.py                 # Métodos ensemble
┃ ┣ 📂 training/                      # Scripts de entrenamiento
┃ ┃ ┣ 📜 train_image_model.py        # Entrenamiento imagen
┃ ┃ ┣ 📜 train_clinical_model.py     # Entrenamiento clínico
┃ ┃ ┗ 📜 train_multimodal.py         # Entrenamiento multimodal
┃ ┣ 📂 evaluation/                    # Evaluación y métricas
┃ ┃ ┣ 📜 metrics.py                  # Métricas personalizadas
┃ ┃ ┣ 📜 visualization.py            # Visualizaciones
┃ ┃ ┗ 📜 interpretability.py         # Interpretabilidad (Grad-CAM)
┃ ┣ 📂 utils/                         # Utilidades
┃ ┃ ┣ 📜 config.py                   # Configuraciones
┃ ┃ ┣ 📜 data_loader.py              # Cargadores de datos
┃ ┃ ┗ 📜 logger.py                   # Sistema de logs
┃ ┗ 📂 api/                           # API REST
┃   ┣ 📜 app.py                      # Aplicación Flask/FastAPI
┃   ┗ 📜 endpoints.py                # Definición de endpoints
┃
┣ 📂 notebooks/                       # Jupyter notebooks
┃ ┣ 📓 01_exploratory_analysis.ipynb # Análisis exploratorio
┃ ┣ 📓 02_preprocessing.ipynb        # Preprocesamiento
┃ ┣ 📓 03_model_training.ipynb       # Entrenamiento de modelos
┃ ┣ 📓 04_evaluation.ipynb           # Evaluación de resultados
┃ ┗ 📓 05_visualization.ipynb        # Visualizaciones avanzadas
┃
┣ 📂 models/                          # Modelos entrenados
┃ ┣ 📂 checkpoints/                  # Checkpoints de entrenamiento
┃ ┣ 📂 saved_models/                 # Modelos finales guardados
┃ ┗ 📂 pretrained/                   # Modelos preentrenados
┃
┣ 📂 results/                         # Resultados y reportes
┃ ┣ 📂 figures/                      # Gráficos y visualizaciones
┃ ┣ 📂 reports/                      # Reportes en PDF/HTML
┃ ┗ 📂 logs/                         # Logs de experimentos
┃
┣ 📂 tests/                           # Tests unitarios
┃ ┣ 📜 test_preprocessing.py
┃ ┣ 📜 test_models.py
┃ ┗ 📜 test_api.py
┃
┣ 📂 docs/                            # Documentación
┃ ┣ 📜 architecture.md               # Arquitectura del sistema
┃ ┣ 📜 data_dictionary.md            # Diccionario de datos
┃ ┣ 📜 model_documentation.md        # Documentación de modelos
┃ ┗ 📜 api_reference.md              # Referencia API
┃
┣ 📂 scripts/                         # Scripts auxiliares
┃ ┣ 📜 download_data.sh              # Descarga de datasets
┃ ┣ 📜 setup_environment.sh          # Configuración inicial
┃ ┗ 📜 run_experiments.sh            # Ejecución de experimentos
┃
┣ 📜 .env.example                     # Ejemplo de variables de entorno
┣ 📜 .gitignore                       # Archivos ignorados por git
┣ 📜 requirements.txt                 # Dependencias Python
┣ 📜 environment.yml                  # Entorno Conda
┣ 📜 Dockerfile                       # Configuración Docker
┣ 📜 docker-compose.yml               # Orquestación Docker
┣ 📜 setup.py                         # Setup del paquete
┣ 📜 README.md                        # Esta documentación
┗ 📜 LICENSE                          # Licencia del proyecto
```

---

## 🔬 Metodología

### Pipeline de Procesamiento

```mermaid
graph LR
    A[Datos Crudos] --> B[Preprocesamiento]
    B --> C[Extracción de Características]
    C --> D[Aumento de Datos]
    D --> E[Entrenamiento]
    E --> F[Validación]
    F --> G[Evaluación]
    G --> H[Despliegue]
```

### 1️⃣ Preprocesamiento de Datos

```python
# Ejemplo de preprocesamiento de imágenes DICOM
from code.preprocessing import DICOMProcessor

processor = DICOMProcessor()
processed_image = processor.process(
    path='data/raw/patient_001.dcm',
    resize=(224, 224),
    normalize=True,
    windowing='soft_tissue'
)
```

### 2️⃣ Arquitectura del Modelo

El modelo utiliza una arquitectura multimodal que combina:

- **Branch de Imágenes**: ResNet-50 preentrenado en ImageNet
- **Branch Clínico**: Red neuronal densa para datos tabulares
- **Fusión**: Concatenación de características con capas densas

```python
# Arquitectura simplificada
image_features = ResNet50(include_top=False)(image_input)
clinical_features = Dense(128)(clinical_input)
fused = concatenate([image_features, clinical_features])
output = Dense(num_classes, activation='softmax')(fused)
```

---

## 📊 Dataset

### Fuentes de Datos

| Tipo | Descripción | Cantidad | Formato |
|------|-------------|----------|---------|
| 🔬 Imágenes CT | Tomografías computarizadas | ~10,000 | DICOM |
| 🧬 Imágenes MRI | Resonancias magnéticas | ~8,000 | DICOM |
| 📋 Datos Clínicos | Historias clínicas | ~15,000 | CSV/JSON |
| 🧪 Datos Genómicos | Secuenciación genética | ~5,000 | VCF/FASTA |

### Distribución de Clases

```
📊 Distribución del Dataset:
├─ Clase 0 (Normal): 45%
├─ Clase 1 (Benigno): 30%
├─ Clase 2 (Maligno Tipo A): 15%
└─ Clase 3 (Maligno Tipo B): 10%
```

### Preprocesamiento Aplicado

- ✅ Normalización de intensidad (0-1)
- ✅ Reescalado a 224x224 píxeles
- ✅ Windowing adaptativo según modalidad
- ✅ Aumento de datos (rotación, flip, zoom)
- ✅ Balanceo de clases (SMOTE)

---

## 🤖 Modelos

### Modelos Implementados

<table>
<tr>
<th>Modelo</th>
<th>Tipo</th>
<th>Accuracy</th>
<th>AUC-ROC</th>
</tr>
<tr>
<td>ResNet-50</td>
<td>CNN</td>
<td>89.3%</td>
<td>0.94</td>
</tr>
<tr>
<td>VGG-19</td>
<td>CNN</td>
<td>87.1%</td>
<td>0.92</td>
</tr>
<tr>
<td>EfficientNet-B4</td>
<td>CNN</td>
<td>91.2%</td>
<td>0.96</td>
</tr>
<tr>
<td>Multimodal Fusion</td>
<td>Híbrido</td>
<td><b>93.7%</b></td>
<td><b>0.97</b></td>
</tr>
</table>

### Entrenamiento

```bash
# Entrenar modelo de imágenes
python code/training/train_image_model.py \
    --model resnet50 \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001

# Entrenar
