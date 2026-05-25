# LupusAI: Motor B2B de Triaje Dermatológico Híbrido

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css)

> **Software as a Medical Device (SaMD) en fase de prototipo.** > Sistema de Visión Artificial diseñado para apoyar a médicos generales en la detección temprana del Lupus Eritematoso Sistémico (LES) en zonas de primer nivel de atención.

---

## El Valor para el Cliente (El Problema y la Solución)
Actualmente, el sistema de salud en zonas rurales y de difícil acceso sufre cuellos de botella severos en reumatología y dermatología. Un paciente con eritema malar puede tardar meses en ser derivado al especialista correcto.

**LupusAI no reemplaza al médico; le otorga "supervisión".** A través de un microservicio de Inteligencia Artificial accesible desde cualquier dispositivo, un médico general puede fotografiar una lesión cutánea sospechosa y recibir, en menos de un segundo, una segunda opinión matemática. Si la IA detecta alto riesgo, el paciente es derivado de urgencia, optimizando los recursos del sistema de salud y salvando tiempo vital.

---

## Transparencia del Stack Tecnológico
La arquitectura del sistema ha sido diseñada bajo estándares B2B, dividiendo las responsabilidades para garantizar escalabilidad:

| Componente | Tecnología | Justificación Técnica |
| :--- | :--- | :--- |
| **Core IA** | `TensorFlow` / `Keras` | Entrenamiento, inferencia y cálculo de tensores matemáticos. |
| **Modelo Base** | `EfficientNetB0` | *Transfer Learning* de ImageNet. Elegido por su bajo peso en memoria (~16MB), crucial para despliegues *Edge* en hospitales con bajo ancho de banda. |
| **Backend (API)** | `FastAPI` / `Uvicorn` | Recepción asíncrona de imágenes y comunicación RESTful ultrarrápida. |
| **Frontend (UI)** | `HTML5` / `Tailwind CSS`| Interfaz clínica responsiva y sin dependencias pesadas, adaptable a cualquier pantalla de consultorio. |

---

## Metodología y Validez de las Métricas Clínicas
El modelo se enfrentó a un problema complejo: distinguir el Lupus de otras 20 patologías cutáneas similares (como Rosácea, Eczema o Psoriasis). Se implementó un **Clasificador Binario (Lupus vs. No-Lupus)** utilizando *Clipped Class Weights* para evitar la destrucción de gradientes en el entrenamiento.

* **AUC (Área Bajo la Curva): `0.84`** - El modelo tiene un 84% de capacidad discriminatoria ciega entre un paciente sano/con otra patología y uno con Lupus.
* **Recall (Sensibilidad Clínicamente Ajustada): `0.64`** - En el ámbito médico, priorizamos minimizar los *Falsos Negativos* (no dejar escapar a un paciente enfermo).
* **Interpretabilidad Neuronal:** Se integró el algoritmo **Grad-CAM** en la fase de I+D. La IA no es una caja negra; audita y mapea térmicamente en rojo las zonas exactas de la piel (como las mejillas) que influyeron en el diagnóstico.

---

## Reproducibilidad Científica (Guía de Instalación)
El proyecto cuenta con un entorno aislado para fácil reproducción en cualquier máquina local.

**1. Clonar el repositorio y crear entorno:**
```bash
git clone [https://github.com/ferchoobarba/Lupus_IA_Proyect/tree/main](https://github.com/ferchoobarba/Lupus_IA_Proyect/tree/main)
cd LupusAI/backend
python -m venv venv