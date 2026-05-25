from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

# Importamos tu cerebro de IA (el archivo que exportamos de Colab)
from inference_core import LupusDetector

# 1. Inicializar la Aplicación API
app = FastAPI(
    title="LupusAI API Médica",
    description="Motor de Inferencia B2B para Detección de Lupus",
    version="1.0"
)

# 2. Configurar CORS (Crítico para que el Frontend web pueda conectarse sin bloqueos)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Cargar el modelo en memoria al iniciar el servidor
MODEL_PATH = "lupus_model.weights.h5"
print("⏳ Cargando red neuronal en memoria... (Esto puede tomar unos segundos)")
detector = LupusDetector(MODEL_PATH)
print("✅ Motor de IA cargado y listo para inferencia.")

# 4. Crear el Endpoint (La ruta que recibe las fotos de los pacientes)
@app.post("/api/v1/analizar_paciente")
async def analizar_paciente(file: UploadFile = File(...)):
    # Guardar temporalmente la imagen que envía el médico
    temp_image_path = f"temp_{file.filename}"
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Pasar la foto a la IA
        probabilidad = detector.predict(temp_image_path)
        riesgo_porcentaje = probabilidad * 100
        
        # Lógica clínica de triaje
        if riesgo_porcentaje >= 45.0:
            alerta = "🚨 ALTO RIESGO"
            recomendacion = "Derivación URGENTE a Reumatología (Sugerir prueba ANA)"
        else:
            alerta = "✅ BAJO RIESGO"
            recomendacion = "Manejo dermatológico estándar (Monitoreo ambulatorio)"

        # Borrar la foto por privacidad médica del paciente (HIPAA compliance básico)
        os.remove(temp_image_path)
        
        # Devolver el JSON estructurado al Frontend
        return {
            "estado": "exito",
            "archivo": file.filename,
            "riesgo_visual_porcentaje": round(riesgo_porcentaje, 2),
            "alerta_clinica": alerta,
            "recomendacion": recomendacion
        }
        
    except Exception as e:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return {"estado": "error", "mensaje": str(e)}

# 5. Endpoint de salud para monitoreo de servidores
@app.get("/")
def health_check():
    return {"estado": "Activo", "servicio": "LupusAI API"}