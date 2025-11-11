# -----------------------------------------------------------------
# ARCHIVO COMPLETO: main.py
# Microservicio de IA y KPIs (Python/FastAPI)
# -----------------------------------------------------------------

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware  # <-- Importación de CORS
import shutil
import random
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- 1. Base de Datos Simulada (para IA y KPIs) ---
# (Basada en tu archivo menu.txt y la imagen a2745e.png)
PLATILLOS_DB = {
    "Bistec Encebollado": {
        "plato_detectado": "Bistec Encebollado",
        "nombres_alternativos": ["Steak and Onions"],
        "ingredientes": ["Bistec de res", "Cebolla", "Ajo", "Salsa de soya"],
        "alergias": ["Soya"]
    },
    "Cotoletta Alla Milanese": {
        "plato_detectado": "Cotoletta Alla Milanese",
        "nombres_alternativos": ["Milanesa de Ternera"],
        "ingredientes": ["Ternera (Chuleta)", "Pan rallado", "Huevo", "Manteca"],
        "alergias": ["Gluten (Pan)", "Huevo", "Lacteos (Manteca)"]
    },
    "Mozzarella In Carrozza": {
        "plato_detectado": "Mozzarella In Carrozza",
        "nombres_alternativos": ["Sándwich de mozzarella frito"],
        "ingredientes": ["Queso Mozzarella", "Pan de molde", "Huevo", "Harina", "Leche"],
        "alergias": ["Gluten (Pan, Harina)", "Lacteos (Queso, Leche)", "Huevo"]
    },
    "Parmigiana Di Melanzane": {
        "plato_detectado": "Parmigiana Di Melanzane",
        "nombres_alternativos": ["Lasaña de Berenjena"],
        "ingredientes": ["Berenjena", "Salsa de tomate", "Queso Parmesano", "Albahaca"],
        "alergias": ["Lacteos (Queso)"]
    },
}

PLATO_DESCONOCIDO = {
    "plato_detectado": "Plato Desconocido",
    "nombres_alternativos": [],
    "ingredientes": ["Ingredientes no encontrados"],
    "alergias": ["No detectadas"]
}

platos_simulados = list(PLATILLOS_DB.keys())


# --- 2. Creación de la App y Configuración de CORS ---
app = FastAPI()

# Configuración de CORS
origins = [
    "http://localhost:1337",  # La URL de tu panel de Strapi
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 3. Endpoints de la API ---

@app.get("/")
def read_root():
    return {"Hello": "IA Backend"}

# --- Endpoint de Reconocimiento de IA (Simulado) ---
@app.post("/analizar-plato")
async def analizar_plato(
    nombre_plato: str = Form(...), 
    foto: UploadFile = File(...)
):
    
    print(f"Foto recibida ({foto.filename}), buscando datos para: {nombre_plato}")
    
    # Busca el nombre del plato en nuestra DB simulada
    respuesta = PLATILLOS_DB.get(nombre_plato, PLATO_DESCONOCIDO)
    
    return respuesta

# --- Endpoint de KPI (Simula +5000 datos en 3 gestiones) ---
@app.get("/api/kpi/ventas-historicas")
async def get_kpi_ventas():
    
    gestiones = [2023, 2024, 2025]
    nombres_platillos = list(PLATILLOS_DB.keys()) 
    
    reporte_total = {
        "total_ventas_simuladas": 0,
        "reporte_por_gestion": []
    }
    
    total_ventas = 0

    for gestion in gestiones:
        ventas_gestion = {
            "gestion": gestion,
            "ventas_por_plato": []
        }
        
        for plato in nombres_platillos:
            ventas_simuladas = random.randint(300, 900) 
            ventas_gestion["ventas_por_plato"].append({
                "plato": plato,
                "ventas": ventas_simuladas
            })
            total_ventas += ventas_simuladas
    
        reporte_total["reporte_por_gestion"].append(ventas_gestion)

    reporte_total["total_ventas_simuladas"] = total_ventas 
    
    return reporte_total

# --- Endpoint de Machine Learning (Predicción) ---
@app.get("/api/prediccion/demanda")
async def predecir_demanda(plato: str):
    
    kpi_data = await get_kpi_ventas()
    
    datos_plato = []
    for gestion_data in kpi_data["reporte_por_gestion"]:
        gestion = gestion_data["gestion"]
        for plato_data in gestion_data["ventas_por_plato"]:
            if plato_data["plato"] == plato:
                datos_plato.append({"gestion": gestion, "ventas": plato_data["ventas"]})

    if not datos_plato:
        return {"error": "Plato no encontrado en los datos históricos"}

    df = pd.DataFrame(datos_plato)
    
    X = df[['gestion']] 
    y = df['ventas']
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    gestion_siguiente = pd.DataFrame([[2026]], columns=['gestion'])
    prediccion = modelo.predict(gestion_siguiente)
    
    ventas_predichas = round(prediccion[0])
    
    return {
        "plato_solicitado": plato,
        "prediccion_gestion_2026": f"Se estima una demanda de {ventas_predichas} unidades.",
        "datos_historicos_usados": datos_plato
    }