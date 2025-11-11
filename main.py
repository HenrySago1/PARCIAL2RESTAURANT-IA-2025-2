# Archivo: main.py
# (Versión con base de datos de platillos)
import pandas as pd
from sklearn.linear_model import LinearRegression

from fastapi import FastAPI, UploadFile, File, Form
import random

# 1. Creamos nuestra base de datos simulada
# (Añadí los ingredientes y alergias de ejemplo)
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

# Un plato por defecto si no lo encontramos
PLATO_DESCONOCIDO = {
    "plato_detectado": "Plato Desconocido",
    "nombres_alternativos": [],
    "ingredientes": ["Ingredientes no encontrados"],
    "alergias": ["No detectadas"]
}


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "IA Backend"}

# 2. RUTA ACTUALIZADA
# Ahora recibe un archivo "foto" Y un campo de formulario "nombre_plato"
@app.post("/analizar-plato")
async def analizar_plato(
    nombre_plato: str = Form(...), 
    foto: UploadFile = File(...)
):
    
    print(f"Foto recibida ({foto.filename}), buscando datos para: {nombre_plato}")
    
    # 3. La lógica de IA simulada
    # Busca el nombre del plato en nuestra DB de platillos
    # Si no lo encuentra, devuelve el plato desconocido
    respuesta = PLATILLOS_DB.get(nombre_plato, PLATO_DESCONOCIDO)
    
    return respuesta

# --- NUEVO ENDPOINT DE KPI ---
# Simula +5000 datos en 3 gestiones
@app.get("/api/kpi/ventas-historicas")
async def get_kpi_ventas():
    
    # 1. Definimos las 3 gestiones (años)
    gestiones = [2023, 2024, 2025]
    
    # 2. Obtenemos los nombres de tus platillos de la DB que ya teníamos
    nombres_platillos = list(PLATILLOS_DB.keys()) 
    
    reporte_total = {
        "total_ventas_simuladas": 0,
        "reporte_por_gestion": []
    }
    
    total_ventas = 0

    # 3. Simulamos los datos
    for gestion in gestiones:
        ventas_gestion = {
            "gestion": gestion,
            "ventas_por_plato": []
        }
        
        # Simulamos las ventas para cada platillo en esa gestión
        for plato in nombres_platillos:
            # Generamos un número aleatorio grande de ventas (para sumar +5000)
            ventas_simuladas = random.randint(300, 900) 
            ventas_gestion["ventas_por_plato"].append({
                "plato": plato,
                "ventas": ventas_simuladas
            })
            total_ventas += ventas_simuladas
    
        reporte_total["reporte_por_gestion"].append(ventas_gestion)

    reporte_total["total_ventas_simuladas"] = total_ventas # Esto será > 5000
    
    return reporte_total

# --- NUEVO ENDPOINT DE MACHINE LEARNING (PREDICCIÓN) ---
# Predice la demanda de un plato específico

@app.get("/api/prediccion/demanda")
async def predecir_demanda(plato: str):
    
    # 1. OBTENER LOS DATOS HISTÓRICOS (Llamamos a nuestra propia API de KPI)
    # (En un proyecto real, esto sería una consulta a la base de datos)
    kpi_data = await get_kpi_ventas()
    
    # 2. PREPARAR LOS DATOS PARA EL MODELO
    # Convertimos el JSON en una estructura que el ML pueda entender
    
    datos_plato = []
    for gestion_data in kpi_data["reporte_por_gestion"]:
        gestion = gestion_data["gestion"]
        for plato_data in gestion_data["ventas_por_plato"]:
            if plato_data["plato"] == plato:
                datos_plato.append({"gestion": gestion, "ventas": plato_data["ventas"]})

    # Si no encontramos el plato, no podemos predecir
    if not datos_plato:
        return {"error": "Plato no encontrado en los datos históricos"}

    # 3. CREAR EL MODELO DE ML (Regresión Lineal Simple)
    # Convertimos los datos a un DataFrame de Pandas
    df = pd.DataFrame(datos_plato)
    
    # X = El año (la "gestión")
    # y = La cantidad de ventas
    X = df[['gestion']] 
    y = df['ventas']
    
    # Creamos y "entrenamos" el modelo de ML
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # 4. HACER LA PREDICCIÓN
    # Vamos a predecir la demanda para la siguiente gestión (2026)
    gestion_siguiente = pd.DataFrame([[2026]], columns=['gestion'])
    prediccion = modelo.predict(gestion_siguiente)
    
    # Redondeamos el resultado
    ventas_predichas = round(prediccion[0])
    
    # 5. DEVOLVER LA RESPUESTA
    return {
        "plato_solicitado": plato,
        "prediccion_gestion_2026": f"Se estima una demanda de {ventas_predichas} unidades.",
        "datos_historicos_usados": datos_plato
    }