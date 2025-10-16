from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os

# Importar funcionalidades desde main.py
from main import (
    APIKeyManager,
    ResultadoExpediente,
    procesar_expedientes_concurrente,
    InfoExpedienteCSV,
    buscar_expedientes_en_csv
)

# Cargar variables de entorno
load_dotenv()

# ============================================================================
# MODELO PYDANTIC PARA REQUEST
# ============================================================================

class ExpedienteRequest(BaseModel):
    """Modelo para el request de procesamiento de expedientes"""
    urls: List[str] = Field(
        description="Lista de URLs de expedientes a procesar",
        min_length=1
    )


class BusquedaExpedientesRequest(BaseModel):
    """Request para buscar expedientes en CSV"""
    ids_expedientes: List[str] = Field(
        description="Lista de IDs de expedientes a buscar",
        min_length=1
    )


# ============================================================================
# CONFIGURACIÓN DE FASTAPI
# ============================================================================

app = FastAPI(
    title="API de Procesamiento de Expedientes Judiciales",
    description="API para procesar expedientes judiciales mexicanos y extraer acuerdos recientes",
    version="1.0.0"
)

# Configurar CORS para permitir peticiones desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# INICIALIZACIÓN DE API KEYS
# ============================================================================

def obtener_api_keys() -> List[str]:
    """Obtiene las API keys desde las variables de entorno"""
    api_keys = [
        os.getenv('OPENAI_API_KEY_1'),
        os.getenv('OPENAI_API_KEY_2'),
        os.getenv('OPENAI_API_KEY_3')
    ]
    
    # Filtrar None en caso de que falte alguna key
    api_keys = [key for key in api_keys if key is not None]
    
    if not api_keys:
        raise ValueError(
            "No se encontraron API keys. Configura las variables de entorno "
            "OPENAI_API_KEY_1, OPENAI_API_KEY_2, OPENAI_API_KEY_3"
        )
    
    return api_keys


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raíz con información básica de la API"""
    return {
        "mensaje": "API de Procesamiento de Expedientes Judiciales",
        "version": "1.0.0",
        "endpoints": {
            "POST /procesar-expedientes": "Procesar lista de URLs de expedientes",
            "POST /buscar-expedientes": "Buscar expedientes por ID en CSV",
            "GET /health": "Verificar estado de la API"
        }
    }


@app.get("/health")
async def health():
    """Endpoint para verificar el estado de la API"""
    try:
        api_keys = obtener_api_keys()
        return {
            "status": "ok",
            "api_keys_disponibles": len(api_keys)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/procesar-expedientes", response_model=List[ResultadoExpediente])
async def procesar_expedientes(request: ExpedienteRequest):
    """
    Procesa una lista de URLs de expedientes y devuelve los resultados
    
    Args:
        request: ExpedienteRequest con lista de URLs
    
    Returns:
        Lista de ResultadoExpediente con los acuerdos más recientes de cada expediente
    """
    
    try:
        # Obtener API keys
        api_keys = obtener_api_keys()
        
        # Inicializar gestor de API keys
        api_manager = APIKeyManager(api_keys)
        
        # Configuración: True = GPT-5, False = GPT-4o
        USAR_GPT5 = False
        MAX_CONCURRENCIA = 5
        
        # Procesar expedientes de forma concurrente
        resultados = await procesar_expedientes_concurrente(
            urls=request.urls,
            api_key_manager=api_manager,
            usar_gpt5=USAR_GPT5,
            max_concurrencia=MAX_CONCURRENCIA
        )
        
        return resultados
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando expedientes: {str(e)}")


@app.post("/buscar-expedientes", response_model=List[InfoExpedienteCSV])
async def buscar_expedientes(request: BusquedaExpedientesRequest):
    """
    Busca expedientes en CSV remoto por sus IDs
    
    Args:
        request: BusquedaExpedientesRequest con lista de IDs
    
    Returns:
        Lista de InfoExpedienteCSV con información encontrada
    """
    
    # URL del CSV en GitHub
    URL_CSV = "https://raw.githubusercontent.com/berrodriquez26/SBM/refs/heads/main/data/expedientes.csv"
    
    try:
        # Buscar expedientes en el CSV
        resultados = await buscar_expedientes_en_csv(
            ids_expedientes=request.ids_expedientes,
            url_csv=URL_CSV
        )
        
        return resultados
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error buscando expedientes: {str(e)}")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Ejecutar servidor en desarrollo
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=False
    )

