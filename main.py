from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import aiohttp
import json
import os
import re
import pandas as pd
from io import StringIO

# Cargar variables de entorno desde archivo .env
load_dotenv()

# ============================================================================
# MODELOS PYDANTIC PARA OUTPUT ESTRUCTURADO
# ============================================================================

class Acuerdo(BaseModel):
    """Modelo para un acuerdo individual del expediente"""
    numero: int = Field(description="Número del acuerdo")
    fecha_auto: str = Field(description="Fecha del auto en formato DD-MM-YYYY")
    tipo_cuaderno: str = Field(description="Tipo de cuaderno (Principal, etc)")
    fecha_publicacion: str = Field(description="Fecha de publicación en formato DD-MM-YYYY")
    resumen: str = Field(description="Resumen completo del acuerdo")
    es_hoy: bool = Field(description="True si la fecha del auto es la fecha de hoy, False en caso contrario")


class ResultadoExpediente(BaseModel):
    """Modelo para el resultado completo del análisis de un expediente"""
    url_expediente: str = Field(description="URL del expediente consultado")
    fecha_consulta: str = Field(description="Fecha de la consulta en formato DD-MM-YYYY")
    total_acuerdos: int = Field(description="Número total de acuerdos encontrados")
    acuerdos_recientes: List[Acuerdo] = Field(
        description="Los 3 acuerdos más recientes, ordenados del más reciente al más antiguo"
    )


class InfoExpedienteCSV(BaseModel):
    """Información de un expediente desde el CSV"""
    num_expediente: str = Field(description="Número del expediente")
    parte_actora: str = Field(description="Parte actora del expediente")
    parte_demandada: str = Field(description="Parte demandada del expediente")
    parte_notificada: str = Field(description="Parte notificada del expediente")
    fecha_actuacion: str = Field(description="Fecha de actuación")
    sintesis: str = Field(description="Síntesis del expediente")
    magistrado: str = Field(description="Magistrado asignado")
    secretario: str = Field(description="Secretario asignado")
    sala: str = Field(description="Sala del expediente")
    encontrado: bool = Field(description="Indica si el expediente fue encontrado en el CSV")


# ============================================================================
# GESTOR DE API KEYS CON ROTACIÓN
# ============================================================================

class APIKeyManager:
    """Gestiona múltiples API keys con rotación round-robin"""
    
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("Se debe proporcionar al menos una API key")
        self.keys = keys
        self.current_index = 0
        self.lock = asyncio.Lock()
    
    async def get_next_key(self) -> str:
        """Obtiene la siguiente API key en la rotación (async)"""
        async with self.lock:
            key = self.keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.keys)
            return key


# ============================================================================
# FUNCIÓN PRINCIPAL CON API DE RESPONSES DE OPENAI
# ============================================================================

async def verificar_expediente_con_api(url_expediente: str, api_key: str, usar_gpt5: bool = True) -> ResultadoExpediente:
    """
    Verifica actualizaciones en el expediente usando la API de Responses de OpenAI con web search (async)
    
    Args:
        url_expediente: URL del expediente a consultar
        api_key: API key de OpenAI
        usar_gpt5: Si True, intenta usar GPT-5, si False usa GPT-4o
    
    Returns:
        ResultadoExpediente con los 3 acuerdos más recientes
    """
    
    # Obtener fecha actual
    fecha_hoy = datetime.now().strftime("%d-%m-%Y")
    
    # Determinar el modelo a usar
    model = "gpt-5" if usar_gpt5 else "gpt-4o"
    
    # Crear el esquema JSON para el output estructurado
    schema = {
        "type": "object",
        "properties": {
            "url_expediente": {"type": "string"},
            "fecha_consulta": {"type": "string"},
            "total_acuerdos": {"type": "integer"},
            "acuerdos_recientes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "numero": {"type": "integer"},
                        "fecha_auto": {"type": "string"},
                        "tipo_cuaderno": {"type": "string"},
                        "fecha_publicacion": {"type": "string"},
                        "resumen": {"type": "string"},
                        "es_hoy": {"type": "boolean"}
                    },
                    "required": ["numero", "fecha_auto", "tipo_cuaderno", "fecha_publicacion", "resumen", "es_hoy"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["url_expediente", "fecha_consulta", "total_acuerdos", "acuerdos_recientes"],
        "additionalProperties": False
    }
    
    # Payload para la API de Responses
    payload = {
        "model": model,
        "input": f"""Ve al siguiente sitio web de expediente legal mexicano: {url_expediente}

Busca en la página la tabla que dice "Síntesis de los Acuerdos asociados al Asunto" 
y analiza la información para extraer TODA la información de cada acuerdo:

- No. (número del acuerdo)
- Fecha del Auto (en formato DD-MM-YYYY)
- Tipo Cuaderno (ejemplo: Principal)
- Fecha de publicación (en formato DD-MM-YYYY)
- Resumen (el texto completo del resumen)

IMPORTANTE:
1. Extrae TODOS los acuerdos de la tabla
2. Ordénalos del más reciente al más antiguo según la "Fecha del Auto"
3. Retorna SOLO los 3 acuerdos más recientes
4. Para cada acuerdo, compara la "Fecha del Auto" con la fecha de hoy ({fecha_hoy})
5. Si la fecha del auto coincide con hoy, marca es_hoy=true, sino es_hoy=false
6. Incluye el total de acuerdos que encontraste en la tabla

FECHA DE HOY: {fecha_hoy}
URL del expediente: {url_expediente}

Responde ÚNICAMENTE en formato JSON válido.""",
        "instructions": "Eres un asistente legal especializado en analizar expedientes judiciales mexicanos. Usa web search para ir al sitio web proporcionado y analizar la tabla de acuerdos. Responde ÚNICAMENTE en formato JSON válido.",
        "tools": [{"type": "web_search"}],
        "text": {
            "format": {
                "name": "expediente_response",
                "type": "json_schema",
                "schema": schema,
                "strict": True
            }
        }
    }
    
    # Nota: reasoning.effort no se puede usar con web_search
    # Por lo tanto no agregamos reasoning cuando usamos web_search
    
    try:
        print(f"✅ Usando {model} con web search para: {url_expediente[:60]}...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Extraer el contenido de la respuesta
                    content = None
                    if 'output' in result and len(result['output']) > 0:
                        for output_item in result['output']:
                            if output_item.get('type') == 'message' and 'content' in output_item:
                                for content_item in output_item['content']:
                                    if content_item.get('type') == 'output_text' and 'text' in content_item:
                                        content = content_item['text']
                                        break
                                if content:
                                    break
                    
                    if content:
                        # Parsear JSON
                        try:
                            data = json.loads(content)
                            return ResultadoExpediente(**data)
                        except json.JSONDecodeError:
                            # Intentar extraer JSON del texto
                            json_match = re.search(r'\{.*\}', content, re.DOTALL)
                            if json_match:
                                try:
                                    data = json.loads(json_match.group())
                                    return ResultadoExpediente(**data)
                                except:
                                    pass
                    
                    print(f"⚠️  No se pudo extraer contenido válido de la respuesta")
                else:
                    error_text = await response.text()
                    print(f"❌ Error HTTP {response.status}: {error_text}")
                
    except Exception as e:
        print(f"❌ Error procesando {url_expediente}: {str(e)}")
    
    # Retornar resultado de error
    return ResultadoExpediente(
        url_expediente=url_expediente,
        fecha_consulta=fecha_hoy,
        total_acuerdos=0,
        acuerdos_recientes=[]
    )


# ============================================================================
# PROCESAMIENTO CONCURRENTE
# ============================================================================

async def procesar_expedientes_concurrente(
    urls: List[str], 
    api_key_manager: APIKeyManager,
    usar_gpt5: bool = True,
    max_concurrencia: int = 5
) -> List[ResultadoExpediente]:
    """
    Procesa múltiples expedientes de forma concurrente usando asyncio
    
    Args:
        urls: Lista de URLs de expedientes a procesar
        api_key_manager: Gestor de API keys
        usar_gpt5: Si True, intenta usar GPT-5
        max_concurrencia: Número máximo de tareas concurrentes (default: 5)
    
    Returns:
        Lista de ResultadoExpediente
    """
    
    print(f"\n🚀 Iniciando procesamiento asíncrono de {len(urls)} expedientes")
    print(f"⚙️  Configuración: max_concurrencia={max_concurrencia}, modelo={'GPT-5' if usar_gpt5 else 'GPT-4o'}")
    print("=" * 80)
    
    # Crear semáforo para limitar la concurrencia
    semaphore = asyncio.Semaphore(max_concurrencia)
    
    async def procesar_con_semaforo(url: str, idx: int):
        """Procesa un expediente con control de concurrencia"""
        async with semaphore:
            try:
                api_key = await api_key_manager.get_next_key()
                resultado = await verificar_expediente_con_api(url, api_key, usar_gpt5)
                print(f"✅ [{idx}/{len(urls)}] Completado: {url[:70]}...")
                print(f"   └─ Total acuerdos: {resultado.total_acuerdos}, Recientes: {len(resultado.acuerdos_recientes)}")
                return resultado
            except Exception as e:
                print(f"❌ [{idx}/{len(urls)}] Error en {url}: {str(e)}")
                # Retornar resultado de error
                return ResultadoExpediente(
                    url_expediente=url,
                    fecha_consulta=datetime.now().strftime("%d-%m-%Y"),
                    total_acuerdos=0,
                    acuerdos_recientes=[]
                )
    
    # Crear tareas para todos los expedientes
    tareas = [
        procesar_con_semaforo(url, idx)
        for idx, url in enumerate(urls, 1)
    ]
    
    # Ejecutar todas las tareas concurrentemente
    resultados = await asyncio.gather(*tareas)
    
    print("=" * 80)
    print(f"✅ Procesamiento completado: {len(resultados)}/{len(urls)} procesados\n")
    
    return resultados


# ============================================================================
# GUARDAR RESULTADOS EN JSON
# ============================================================================

def guardar_resultados(resultados: List[ResultadoExpediente], archivo: str = "output.json") -> str:
    """
    Guarda los resultados en un archivo JSON
    
    Args:
        resultados: Lista de ResultadoExpediente
        archivo: Nombre del archivo (por defecto: "output.json")
    
    Returns:
        Nombre del archivo guardado
    """
    
    # Convertir a diccionarios
    datos = [r.model_dump() for r in resultados]
    
    # Guardar en archivo
    with open(archivo, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Resultados guardados en: {archivo}")
    return archivo


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def imprimir_resumen(resultados: List[ResultadoExpediente]):
    """Imprime un resumen de los resultados en consola"""
    
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 80)
    
    for idx, resultado in enumerate(resultados, 1):
        print(f"\n🔍 Expediente {idx}: {resultado.url_expediente[:70]}...")
        print(f"   📅 Fecha de consulta: {resultado.fecha_consulta}")
        print(f"   📄 Total de acuerdos: {resultado.total_acuerdos}")
        print(f"   📋 Acuerdos recientes mostrados: {len(resultado.acuerdos_recientes)}")
        
        # Verificar si hay actualizaciones hoy
        hay_actualizacion_hoy = any(acuerdo.es_hoy for acuerdo in resultado.acuerdos_recientes)
        
        if hay_actualizacion_hoy:
            print(f"   ✅ ¡HAY ACTUALIZACIÓN HOY!")
        else:
            print(f"   ❌ No hay actualización hoy")
        
        # Mostrar los 3 acuerdos más recientes
        if resultado.acuerdos_recientes:
            print(f"\n   📑 Últimos 3 acuerdos:")
            for i, acuerdo in enumerate(resultado.acuerdos_recientes, 1):
                print(f"      {i}. [{acuerdo.fecha_auto}] {acuerdo.tipo_cuaderno} - No. {acuerdo.numero}")
                print(f"         {'🟢 ES HOY' if acuerdo.es_hoy else '⚪ No es hoy'}")
                print(f"         Resumen: {acuerdo.resumen[:100]}...")
    
    print("\n" + "=" * 80)


# ============================================================================
# FUNCIONES PARA BÚSQUEDA EN CSV
# ============================================================================

def leer_csv_expedientes(url_csv: str) -> Dict[str, InfoExpedienteCSV]:
    """
    Lee el CSV de expedientes directamente desde URL usando pandas
    
    Args:
        url_csv: URL del CSV en GitHub
    
    Returns:
        Dict con Num Expediente (lowercase) como key e InfoExpedienteCSV como value
    """
    
    print(f"📥 Leyendo CSV desde: {url_csv[:80]}...")
    
    try:
        # Leer CSV directamente con pandas desde la URL
        df = pd.read_csv(url_csv)
        
        print(f"✅ CSV leído: {len(df)} expedientes encontrados")
        
        # Crear diccionario indexado por Num Expediente (lowercase)
        expedientes_dict = {}
        
        for _, row in df.iterrows():
            # Obtener valores, reemplazar NaN con strings vacíos
            num_exp = str(row.get('Num Expediente', '')).strip()
            
            if num_exp and num_exp.lower() != 'nan':
                expediente = InfoExpedienteCSV(
                    num_expediente=num_exp,
                    parte_actora=str(row.get('Parte Actora', '')).replace('nan', ''),
                    parte_demandada=str(row.get('Parte Demandada', '')).replace('nan', ''),
                    parte_notificada=str(row.get('Parte Notificada', '')).replace('nan', ''),
                    fecha_actuacion=str(row.get('Fecha de Actuación', '')).replace('nan', ''),
                    sintesis=str(row.get('Síntesis', '')).replace('nan', ''),
                    magistrado=str(row.get('Magistrado', '')).replace('nan', ''),
                    secretario=str(row.get('Secretario', '')).replace('nan', ''),
                    sala=str(row.get('Sala', '')).replace('nan', ''),
                    encontrado=True
                )
                
                # Usar lowercase para búsqueda case-insensitive
                expedientes_dict[num_exp.lower()] = expediente
        
        print(f"✅ Diccionario creado con {len(expedientes_dict)} expedientes")
        return expedientes_dict
                    
    except Exception as e:
        print(f"❌ Error leyendo CSV: {str(e)}")
        raise


async def buscar_expedientes_en_csv(
    ids_expedientes: List[str],
    url_csv: str
) -> List[InfoExpedienteCSV]:
    """
    Busca lista de expedientes en el CSV
    
    Args:
        ids_expedientes: Lista de IDs a buscar
        url_csv: URL del CSV
    
    Returns:
        Lista de InfoExpedienteCSV (encontrado=True si existe, False con campos vacíos si no)
    """
    
    print(f"\n🔍 Buscando {len(ids_expedientes)} expedientes en CSV...")
    
    # Leer CSV y crear diccionario
    expedientes_dict = leer_csv_expedientes(url_csv)
    
    # Buscar cada expediente
    resultados = []
    
    for id_expediente in ids_expedientes:
        id_lower = id_expediente.strip().lower()
        
        if id_lower in expedientes_dict:
            # Expediente encontrado
            expediente = expedientes_dict[id_lower]
            print(f"   ✅ Encontrado: {id_expediente}")
            resultados.append(expediente)
        else:
            # Expediente no encontrado - retornar con encontrado=False y campos vacíos
            print(f"   ❌ No encontrado: {id_expediente}")
            expediente_vacio = InfoExpedienteCSV(
                num_expediente=id_expediente,
                parte_actora="",
                parte_demandada="",
                parte_notificada="",
                fecha_actuacion="",
                sintesis="",
                magistrado="",
                secretario="",
                sala="",
                encontrado=False
            )
            resultados.append(expediente_vacio)
    
    print(f"✅ Búsqueda completada: {sum(1 for r in resultados if r.encontrado)}/{len(ids_expedientes)} encontrados\n")
    
    return resultados


# ============================================================================
# FUNCIÓN PRINCIPAL ASYNC
# ============================================================================

async def main():
    """Función principal asíncrona"""
    
    # Configuración de API Keys desde variables de entorno
    API_KEYS = [
        os.getenv('OPENAI_API_KEY_1'),
        os.getenv('OPENAI_API_KEY_2'),
        os.getenv('OPENAI_API_KEY_3')
    ]
    
    # Filtrar None en caso de que falte alguna key
    API_KEYS = [key for key in API_KEYS if key is not None]
    
    if not API_KEYS:
        raise ValueError("❌ No se encontraron API keys. Configura las variables de entorno OPENAI_API_KEY_1, OPENAI_API_KEY_2, OPENAI_API_KEY_3")
    
    print(f"✅ Cargadas {len(API_KEYS)} API key(s) desde variables de entorno")
    
    # URLs de expedientes a procesar
    urls_expedientes = [
        "https://www.dgej.cjf.gob.mx/siseinternet/reportes/vercaptura.aspx?tipoasunto=1&organismo=76&expediente=1207/2024&tipoprocedimiento=0",
        "https://www.dgej.cjf.gob.mx/siseinternet/reportes/vercaptura.aspx?tipoasunto=1&organismo=76&expediente=1207/2024&tipoprocedimiento=0",
        "https://www.dgej.cjf.gob.mx/siseinternet/reportes/vercaptura.aspx?tipoasunto=1&organismo=76&expediente=1207/2024&tipoprocedimiento=0",
    ]
    
    # Inicializar gestor de API keys
    api_manager = APIKeyManager(API_KEYS)
    
    # Configuración: True = GPT-5, False = GPT-4o
    USAR_GPT5 = False
    
    # Procesar expedientes de forma concurrente con asyncio
    resultados = await procesar_expedientes_concurrente(
        urls=urls_expedientes,
        api_key_manager=api_manager,
        usar_gpt5=USAR_GPT5,
        max_concurrencia=5
    )
    
    # Guardar resultados en JSON
    archivo_guardado = guardar_resultados(resultados)
    
    # Imprimir resumen en consola
    imprimir_resumen(resultados)
    
    print(f"\n✅ Proceso completado. Resultados guardados en: {archivo_guardado}")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Ejecutar la función principal asíncrona
    asyncio.run(main())
