import faiss
import numpy as np
import warnings
import mysql.connector
from mysql.connector import Error
import uvicorn
import re
import meilisearch
import requests
import time
import logging

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional

# --- Configuración y Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning)

DB_CONFIG = {
    'host': 'localhost',
    'database': 'fireclub_back_pub',
    'user': 'root',
    'password': 'pass'
}

MEILI_CONFIG = {
    'url': 'http://localhost:7700',
    'api_key': 'masterKey' # ¡ADVERTENCIA! Usar una API Key segura en producción.
}

app = FastAPI(title="API Búsqueda de Productos", version="2.0.0")

# --- Modelos Pydantic para la respuesta de la API ---
class ResultadoProducto(BaseModel):
    id: int
    nombre: Optional[str]
    descripcion: Optional[str]
    variantes_comb: Optional[str]
    similitud: float

class ResultadoBusqueda(BaseModel):
    query: str
    resultados: List[ResultadoProducto]

# --- Variables globales para recursos pesados ---
model = None
faiss_index = None
meili_client = None
meili_index = None
productos = []
id_map = {}

# --- Funciones de Utilidad ---
def obtener_productos_desde_mysql():
    """Conecta a MySQL y recupera los datos de los productos."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT
            v.id,
            v.id_padre,
            v.slug,
            v.tags,
            v.activo,
            CASE
                WHEN v.variante_comb IS NULL OR JSON_LENGTH(v.variante_comb) = 0 THEN NULL
                ELSE (
                    SELECT GROUP_CONCAT(
                            CASE
                                WHEN JSON_TYPE(jt.atributo) = 'STRING'
                                THEN CONCAT(jt.atributo, ' : ', REPLACE(REPLACE(REPLACE(jt.valor_limpio, '["', ''), '"]', ''), '","', ', '))
                                WHEN JSON_TYPE(jt.atributo) = 'OBJECT'
                                THEN CONCAT(JSON_UNQUOTE(JSON_EXTRACT(jt.atributo, '$.nombre')), ' : ', REPLACE(REPLACE(REPLACE(jt.valor_limpio, '["', ''), '"]', ''), '","', ', '))
                                ELSE NULL
                            END
                            SEPARATOR ', '
                        )
                    FROM JSON_TABLE(
                        v.variante_comb,
                        '$[*]' COLUMNS (
                            atributo JSON PATH '$.atributo',
                            valor JSON PATH '$.valor',
                            valor_limpio TEXT PATH '$.valor'
                        )
                    ) jt
                )
            END AS variante_comb,
            p.nombre AS nombre,
            p.descripcion AS descripcion,
            c.slug AS slug_categoria,
            m.slug AS slug_marca
        FROM tienda_catalogoproductos v
        LEFT JOIN tienda_catalogoproductopadre p ON v.id_padre = p.id
        LEFT JOIN tienda_categoriasproductos c ON c.id = p.id_categoria
        LEFT JOIN tienda_marcas m ON m.id = p.id_marca
        WHERE v.activo = '1';
        """
        cursor.execute(query)
        productos = cursor.fetchall()
        logging.info(f"✅ Se cargaron {len(productos)} productos desde MySQL.")
        return productos
    except Error as e:
        logging.error(f"❌ Error al conectar con MySQL: {e}")
        return []
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def wait_for_meilisearch(url: str, max_attempts: int = 10, delay: int = 2):
    """Espera a que Meilisearch esté disponible antes de continuar."""
    for i in range(max_attempts):
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                logging.info("✅ Meilisearch está listo.")
                return True
        except requests.exceptions.RequestException:
            pass
        logging.warning(f"Meilisearch no está disponible. Reintentando en {delay}s...")
        time.sleep(delay)
    return False

async def setup_meilisearch():
    """
    Inicializa el cliente de Meilisearch y configura el índice.
    Si el índice no existe, lo crea y lo puebla con los datos de productos.
    """
    global meili_client, meili_index
    
    if not wait_for_meilisearch(MEILI_CONFIG['url']):
        raise RuntimeError("❌ Meilisearch no está disponible.")

    meili_client = meilisearch.Client(MEILI_CONFIG['url'], MEILI_CONFIG['api_key'])
    meili_index = meili_client.index('productos')
    
    # Verificar si el índice está vacío para poblarlo
    stats = meili_index.get_stats()
    if stats.numberOfDocuments == 0:
        logging.info("🔄 El índice de Meilisearch está vacío. Cargando documentos...")
        
        # Preparar los documentos para Meilisearch
        docs = []
        for producto in productos:
            docs.append({
                #"id": producto["id"],
                "nombre": producto.get("nombre", ""),
                #"descripcion": producto.get("descripcion", ""),
                #"tags": producto.get("tags", ""),
                "variante_comb": producto.get("variante_comb", "")
            })
        
        #Añadir documentos y esperar a que la tarea se complete
        task = await meili_index.add_documents(docs)
        await meili_client.wait_for_task(task['taskUid'])
        logging.info("✅ Documentos de productos cargados en Meilisearch.")
        
        #Configuración del índice para mejorar la búsqueda
        settings = {
            "searchableAttributes": ["nombre", "descripcion", "tags", "variante_comb"],
            "rankingRules": [
                "words",
                "typo",
                "proximity",
                "attribute",
                "sort",
                "exactness"
            ],
            "stopWords": ["el", "la", "de", "en", "y", "a"],
            "synonyms": {
                "polera": ["remera", "camiseta"],
                "pantalon": ["pantalón", "jeans"],
            },
        }
        task = await meili_index.update_settings(settings)
        await meili_client.wait_for_task(task['taskUid'])
        logging.info("✅ Configuración del índice de Meilisearch actualizada.")

# --- Evento de Inicio de la Aplicación ---
@app.on_event("startup")
async def startup_event():
    """
    Carga todos los recursos pesados al iniciar el servidor para
    evitar recargarlos en cada request.
    """
    global model, faiss_index, productos, id_map
    
    # Paso 1: Cargar datos desde la base de datos
    productos_db = obtener_productos_desde_mysql()
    if not productos_db:
        raise RuntimeError("❌ No se encontraron productos en la base de datos.")

    productos = productos_db
    id_map = {i: producto["id"] for i, producto in enumerate(productos)}

    # Paso 2: Configurar Meilisearch
    await setup_meilisearch()

    # Paso 3: Configurar el índice FAISS
    logging.info("🔄 Generando embeddings y creando índice FAISS...")
    corpus = []
    for producto in productos:
        nombre = producto.get('nombre', '') or ''
        descripcion = producto.get('descripcion', '') or ''
        tags = producto.get('tags', '') or ''
        variante_comb = producto.get('variante_comb', '') or ''
        text = f"{nombre} {descripcion} {tags} {variante_comb}"
        corpus.append(text)

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(corpus, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(np.array(embeddings, dtype=np.float32))
    logging.info("✅ Índice FAISS creado exitosamente.")

# --- Funciones de Búsqueda ---
def buscar_faiss(query: str, threshold: float):
    """Realiza la búsqueda semántica con FAISS."""
    query_vec = model.encode([query], normalize_embeddings=True)
    total_productos = len(productos)
    D, I = faiss_index.search(np.array(query_vec, dtype=np.float32), total_productos)
    
    resultados_faiss = []
    for score, idx in zip(D[0], I[0]):
        if score >= threshold:
            resultados_faiss.append({"id": id_map[idx], "score": float(score)})
    return resultados_faiss

def buscar_meilisearch(query: str):
    """Realiza la búsqueda por palabras clave con Meilisearch."""
    resultados_meili = meili_index.search(query)['hits']
    # Meilisearch no da un score de similitud, lo simulamos para la hibridación.
    return [{"id": hit['id'], "score": 0.5} for hit in resultados_meili]

def buscar_hibrido(query: str, threshold: float):
    """
    Combina los resultados de la búsqueda semántica (FAISS)
    y la búsqueda por palabras clave (Meilisearch) para un resultado
    más completo.
    """
    # Paso 1: Realizar ambas búsquedas
    faiss_results = buscar_faiss(query, threshold)
    meili_results = buscar_meilisearch(query)

    # Paso 2: Combinar y re-puntuar los resultados
    final_results = {}
    
    # Agregar resultados de Meilisearch
    for res in meili_results:
        final_results[res['id']] = {"id": res['id'], "score": res['score']}
        
    # Agregar o mejorar el score de los resultados de FAISS
    for res in faiss_results:
        producto_id = res['id']
        faiss_score = res['score']
        
        if producto_id in final_results:
            # Producto encontrado en ambos: Potenciamos el score.
            # Aquí combinamos el score de Faiss con un boost.
            # Se usa una fórmula simple: (score de Faiss + boost)
            final_results[producto_id]['score'] = faiss_score + 0.2
        else:
            # Producto solo en FAISS: Lo agregamos con su score original
            final_results[producto_id] = {"id": producto_id, "score": faiss_score}

    # Convertir el diccionario a una lista y ordenar por score
    resultados_finales = list(final_results.values())
    resultados_finales.sort(key=lambda x: x['score'], reverse=True)
    
    return resultados_finales

def obtener_producto_por_id(producto_id: int):
    """Busca un producto por su ID en la lista global."""
    return next((p for p in productos if p["id"] == producto_id), None)

# --- Endpoint de la API ---
@app.get("/buscar", response_model=ResultadoBusqueda)
def endpoint_buscar(query: str = Query(..., description="Texto a buscar"), threshold: float = 0.45):
    """
    Endpoint principal de búsqueda. Realiza una búsqueda híbrida combinando
    la similitud semántica (FAISS) y la relevancia por palabras clave (Meilisearch).
    """
    if not model or not faiss_index or not meili_client:
        raise HTTPException(status_code=503, detail="El modelo no se ha cargado todavía.")
        
    resultados = buscar_hibrido(query, threshold)
    
    data = []
    for res in resultados:
        producto = obtener_producto_por_id(res['id'])
        if producto:
            data.append({
                "id": producto["id"],
                "nombre": producto.get("nombre"),
                "descripcion": producto.get("descripcion"),
                "variantes_comb": producto.get("variante_comb"),
                "similitud": round(res['score'], 3)
            })

    return {"query": query, "resultados": data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
