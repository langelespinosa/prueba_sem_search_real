import warnings
import mysql.connector
from mysql.connector import Error
import uvicorn
import logging
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuración de la Base de Datos
DB_CONFIG = {
    'host': 'localhost',
    'database': 'fireclub_back_pub',
    'user': 'root',
    'password': 'pass'
}

# Configuración de Milvus
MILVUS_CONFIG = {
    'alias': 'default',
    'host': 'localhost',
    'port': '19530'
}

app = FastAPI(title="API Búsqueda de Productos", version="3.1.0")

class ResultadoProducto(BaseModel):
    id: int
    nombre: Optional[str]
    descripcion: Optional[str]
    variantes_comb: Optional[str]
    id_padre: Optional[int]
    categoria: Optional[str]
    marca: Optional[str]
    similitud: float

class ResultadoBusqueda(BaseModel):
    query: str
    resultados: List[ResultadoProducto]

# Variables globales
milvus_collection = None
productos_data = []
id_map = {}
sentence_model = None

def inicializar_sentence_transformer():
    """Inicializa el modelo de Sentence Transformers"""
    global sentence_model
    try:
        # Modelo multilingüe que funciona bien para español
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        
        # Si quieres un modelo específico para español, puedes usar:
        # model_name = "sentence-transformers/distiluse-base-multilingual-cased"
        
        sentence_model = SentenceTransformer(model_name)
        
        # Verificar dimensiones del modelo
        test_embedding = sentence_model.encode("test")
        logging.info(f"✅ Modelo Sentence Transformer cargado: {model_name}")
        logging.info(f"✅ Dimensiones del embedding: {len(test_embedding)}")
        
        return len(test_embedding)
    except Exception as e:
        logging.error(f"❌ Error al cargar Sentence Transformer: {e}")
        raise RuntimeError("❌ No se pudo cargar el modelo de embeddings.")

def procesar_texto_producto(producto):
    """Procesa la información del producto para crear un texto descriptivo"""
    elementos = []
    
    # Agregar nombre si existe
    if producto.get('nombre'):
        elementos.append(producto['nombre'])
    
    # Agregar descripción si existe
    if producto.get('descripcion'):
        elementos.append(producto['descripcion'])
    
    # Agregar tags si existen
    if producto.get('tags'):
        elementos.append(producto['tags'])
    
    # Agregar variantes si existen
    if producto.get('variante_comb'):
        elementos.append(producto['variante_comb'])
    
    # Agregar categoría y marca si existen
    if producto.get('slug_categoria'):
        elementos.append(f"categoría: {producto['slug_categoria']}")
    
    if producto.get('slug_marca'):
        elementos.append(f"marca: {producto['slug_marca']}")
    
    # Unir todo el texto
    texto_completo = " ".join(elementos)
    
    # Limpiar y limitar el texto
    texto_limpio = texto_completo.replace('\n', ' ').replace('\r', ' ')
    texto_limpio = ' '.join(texto_limpio.split())  # Eliminar espacios múltiples
    
    # Limitar a 500 caracteres para evitar textos muy largos
    if len(texto_limpio) > 500:
        texto_limpio = texto_limpio[:497] + "..."
    
    return texto_limpio

def obtener_productos_desde_mysql():
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
            v.descripcion_larga,
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

def setup_milvus_collection(collection_name: str, embedding_dim: int):
    """Configura la colección de Milvus con las dimensiones correctas"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]
    schema = CollectionSchema(fields, "Colección de productos para búsqueda semántica")
    
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    collection = Collection(collection_name, schema)
    
    # Crear índice optimizado para búsqueda semántica
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    collection.create_index(
        field_name="vector", 
        index_params=index_params
    )
    
    return collection

def generar_embeddings_batch(textos: List[str], batch_size: int = 32):
    """Genera embeddings en lotes para optimizar el rendimiento"""
    embeddings = []
    
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i + batch_size]
        batch_embeddings = sentence_model.encode(
            batch, 
            convert_to_tensor=True,
            show_progress_bar=True if i == 0 else False
        )
        
        # Convertir a CPU y numpy si es necesario
        if isinstance(batch_embeddings, torch.Tensor):
            batch_embeddings = batch_embeddings.cpu().numpy()
            
        embeddings.extend(batch_embeddings.tolist())
        
        if i % (batch_size * 10) == 0:
            logging.info(f"✅ Procesados {min(i + batch_size, len(textos))}/{len(textos)} embeddings")
    
    return embeddings

def insert_data_into_milvus(collection: Collection, data: List[dict]):
    """Inserta datos con embeddings reales en Milvus"""
    logging.info("🔄 Generando embeddings para productos...")
    
    # Procesar textos de productos
    textos_productos = []
    for producto in data:
        texto = procesar_texto_producto(producto)
        textos_productos.append(texto)
    
    # Generar embeddings
    embeddings = generar_embeddings_batch(textos_productos)
    
    # Preparar entidades para insertar
    entities = [
        [d["id"] for d in data],
        textos_productos,
        embeddings
    ]
    
    # Insertar en lotes para evitar problemas de memoria
    batch_size = 100
    total_insertados = 0
    
    for i in range(0, len(data), batch_size):
        batch_entities = [
            entities[0][i:i + batch_size],
            entities[1][i:i + batch_size],
            entities[2][i:i + batch_size]
        ]
        
        insert_result = collection.insert(batch_entities)
        total_insertados += insert_result.insert_count
        
        logging.info(f"✅ Insertado lote {i//batch_size + 1}: {insert_result.insert_count} entidades")
    
    logging.info(f"✅ Total insertadas: {total_insertados} entidades en Milvus.")
    collection.flush()
    collection.load()

def buscar_milvus_semantic(query: str, top_k: int = 10):
    """Realiza búsqueda semántica usando embeddings"""
    try:
        # Generar embedding para la consulta
        query_embedding = sentence_model.encode([query])
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        # Realizar búsqueda vectorial
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = milvus_collection.search(
            data=query_embedding.tolist(),
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["id", "text"]
        )
        
        # Formatear resultados
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "score": float(hit.score),
                    "text": hit.entity.get("text", "")
                })
        
        logging.info(f"✅ Búsqueda semántica completada: {len(formatted_results)} resultados")
        return formatted_results
        
    except Exception as e:
        logging.error(f"❌ Error durante la búsqueda semántica en Milvus: {e}")
        return []

@app.on_event("startup")
async def startup_event():
    global milvus_collection, productos_data, id_map
    
    # Inicializar Sentence Transformer
    embedding_dim = inicializar_sentence_transformer()
    
    # Cargar productos desde MySQL
    productos_db = obtener_productos_desde_mysql()
    if not productos_db:
        raise RuntimeError("❌ No se encontraron productos en la base de datos.")

    productos_data = productos_db
    id_map = {producto["id"]: producto for producto in productos_data}

    # Conectar a Milvus
    try:
        connections.connect(**MILVUS_CONFIG)
        logging.info("✅ Conectado a Milvus.")
    except Exception as e:
        logging.error(f"❌ Error al conectar a Milvus: {e}")
        raise RuntimeError("❌ No se pudo conectar a Milvus.")

    # Configurar colección y cargar datos
    milvus_collection = setup_milvus_collection("productos_collection", embedding_dim)
    insert_data_into_milvus(milvus_collection, productos_data)
    
    logging.info("🚀 Sistema de búsqueda semántica inicializado correctamente.")

@app.get("/buscar", response_model=ResultadoBusqueda)
def endpoint_buscar(
    query: str = Query(..., description="Texto a buscar"),
    limite: int = Query(10, ge=1, le=50, description="Número máximo de resultados")
):
    if not milvus_collection:
        raise HTTPException(status_code=503, detail="La colección de Milvus no se ha cargado todavía.")
        
    if not sentence_model:
        raise HTTPException(status_code=503, detail="El modelo de embeddings no se ha cargado todavía.")
        
    resultados_milvus = buscar_milvus_semantic(query, top_k=limite)
    
    data = []
    for res in resultados_milvus:
        producto = id_map.get(res["id"])
        if producto:
            data.append({
                "id": producto["id"],
                "nombre": producto.get("nombre"),
                "descripcion": producto.get("descripcion"),
                "variantes_comb": producto.get("variante_comb"),
                "id_padre": producto.get("id_padre"),
                "categoria": producto.get("slug_categoria"),
                "marca": producto.get("slug_marca"),
                "similitud": round(res['score'], 4)
            })

    return {"query": query, "resultados": data}

@app.get("/health")
def health_check():
    """Endpoint para verificar el estado del sistema"""
    return {
        "status": "ok",
        "milvus_connected": milvus_collection is not None,
        "sentence_model_loaded": sentence_model is not None,
        "productos_cargados": len(productos_data)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)