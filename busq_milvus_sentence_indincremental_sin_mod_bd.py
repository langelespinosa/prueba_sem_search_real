import warnings
import mysql.connector
from mysql.connector import Error
import uvicorn
import logging
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime, timedelta
import asyncio
import threading
import time

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

app = FastAPI(title="API Búsqueda de Productos con Indexación Incremental", version="4.0.0")

# Modelos Pydantic
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

class ProductoRequest(BaseModel):
    id: int
    accion: str  # 'crear', 'actualizar', 'eliminar'
    
class SincronizacionResponse(BaseModel):
    productos_procesados: int
    productos_agregados: int
    productos_actualizados: int
    productos_eliminados: int
    tiempo_procesamiento: float

# Variables globales
milvus_collection = None
productos_data = []
id_map = {}
sentence_model = None
ultima_sincronizacion = None
sincronizacion_en_progreso = False

def inicializar_sentence_transformer():
    """Inicializa el modelo de Sentence Transformers"""
    global sentence_model
    try:
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        sentence_model = SentenceTransformer(model_name)
        
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
    
    if producto.get('nombre'):
        elementos.append(producto['nombre'])
    if producto.get('descripcion'):
        elementos.append(producto['descripcion'])
    if producto.get('tags'):
        elementos.append(producto['tags'])
    if producto.get('variante_comb'):
        elementos.append(producto['variante_comb'])
    if producto.get('slug_categoria'):
        elementos.append(f"categoría: {producto['slug_categoria']}")
    if producto.get('slug_marca'):
        elementos.append(f"marca: {producto['slug_marca']}")
    
    texto_completo = " ".join(elementos)
    texto_limpio = texto_completo.replace('\n', ' ').replace('\r', ' ')
    texto_limpio = ' '.join(texto_limpio.split())
    
    if len(texto_limpio) > 500:
        texto_limpio = texto_limpio[:497] + "..."
    
    return texto_limpio

def obtener_producto_por_id(producto_id: int):
    """Obtiene un producto específico por ID desde MySQL"""
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
        WHERE v.id = %s AND v.activo = '1';
        """
        cursor.execute(query, (producto_id,))
        producto = cursor.fetchone()
        return producto
    except Error as e:
        logging.error(f"❌ Error al obtener producto {producto_id}: {e}")
        return None
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def obtener_productos_actualizados(desde: datetime):
    """Obtiene productos que han sido modificados desde una fecha específica"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor(dictionary=True)
        
        # OPCIÓN 1: Con campos updated_at (si existen)
        if tiene_campo_updated_at():
            query = """
            SELECT
                v.id,
                v.id_padre,
                v.slug,
                v.tags,
                v.activo,
                v.descripcion_larga,
                v.updated_at,
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
            WHERE (v.updated_at > %s OR p.updated_at > %s) 
            ORDER BY v.updated_at DESC;
            """
            cursor.execute(query, (desde, desde))
        else:
            # OPCIÓN 2: Sin campos updated_at - usar tabla de eventos
            logging.info("⚠️ No se encontraron campos updated_at, usando sistema de eventos...")
            return obtener_productos_de_cola_eventos()
            
        productos = cursor.fetchall()
        return productos
    except Error as e:
        logging.error(f"❌ Error al obtener productos actualizados: {e}")
        return []
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def tiene_campo_updated_at():
    """Verifica si la tabla tiene campos updated_at"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        cursor.execute("SHOW COLUMNS FROM tienda_catalogoproductos LIKE 'updated_at'")
        return cursor.fetchone() is not None
    except:
        return False
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def obtener_productos_de_cola_eventos():
    """Obtiene productos de una cola de eventos (alternativa sin timestamps)"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor(dictionary=True)
        
        # Obtener eventos pendientes de una tabla de eventos
        cursor.execute("""
            SELECT DISTINCT producto_id, accion, fecha_evento
            FROM producto_eventos 
            WHERE procesado = 0 
            ORDER BY fecha_evento ASC
        """)
        
        eventos = cursor.fetchall()
        productos_actualizados = []
        
        for evento in eventos:
            producto = obtener_producto_por_id(evento['producto_id'])
            if producto:
                producto['evento_accion'] = evento['accion']
                productos_actualizados.append(producto)
            
            # Marcar evento como procesado
            cursor.execute("""
                UPDATE producto_eventos 
                SET procesado = 1 
                WHERE producto_id = %s AND fecha_evento = %s
            """, (evento['producto_id'], evento['fecha_evento']))
        
        connection.commit()
        return productos_actualizados
        
    except Error as e:
        logging.error(f"❌ Error al obtener eventos de productos: {e}")
        return []
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def obtener_productos_desde_mysql():
    """Obtiene todos los productos desde MySQL (carga inicial)"""
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
    """Configura la colección de Milvus"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]
    schema = CollectionSchema(fields, "Colección de productos para búsqueda semántica")
    
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    collection = Collection(collection_name, schema)
    
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    
    return collection

def agregar_producto_a_milvus(producto: dict):
    """Agrega un producto individual a Milvus"""
    try:
        texto = procesar_texto_producto(producto)
        embedding = sentence_model.encode([texto])
        
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        entities = [
            [producto["id"]],
            [texto],
            embedding.tolist()
        ]
        
        insert_result = milvus_collection.insert(entities)
        milvus_collection.flush()
        
        # Actualizar datos locales
        global productos_data, id_map
        if producto not in productos_data:
            productos_data.append(producto)
        id_map[producto["id"]] = producto
        
        logging.info(f"✅ Producto {producto['id']} agregado a Milvus")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error al agregar producto {producto['id']} a Milvus: {e}")
        return False

def actualizar_producto_en_milvus(producto: dict):
    """Actualiza un producto en Milvus"""
    try:
        # Primero eliminar el producto existente
        eliminar_producto_de_milvus(producto["id"])
        
        # Luego agregarlo con la nueva información
        return agregar_producto_a_milvus(producto)
        
    except Exception as e:
        logging.error(f"❌ Error al actualizar producto {producto['id']} en Milvus: {e}")
        return False

def eliminar_producto_de_milvus(producto_id: int):
    """Elimina un producto de Milvus"""
    try:
        expr = f"id == {producto_id}"
        milvus_collection.delete(expr)
        milvus_collection.flush()
        
        # Actualizar datos locales
        global productos_data, id_map
        productos_data = [p for p in productos_data if p["id"] != producto_id]
        if producto_id in id_map:
            del id_map[producto_id]
        
        logging.info(f"✅ Producto {producto_id} eliminado de Milvus")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error al eliminar producto {producto_id} de Milvus: {e}")
        return False

def generar_embeddings_batch(textos: List[str], batch_size: int = 32):
    """Genera embeddings en lotes"""
    embeddings = []
    
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i + batch_size]
        batch_embeddings = sentence_model.encode(
            batch, 
            convert_to_tensor=True,
            show_progress_bar=True if i == 0 else False
        )
        
        if isinstance(batch_embeddings, torch.Tensor):
            batch_embeddings = batch_embeddings.cpu().numpy()
            
        embeddings.extend(batch_embeddings.tolist())
        
        if i % (batch_size * 10) == 0:
            logging.info(f"✅ Procesados {min(i + batch_size, len(textos))}/{len(textos)} embeddings")
    
    return embeddings

def insert_data_into_milvus(collection: Collection, data: List[dict]):
    """Inserta datos masivos en Milvus (solo para carga inicial)"""
    logging.info("🔄 Generando embeddings para productos...")
    
    textos_productos = []
    for producto in data:
        texto = procesar_texto_producto(producto)
        textos_productos.append(texto)
    
    embeddings = generar_embeddings_batch(textos_productos)
    
    entities = [
        [d["id"] for d in data],
        textos_productos,
        embeddings
    ]
    
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

def sincronizar_productos_incremental():
    """Sincroniza productos de forma incremental"""
    global ultima_sincronizacion, sincronizacion_en_progreso
    
    if sincronizacion_en_progreso:
        logging.info("⏳ Sincronización ya en progreso, saltando...")
        return {"error": "Sincronización ya en progreso"}
    
    sincronizacion_en_progreso = True
    inicio = time.time()
    
    try:
        # Si es la primera sincronización, usar fecha muy antigua
        desde = ultima_sincronizacion or datetime.now() - timedelta(days=365)
        
        productos_actualizados = obtener_productos_actualizados(desde)
        
        agregados = 0
        actualizados = 0
        eliminados = 0
        
        for producto in productos_actualizados:
            if producto["activo"] == "0":
                # Producto desactivado, eliminarlo del índice
                if eliminar_producto_de_milvus(producto["id"]):
                    eliminados += 1
            else:
                # Producto activo, agregarlo o actualizarlo
                if producto["id"] in id_map:
                    if actualizar_producto_en_milvus(producto):
                        actualizados += 1
                else:
                    if agregar_producto_a_milvus(producto):
                        agregados += 1
        
        ultima_sincronizacion = datetime.now()
        tiempo_procesamiento = time.time() - inicio
        
        resultado = {
            "productos_procesados": len(productos_actualizados),
            "productos_agregados": agregados,
            "productos_actualizados": actualizados,
            "productos_eliminados": eliminados,
            "tiempo_procesamiento": round(tiempo_procesamiento, 2)
        }
        
        logging.info(f"✅ Sincronización completada: {resultado}")
        return resultado
        
    except Exception as e:
        logging.error(f"❌ Error durante la sincronización incremental: {e}")
        return {"error": str(e)}
    finally:
        sincronizacion_en_progreso = False

def buscar_milvus_semantic(query: str, top_k: int = 10):
    """Realiza búsqueda semántica usando embeddings"""
    try:
        query_embedding = sentence_model.encode([query])
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = milvus_collection.search(
            data=query_embedding.tolist(),
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["id", "text"]
        )
        
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

# Tarea en background para sincronización automática
def sincronizacion_automatica():
    """Ejecuta sincronización automática cada 5 minutos"""
    while True:
        try:
            time.sleep(300)  # 5 minutos
            if milvus_collection is not None:
                sincronizar_productos_incremental()
        except Exception as e:
            logging.error(f"❌ Error en sincronización automática: {e}")

# Inicializar hilo de sincronización automática
def iniciar_sincronizacion_automatica():
    thread = threading.Thread(target=sincronizacion_automatica, daemon=True)
    thread.start()
    logging.info("🔄 Sincronización automática iniciada (cada 5 minutos)")

@app.on_event("startup")
async def startup_event():
    global milvus_collection, productos_data, id_map, ultima_sincronizacion
    
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
    
    ultima_sincronizacion = datetime.now()
    
    # Iniciar sincronización automática
    iniciar_sincronizacion_automatica()
    
    logging.info("🚀 Sistema de búsqueda semántica inicializado correctamente.")

# Endpoints existentes
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

# Nuevos endpoints para manejo incremental
@app.post("/productos/manejar")
def manejar_producto(request: ProductoRequest):
    """Maneja operaciones individuales de productos (crear, actualizar, eliminar)"""
    if not milvus_collection:
        raise HTTPException(status_code=503, detail="La colección de Milvus no se ha cargado todavía.")
    
    if request.accion == "eliminar":
        if eliminar_producto_de_milvus(request.id):
            return {"mensaje": f"Producto {request.id} eliminado exitosamente"}
        else:
            raise HTTPException(status_code=500, detail="Error al eliminar producto")
    
    elif request.accion in ["crear", "actualizar"]:
        producto = obtener_producto_por_id(request.id)
        if not producto:
            raise HTTPException(status_code=404, detail="Producto no encontrado en la base de datos")
        
        if request.accion == "crear":
            if agregar_producto_a_milvus(producto):
                return {"mensaje": f"Producto {request.id} creado exitosamente"}
        else:  # actualizar
            if actualizar_producto_en_milvus(producto):
                return {"mensaje": f"Producto {request.id} actualizado exitosamente"}
        
        raise HTTPException(status_code=500, detail=f"Error al {request.accion} producto")
    
    else:
        raise HTTPException(status_code=400, detail="Acción no válida. Use: crear, actualizar, eliminar")

@app.post("/sincronizar", response_model=SincronizacionResponse)
def sincronizar_manual():
    """Ejecuta una sincronización manual de productos"""
    if not milvus_collection:
        raise HTTPException(status_code=503, detail="La colección de Milvus no se ha cargado todavía.")
    
    resultado = sincronizar_productos_incremental()
    
    if "error" in resultado:
        raise HTTPException(status_code=500, detail=resultado["error"])
    
    return resultado

@app.get("/health")
def health_check():
    """Endpoint para verificar el estado del sistema"""
    return {
        "status": "ok",
        "milvus_connected": milvus_collection is not None,
        "sentence_model_loaded": sentence_model is not None,
        "productos_cargados": len(productos_data),
        "ultima_sincronizacion": ultima_sincronizacion.isoformat() if ultima_sincronizacion else None,
        "sincronizacion_en_progreso": sincronizacion_en_progreso
    }

@app.get("/estadisticas")
def obtener_estadisticas():
    """Obtiene estadísticas del sistema de búsqueda"""
    try:
        # Obtener estadísticas de la colección de Milvus
        num_entities = milvus_collection.num_entities
        
        return {
            "productos_en_memoria": len(productos_data),
            "productos_en_milvus": num_entities,
            "ultima_sincronizacion": ultima_sincronizacion.isoformat() if ultima_sincronizacion else None,
            "sincronizacion_automatica_activa": True,
            "intervalo_sincronizacion_minutos": 5
        }
    except Exception as e:
        return {"error": f"Error al obtener estadísticas: {e}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)