from contextlib import asynccontextmanager
import threading
import faiss
import numpy as np
import warnings
import mysql.connector
from mysql.connector import Error
import logging
from fastapi import FastAPI,Query , HTTPException
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List, Optional
import meilisearch
import time
import requests

faiss_lock = threading.RLock()
productos_by_id = {}
productos = []
faiss_index = None
model = None
meili_client = None
meili_index = None

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
    'api_key': 'masterKey' 
}

class ResultadoProducto(BaseModel):
    id: int
    nombre: Optional[str]
    descripcion: Optional[str]
    variante_comb: Optional[str]
    id_padre: Optional[int]
    categoria: Optional[str]
    marca: Optional[str]
    
    similitud: float

class ResultadoBusqueda(BaseModel):
    query: str
    resultados: List[ResultadoProducto]
    
id_map = {}

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

# ---------- Helpers ----------
def build_text(producto: dict) -> str:
    nombre = producto.get('nombre', '') or ''
    descripcion = producto.get('descripcion', '') or ''
    tags = producto.get('tags', '') or ''
    variante_comb = producto.get('variante_comb', '') or ''
    return f"{nombre} {descripcion} {tags} {variante_comb}"

def fetch_product_from_db(product_id: int):
    """Lee un producto concreto desde MySQL (usa el SELECT apropiado)."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor(dictionary=True)
        query = """-- adapta aquí el SELECT para traer los mismos campos que usas
        SELECT 
            v.id, 
            v.id_padre, 
            p.nombre, 
            p.descripcion, 
            v.tags, 
            v.variante_comb, 
            v.descripcion_larga,
            c.slug AS slug_categoria, 
            m.slug AS slug_marca
        FROM tienda_catalogoproductos v
        LEFT JOIN tienda_catalogoproductopadre p ON v.id_padre = p.id
        LEFT JOIN tienda_categoriasproductos c ON c.id = p.id_categoria
        LEFT JOIN tienda_marcas m ON m.id = p.id_marca
        WHERE v.id = %s AND v.activo = '1';
        """
        cur.execute(query, (product_id,))
        return cur.fetchone()
    except Exception as e:
        logging.exception("Error fetch_product_from_db")
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            cur.close()
            conn.close()

def faiss_remove_id_safe(pid: int):
    """Intenta remover un id de FAISS; si falla, lanza excepción para manejar fallback."""
    try:
        sel = faiss.IDSelectorBatch(np.array([pid], dtype=np.int64))
        faiss_index.remove_ids(sel)
    except Exception as e:
        # remove_ids puede no estar soportado por ciertos wrappers; subir la excepción para fallback
        raise

def embed_text(text: str):
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    vec = np.asarray(vec, dtype=np.float32)
    vec = np.ascontiguousarray(vec)
    return vec

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
    global meili_client, meili_index
    
    if not wait_for_meilisearch(MEILI_CONFIG['url']):
        raise RuntimeError("❌ Meilisearch no está disponible.")

    meili_client = meilisearch.Client(MEILI_CONFIG['url'], MEILI_CONFIG['api_key'])
    meili_index = meili_client.index('productos')
    
    # Verif índ vacio
    stats = meili_index.get_stats()
    if stats["numberOfDocuments"] == 0:
        docs = [
            {
                "id": p["id"],
                "nombre": p.get("nombre",""),
                "descripcion": p.get("descripcion",""),
                "tags": p.get("tags",""),
                "variante_comb": p.get("variante_comb","")
            }
            for p in productos
        ]
            
        docs = []
        for producto in productos:
            docs.append({
                "nombre": producto.get("nombre", ""),
                "variante_comb": producto.get("variante_comb", "")
            })
        
        task = meili_index.add_documents(docs)
        meili_client.wait_for_task(task["taskUid"])
        logging.info("✅ Documentos de productos cargados en Meilisearch.")
        
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- Startup ----------
    #global model, faiss_index, productos_by_id, meili_client, meili_index, productos
    global model, faiss_index, productos_by_id, productos
    productos = obtener_productos_desde_mysql()
    if not productos:
        raise RuntimeError("No se encontraron productos")

    productos_by_id = {p['id']: p for p in productos}
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    ids = np.array(list(productos_by_id.keys()), dtype=np.int64)
    texts = [build_text(productos_by_id[int(pid)]) for pid in ids]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = np.ascontiguousarray(embeddings)

    dimension = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(dimension)
    new_index = faiss.IndexIDMap2(base_index)
    new_index.add_with_ids(embeddings, ids)

    with faiss_lock:
        faiss_index = new_index

    await setup_meilisearch()
    logging.info("✅ Startup completo: FAISS y Meili listos.")

    yield  # 👈 aquí FastAPI mantiene el contexto activo

    # ---------- Shutdown ----------
    logging.info("🛑 Apagando API, liberando recursos...")
    # aquí podrías cerrar conexiones si fuera necesario

app = FastAPI(lifespan=lifespan, title="API Búsqueda de Productos", version="2.0.0")
    
"""        
# ---------- Startup: construir índice FAISS con IndexIDMap2 ----------
@app.on_event("startup")
async def startup_event():
    global model, faiss_index, productos_by_id, meili_client, meili_index

    # 1) Cargar productos desde DB (misma función que ya tenías)
    productos = obtener_productos_desde_mysql()  # tu función actual
    if not productos:
        raise RuntimeError("No se encontraron productos")

    # crear mapa por id
    productos_by_id = {p['id']: p for p in productos}

    # 2) cargar modelo
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # 3) preparar embeddings en el orden de los ids
    ids = np.array(list(productos_by_id.keys()), dtype=np.int64)
    texts = [build_text(productos_by_id[int(pid)]) for pid in ids]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings = np.ascontiguousarray(embeddings)

    # 4) crear índice con mapping de ids (IndexIDMap2)
    dimension = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(dimension)               # inner product sobre vectores normalizados = cosine
    new_index = faiss.IndexIDMap2(base_index)               # index wrapper que permite IDs externos
    new_index.add_with_ids(embeddings, ids)

    # asignar atomically (aunque estamos en startup)
    with faiss_lock:
        faiss_index = new_index

    # 5) inicializar Meilisearch (tu setup)
    await setup_meilisearch()   # asumes función existente que crea meili_index
    logging.info("Startup completo: FAISS y Meili listos.")
"""
# ---------- Búsqueda (usa ids reales devueltos por FAISS) ----------
def buscar_faiss(query: str, threshold: float, top_k: int = 50):
    if faiss_index is None:
        return []
    qvec = embed_text(query)
    #k = min(top_k, max(1, faiss_index.ntotal))
    k = min(20, faiss_index.ntotal)
    with faiss_lock:
        D, I = faiss_index.search(qvec, k)
    resultados = []
    for score, pid in zip(D[0], I[0]):
        if int(pid) == -1:
            continue
        if score >= threshold:
            resultados.append({"id": int(pid), "score": float(score)})
    return resultados

# ---------- Endpoint para UPsert (create/update) ----------
@app.post("/productos/upsert/{product_id}")
async def upsert_product_endpoint(product_id: int):
    """
    Este endpoint: 1) obtiene el producto desde la BD, 2) genera embedding,
    3) reemplaza (remove+add) en FAISS de forma atómica, 4) actualiza Meili.
    Debes llamar a este endpoint desde tu servicio que modifica la BD (push).
    """
    product = fetch_product_from_db(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Producto no encontrado")

    text = build_text(product)
    vec = embed_text(text)   # shape (1, dim)

    with faiss_lock:
        # si ya existe, intentar remover primero (para evitar duplicados)
        if product_id in productos_by_id:
            try:
                faiss_remove_id_safe(product_id)
            except Exception:
                logging.warning("remove_ids no soportado o falló; se hará reindex parcial en background.")
                # fallback: reconstruir índice excluyendo el id (o full rebuild); aquí hacemos full rebuild sync
                # Nota: para grandes colecciones, haz un rebuild asíncrono/batchado en lugar de bloqueo largo.
                tmp_products = {k: v for k, v in productos_by_id.items() if k != product_id}
                # agregar el nuevo producto al tmp_products antes de reconstruir
                tmp_products[product_id] = product
                new_idx = rebuild_index_from_products(tmp_products)  # función abajo
                faiss_index = new_idx
                productos_by_id.clear()
                productos_by_id.update(tmp_products)
        # ahora agregar vector nuevo
        try:
            faiss_index.add_with_ids(np.ascontiguousarray(vec), np.array([product_id], dtype=np.int64))
        except Exception as e:
            logging.exception("Error al add_with_ids en FAISS")
            raise HTTPException(status_code=500, detail="Error indexando en FAISS")

        # actualizar mapa local
        productos_by_id[product_id] = product

    # 4) actualizar Meilisearch (async)
    doc = {
        "id": product_id,
        "nombre": product.get("nombre", ""),
        "descripcion": product.get("descripcion", ""),
        "tags": product.get("tags", ""),
        "variante_comb": product.get("variante_comb", "")
    }
    # dependiendo de la versión de meilisearch-client, puede ser awaitable
    task = meili_index.add_documents([doc])   # si tu cliente es async: task = await meili_index.add_documents([doc])
    try:
        # para cliente sync/async: intenta esperar la tarea si es posible
        if hasattr(meili_client, "wait_for_task"):
            # await si es coroutine; si no, la función puede ser sync
            maybe = meili_client.wait_for_task
            if hasattr(maybe, "__await__"):
                await meili_client.wait_for_task(task['taskUid'])
            else:
                meili_client.wait_for_task(task['taskUid'])
    except Exception:
        logging.warning("No fue posible esperar la tarea de Meili (o cliente sync/async mismatch).")
    return {"ok": True, "id": product_id}

# ---------- Endpoint para DELETE ----------
@app.delete("/productos/{product_id}")
async def delete_product_endpoint(product_id: int):
    # remove from FAISS
    with faiss_lock:
        if product_id not in productos_by_id:
            raise HTTPException(status_code=404, detail="Producto no en índice")
        try:
            faiss_remove_id_safe(product_id)
        except Exception as e:
            logging.warning("remove_ids no soportado -> fallback: rebuild sin el id")
            tmp_products = {k: v for k, v in productos_by_id.items() if k != product_id}
            new_idx = rebuild_index_from_products(tmp_products)
            faiss_index = new_idx
            productos_by_id.clear()
            productos_by_id.update(tmp_products)
        else:
            # si remove_ids OK, solo quitamos del mapa local
            productos_by_id.pop(product_id, None)

    # delete from Meilisearch
    try:
        task = meili_index.delete_document(product_id)  # o delete_documents([product_id])
        if hasattr(meili_client, "wait_for_task"):
            if hasattr(meili_client.wait_for_task, "__await__"):
                await meili_client.wait_for_task(task['taskUid'])
            else:
                meili_client.wait_for_task(task['taskUid'])
    except Exception as e:
        logging.warning("No se pudo borrar en Meili: %s", e)

    return {"ok": True, "deleted": product_id}

# ---------- Rebuild helper y endpoint ----------
def rebuild_index_from_products(products_map: dict):
    """Reconstruye un nuevo índice FAISS en memoria a partir del mapa de productos (no bloquea el índice actual)."""
    ids = np.array(list(products_map.keys()), dtype=np.int64)
    texts = [build_text(products_map[int(pid)]) for pid in ids]
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    emb = np.ascontiguousarray(emb)
    base = faiss.IndexFlatIP(emb.shape[1])
    new_idx = faiss.IndexIDMap2(base)
    new_idx.add_with_ids(emb, ids)
    return new_idx

@app.post("/reindex")
def reindex_full():
    """Reconstruye todo el índice leyendo desde la DB (útil si hay corrupción o cambios masivos)."""
    productos = obtener_productos_desde_mysql()
    if not productos:
        raise HTTPException(status_code=500, detail="No hay productos para reindexar")
    new_map = {p['id']: p for p in productos}
    new_idx = rebuild_index_from_products(new_map)
    # swap atómico
    with faiss_lock:
        global faiss_index, productos_by_id
        faiss_index = new_idx
        productos_by_id = new_map
    # opcional: persistir
    faiss.write_index(faiss_index, "faiss.index")
    return {"ok": True, "count": len(productos_by_id)}
def buscar_meilisearch(query: str):
    #resultados_meili = meili_index.search(query)['hits']
    hits = meili_index.search(query)["hits"]
    #return [{"id": hit['id'], "score": 0.5} for hit in resultados_meili]
    return [{"id": h["id"], "score": 0.5} for h in hits]
def buscar_hibrido(query: str, threshold: float):
    faiss_results = buscar_faiss(query, threshold)
    meili_results = buscar_meilisearch(query)

    final_results = {}
    
    for res in meili_results:
        final_results[res['id']] = {"id": res['id'], "score": res['score']}
        
    for res in faiss_results:
        producto_id = res['id']
        faiss_score = res['score']
        
        if producto_id in final_results:
            final_results[producto_id]['score'] = faiss_score + 0.2
        else:
            final_results[producto_id] = {"id": producto_id, "score": faiss_score}

    resultados_finales = list(final_results.values())
    resultados_finales.sort(key=lambda x: x['score'], reverse=True)
    
    return resultados_finales

def obtener_producto_por_id(producto_id: int):
    """Busca un producto por su ID en la lista global."""
    return next((p for p in productos if p["id"] == producto_id), None)

@app.get("/buscar", response_model=ResultadoBusqueda)
def endpoint_buscar(query: str = Query(..., description="Texto a buscar"), threshold: float = 0.45):
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
                "descripcion_larga": producto.get("descripcion_larga"),
                "id_padre": producto.get("id_padre"),
                "categoria": producto.get("slug_categoria"),
                "marca": producto.get("slug_marca"),
                
                "similitud": round(res['score'], 3)
            })

    return {"query": query, "resultados": data}