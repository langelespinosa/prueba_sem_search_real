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

app = FastAPI(title="API Búsqueda de Productos", version="3.0.0")

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

milvus_collection = None
productos_data = []
id_map = {}

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

def setup_milvus_collection(collection_name: str):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, "Colección de productos para búsqueda semántica")
    
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    collection = Collection(collection_name, schema)
    
    #crear ind
    index_params = {
        "index_type": "FLAT",
        "metric_type": "COSINE",
        "params": {}
    }
    collection.create_index(
        field_name="vector", 
        index_params=index_params
    )
    
    return collection

def insert_data_into_milvus(collection: Collection, data: List[dict]):
    
    entities = [
        [d["id"] for d in data],
        #[f"{d.get('nombre', '') or ''} {d.get('descripcion', '') or ''} {d.get('tags', '') or ''} {d.get('variante_comb', '') or ''}" for d in data],
        [f"{d.get('nombre del producto', '') or ''} {d.get('descripcion del producto', '') or ''} {d.get('paabras clave', '') or ''} {d.get('variantes del produco', '') or ''}" for d in data],
        [np.zeros(768).tolist() for _ in data]
    ]
    
    insert_result = collection.insert(entities)
    logging.info(f"✅ Se insertaron {insert_result.insert_count} entidades en Milvus.")
    collection.flush()
    collection.load()

def buscar_milvus_semantic(query: str):
    try:
        query_words = query.strip().lower().split()
        if not query_words:
            return []

        #construir expr compl
        expr = " && ".join([f"text like '{word}%'" for word in query_words])

        results = milvus_collection.query(
            expr=expr,
            output_fields=["id", "text"],
            limit=10,
        )
        print(results)

        formatted_results = []
        for hit in results:
            formatted_results.append({
                "id": hit.get("id"),
                "score": 0.5
            })
            
        return formatted_results
    except Exception as e:
        logging.error(f"❌ Error durante la búsqueda en Milvus: {e}")
        return []

@app.on_event("startup")
async def startup_event():
    global milvus_collection, productos_data, id_map
    
    productos_db = obtener_productos_desde_mysql()
    #print(productos_db)
    if not productos_db:
        raise RuntimeError("❌ No se encontraron productos en la base de datos.")

    productos_data = productos_db
    id_map = {producto["id"]: producto for producto in productos_data}

    try:
        connections.connect(**MILVUS_CONFIG)
        logging.info("✅ Conectado a Milvus.")
    except Exception as e:
        logging.error(f"❌ Error al conectar a Milvus: {e}")
        raise RuntimeError("❌ No se pudo conectar a Milvus.")

    milvus_collection = setup_milvus_collection("productos_collection")
    insert_data_into_milvus(milvus_collection, productos_data)
    
    #carg la coleccion
    milvus_collection.load()

@app.get("/buscar", response_model=ResultadoBusqueda)
def endpoint_buscar(query: str = Query(..., description="Texto a buscar")):
    if not milvus_collection:
        raise HTTPException(status_code=503, detail="La colección de Milvus no se ha cargado todavía.")
        
    resultados_milvus = buscar_milvus_semantic(query)
    
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
                "similitud": round(res['score'], 3)
            })

    return {"query": query, "resultados": data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)