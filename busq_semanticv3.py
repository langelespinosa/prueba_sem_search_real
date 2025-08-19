from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import meilisearch
import mysql.connector
import decimal
import uvicorn

app = FastAPI(title="API Búsqueda Semántica")

#cargar modelo de embedd
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

#con mysql
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="pass",
        database="fireclub_back_pub"
    )
    cursor = conn.cursor(dictionary=True)
    print("✅ Conexión exitosa a MySQL")
except Exception as e:
    print(f"❌ Error conectando a MySQL: {e}")
    exit(1)

cursor.execute("SELECT id, nombre, descripcion, activo FROM tienda_catalogoproductopadre WHERE activo='1'")
documentos_db = cursor.fetchall()
#print(documentos_db)
if not documentos_db:
    print("⚠ No hay productos en la base de datos.")
    exit(1)

#conex meilisearch
try:
    client = meilisearch.Client("http://localhost:7700", "masterKey")
    #index = client.index("autos")
    index = client.index("productos_info")

    index.update_typo_tolerance({
        'minWordSizeForTypos': {
            'oneTypo': 4,
            'twoTypos': 10
        }
    })

    client.health()
    print("✅ Conexión exitosa con Meilisearch")
except Exception as e:
    print("❌ Error conectando con Meilisearch:", e)
    exit(1)

#preproc con embed
documentos_con_embeddings = []
for doc in documentos_db:
    for key, value in doc.items():
        if isinstance(value, decimal.Decimal):
            doc[key] = float(value)

    #texto_busqueda = f"{doc['marca']} {doc['modelo']} {doc['descripcion']} {doc['categoria']}"
    texto_busqueda = f"{doc['id']} {doc['nombre']} {doc['descripcion']} "
    
    embedding = model.encode(texto_busqueda).tolist()

    doc_con_embedding = doc.copy()
    doc_con_embedding['_vectors'] = {'default': embedding}
    doc_con_embedding['texto_busqueda'] = texto_busqueda
    documentos_con_embeddings.append(doc_con_embedding)

#subir a meili
index.add_documents(documentos_con_embeddings)

# config semant
index.update_settings({
    'embedders': {
        'default': {
            'source': 'userProvided',
            'dimensions': 384
        }
    },
    'searchableAttributes': [
        #'marca', 'modelo', 'descripcion', 'categoria', 'texto_busqueda'
        'id', 'nombre', 'descripcion', 'categoria', 'texto_busqueda'
    ],
    'displayedAttributes': [
        'id', 'nombre', 'descripcion','marca', 'modelo', 'precio', 'año', 'categoria'
    ]
})

def busqueda_semantica(query: str, limit: int = 100):
    query_embedding = model.encode(query).tolist()
    return index.search(
        "",  
        {
            'vector': query_embedding,
            'limit': limit,
            'hybrid': {
                'embedder': 'default',  #embedder default porque ya se procesò el query embbedding
                'semanticRatio': 1.0    #100%->1 búsqueda semantica, 0.0 busqueda tradicional
            }
        }
    )

@app.get("/buscar/semantica")
def endpoint_semantica(q: str = Query(..., description="Consulta de búsqueda"), limit: int = 100):
    resultados = busqueda_semantica(q, limit)
    return {"resultados": resultados["hits"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
