from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import meilisearch
import decimal
import uvicorn
import mysql.connector
import json
"""
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

cursor.execute("SELECT id, variantes, nombre, descripcion, activo FROM tienda_catalogoproductopadre WHERE activo='1'")
documentos_db = cursor.fetchall()

if not documentos_db:
    print("⚠ No hay productos en la base de datos.")
    exit(1)
"""

# Imprimir la salida esperada
#for p_doc in processed_documents:
#    print(f"{p_doc['id']} | {p_doc['variantes']} | {p_doc['nombre']} | {p_doc['descripcion']} | {p_doc['activo']}")

# Cerrar la conexión

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

cursor.execute("""
            SELECT    
                p.id, 
                p.id_categoria,    
                p.id_marca,
                p.variantes,    
                p.nombre,    
                p.descripcion,    
                p.activo,   
                
                v.modelo,
                v.precio,
                
                c.descripcion AS descripcion_categoria,   
                c.slug AS slug_categoria, 
                        
                m.slug AS slug_marca
                
                FROM 
                    tienda_catalogoproductopadre p
                LEFT JOIN tienda_catalogoproductos v 
                    ON v.id_padre = p.id       
                LEFT JOIN tienda_categoriasproductos c         
                    ON c.id = p.id_categoria   
                LEFT JOIN tienda_marcas m 
                    ON m.id = p.id_marca
                
            WHERE p.activo='1';""")

documentos_db = cursor.fetchall()

cursor.close()
conn.close()

#print(documentos_db)
if not documentos_db:
    print("⚠ No hay productos en la base de datos.")
    exit(1)

# Procesar los datos
processed_documents = []
for doc in documentos_db:
    variantes_str = doc['variantes']
    processed_variantes = ""

    if variantes_str and isinstance(variantes_str, str):
        try:
            # Asegurarse de que el string es un JSON válido
            if variantes_str == "0": # Manejar el caso de '0' como variante
                processed_variantes = ""
            else:
                variantes_list = json.loads(variantes_str)
                if isinstance(variantes_list, list) and variantes_list:
                    # Construir la cadena de variantes como en el ejemplo
                    parts = []
                    for item in variantes_list:
                        atributo_nombre = item.get('atributo', {}).get('nombre', '')
                        valor = item.get('valor', [])
                        if atributo_nombre and valor:
                            parts.append(f'"{atributo_nombre}":"{"\",\"".join(valor)}"')
                    processed_variantes = ",".join(parts)
                else:
                    processed_variantes = "" # Caso de JSON vacío '[]'
        except json.JSONDecodeError:
            # Manejar errores de decodificación JSON (ej. si no es un JSON válido)
            processed_variantes = ""
    else:
        processed_variantes = "" # Manejar casos donde variantes_str es None o no es un string

    processed_documents.append({
        'id': doc['id'],
        'variantes': processed_variantes,
        'nombre': doc['nombre'] if doc['nombre'] is not None else "",
        'descripcion': doc['descripcion'] if doc['descripcion'] is not None else "",
        'activo': doc['activo']
    })

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
    #texto_busqueda = f"{doc['id']} {doc['variantes']} {doc['nombre']} {doc['descripcion']} {doc['modelo']} {doc['precio']} {doc['descripcion_categoria']} {doc['slug_categoria']} {doc['slug_marca']}"
    texto_busqueda = f"{doc['nombre']} {doc['descripcion']}"
    
    embedding = model.encode(texto_busqueda).tolist()

    doc_con_embedding = doc.copy()
    doc_con_embedding['_vectors'] = {'default': embedding}
    doc_con_embedding['texto_busqueda'] = texto_busqueda
    documentos_con_embeddings.append(doc_con_embedding)
    print(texto_busqueda)
    
#subir a meili
index.add_documents(documentos_con_embeddings)

# config semant
index.update_settings(
    {
        'embedders': {
            'default': {
                'source': 'userProvided',
                'dimensions': 384
            }
        },
        'searchableAttributes': [
            #'marca', 'modelo', 'descripcion', 'categoria', 'texto_busqueda'
            #'id', 'variantes', 'nombre', 'descripcion','descripcion_categoria', 'slug_categoria','slug_marca', 'modelo', 'precio','texto_busqueda'
            'nombre', 'descripcion', 'texto_busqueda'
        
        ],
        'displayedAttributes': [
            'id', 'variantes', 'nombre', 'descripcion','descripcion_categoria', 'slug_categoria','slug_marca', 'modelo', 'precio', 'categoria'
        ]
    }
)

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
