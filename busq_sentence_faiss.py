from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
#import re
import warnings
import mysql.connector
from mysql.connector import Error
import uvicorn

warnings.filterwarnings("ignore", category=FutureWarning)

# config DB
DB_CONFIG = {
    'host': 'localhost',
    #'database': 'prueba_usuarios',
    'database': 'fireclub_back_pub',
    'user': 'root',
    'password': 'pass'

}

app = FastAPI(title="API Búsqueda Productos", version="1.0.0")

#load dat
def obtener_productos_desde_mysql():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            #query = "SELECT id, login, email, grupo, identificacion FROM usuarios"
            query = "SELECT id, nombre, descripcion FROM tienda_catalogoproductopadre"
            
            cursor.execute(query)
            productos = cursor.fetchall()
            return productos
    except Error as e:
        print(f"❌ Error al conectar con MySQL: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

print("🔄 Cargando productos desde MySQL...")
productos = obtener_productos_desde_mysql()
if not productos:
    raise RuntimeError("No se encontraron usuarios en la base de datos")

corpus = []
id_map = {}

for i, producto in enumerate(productos):
    #text = f"ID: {usuario['id']} Login: {usuario['login']} Email: {usuario['email']} Grupo: {usuario['grupo']} Identificacion: {usuario['identificacion']}"
    text = f"""ID producto: {producto['id']} nombre del producto: {producto['nombre']} descripcion del producto: {producto['descripcion']}"""
    
    corpus.append(text)
    id_map[i] = producto["id"]

print("🔄 Generando embeddings...")
#model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embeddings = model.encode(corpus, normalize_embeddings=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings, dtype=np.float32))
print("✅ Índice FAISS creado exitosamente")

def buscar(query, threshold=0.3):
    query_vec = model.encode([query], normalize_embeddings=True)
    total_productos = len(productos)
    D, I = index.search(np.array(query_vec, dtype=np.float32), total_productos)
    resultados = []
    for score, idx in zip(D[0], I[0]):
        if score >= threshold:
            resultados.append((id_map[idx], float(score)))
    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados

def buscar_hibrido(query, threshold=0.3):
    # Búsqueda semántica con la frase completa
    resultados = buscar(query, threshold)

    # Dividir en palabras clave
    palabras = query.lower().split()

    for producto in productos:
        match_exacto = False
        score_exacto = 1.0

        # Revisar coincidencia exacta de cualquier palabra en cualquier campo
        campos_busqueda = [
            producto['nombre'], producto['descripcion']
        ]

        for palabra in palabras:
            for campo in campos_busqueda:
                if palabra in campo.lower():
                    match_exacto = True
                    break
            if match_exacto:
                break  # Si ya coincidió una palabra, no seguir buscando

        if match_exacto:
            user_id = producto['id']
            if not any(result[0] == user_id for result in resultados):
                resultados.insert(0, (user_id, score_exacto))

    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados
"""
def buscar_hibrido(query, threshold=0.3):
    resultados = buscar(query, threshold)
    query_lower = query.lower()

    for usuario in usuarios:
        match_exacto = False
        score_exacto = 1.0
        
        #if (query_lower in usuario['login'].lower() or
        #    query_lower in usuario['email'].lower() or
        #    query_lower in usuario['grupo'].lower() or
        #    query_lower in usuario['identificacion'].lower()):
        #    match_exacto = True

        #campos_busqueda = [
        #    usuario['login'], usuario['email'],
        #    usuario['grupo'], usuario['identificacion']
        #]
        
        if (query_lower in usuario['login'].lower() or
            query_lower in usuario['email'].lower() or
            query_lower in usuario['maildir'].lower() or
            query_lower in usuario['identificacion'].lower() or
            query_lower in usuario['grupo'].lower()
            ):
            match_exacto = True

        campos_busqueda = [
            usuario['login'], usuario['email'], usuario['maildir'],
            usuario['grupo'], usuario['identificacion']
        ]
        
        for campo in campos_busqueda:
            if re.search(re.escape(query_lower), campo.lower()):
                match_exacto = True
                break

        if match_exacto:
            user_id = usuario['id']
            if not any(result[0] == user_id for result in resultados):
                resultados.insert(0, (user_id, score_exacto))

    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados
"""
def obtener_producto_por_id(producto_id):
    return next((u for u in productos if u["id"] == producto_id), None)

#endp
@app.get("/buscar")
def endpoint_buscar(query: str = Query(..., description="Texto a buscar"), threshold: float = 0.45):
    print("Aún no hubo Invalid HTTP request received.")
    try:
        resultados = buscar_hibrido(query, threshold)
        data = []
        for user_id, score in resultados:
            usuario = obtener_producto_por_id(user_id)
            if usuario:
                data.append({
                    "id": usuario["id"],
                    "nombre": usuario["nombre"],
                    "descripcion": usuario["descripcion"],
                    
                    "similitud": round(score, 3)
                })
        return JSONResponse(content={"query": query, "resultados": data})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    print("Antes de Invalid HTTP request received.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
