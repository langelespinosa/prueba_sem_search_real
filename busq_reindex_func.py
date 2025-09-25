#faiss_search: Siempre está recibiendo consultas y llevando resultados a un frontend
#updater.py: Espera eventos(agregar,actualizar,eliminar) de una cola en go y realiza la actualización del archivo search_backup.pkl y faiss_index.bin 
#(estarán en un OSS o NAS)
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import warnings
import mysql.connector
import uvicorn
from mysql.connector import Error
from typing import Dict, List, Tuple, Optional
import threading
import pickle
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

DB_CONFIG = {
    'host': 'localhost',
    'database': 'fireclub_back_pub',
    'user': 'root',
    'password': 'pass'
}

app = FastAPI(title="API Búsqueda Productos", version="1.0.0")

class SemanticSearchManager:
    def __init__(self):
        #self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        self.index = None
        self.productos = {}  #acceso por ID
        self.corpus = {}     #ID a text
        self.id_to_faiss_idx = {}  #ID de producto a FAISS ind
        self.faiss_idx_to_id = {}  #índ FAISS a ID de producto
        self.dimension = 768  # Dimensión del modelo 
        self.lock = threading.RLock()  #para opr thread
        self.next_faiss_idx = 0
        
        self.index = faiss.IndexFlatIP(self.dimension)
        
        self._load_backup()
        
        if not self.productos:
            self._initial_load()
    
    def _get_db_connection(self):
        try:
            return mysql.connector.connect(**DB_CONFIG)
        except Error as e:
            print(f"❌ Error al conectar con MySQL: {e}")
            return None
    
    def _obtener_producto_desde_mysql(self, producto_id: int) -> Optional[Dict]:
        connection = self._get_db_connection()
        if not connection:
            return None
            
        try:
            cursor = connection.cursor(dictionary=True)
            query = """
            SELECT
                v.id,
                v.id_padre,
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
                p.descripcion AS descripcion
            FROM tienda_catalogoproductos v
            LEFT JOIN tienda_catalogoproductopadre p
                ON v.id_padre = p.id
            WHERE v.id = %s AND v.activo = '1';
            """
            cursor.execute(query, (producto_id,))
            producto = cursor.fetchone()
            return producto
            
        except Error as e:
            print(f"❌ Error al obtener producto {producto_id}: {e}")
            return None
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def _obtener_todos_productos_desde_mysql(self) -> List[Dict]:
        connection = self._get_db_connection()
        if not connection:
            return []
            
        try:
            cursor = connection.cursor(dictionary=True)
            query = """
            SELECT
                v.id,
                v.id_padre,
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
                p.descripcion AS descripcion
            FROM tienda_catalogoproductos v
            LEFT JOIN tienda_catalogoproductopadre p
                ON v.id_padre = p.id
            WHERE v.activo = '1';
            """
            cursor.execute(query)
            productos = cursor.fetchall()
            return productos
            
        except Error as e:
            print(f"❌ Error al obtener productos: {e}")
            return []
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
    
    def _crear_texto_producto(self, producto: Dict) -> str:
        nombre = producto.get('nombre', '') or ''
        descripcion = producto.get('descripcion', '') or ''
        tags = producto.get('tags', '') or ''
        variante_comb = producto.get('variante_comb', '') or ''
        
        return f"{nombre} {descripcion} {tags} {variante_comb}".strip()
    
    def _initial_load(self):
        print("🔄 Carga inicial desde MySQL...")
        productos_list = self._obtener_todos_productos_desde_mysql()
        
        if not productos_list:
            print("⚠️  No se encontraron productos en la base de datos")
            return
        
        with self.lock:
            self.productos.clear()
            self.corpus.clear()
            self.id_to_faiss_idx.clear()
            self.faiss_idx_to_id.clear()
            self.next_faiss_idx = 0
            
            self.index = faiss.IndexFlatIP(self.dimension)
            
            textos = []
            for producto in productos_list:
                producto_id = producto['id']
                texto = self._crear_texto_producto(producto)
                
                self.productos[producto_id] = producto
                self.corpus[producto_id] = texto
                self.id_to_faiss_idx[producto_id] = self.next_faiss_idx
                self.faiss_idx_to_id[self.next_faiss_idx] = producto_id
                
                textos.append(texto)
                self.next_faiss_idx += 1
            
            if textos:
                print("🔄 Generando embeddings...")
                embeddings = self.model.encode(textos, normalize_embeddings=True)
                self.index.add(np.array(embeddings, dtype=np.float32))
                print(f"✅ Índice FAISS creado con {len(textos)} productos")
                
                self._save_backup()
    
    def _save_backup(self):
        try:
            backup_data = {
                'productos': self.productos,
                'corpus': self.corpus,
                'id_to_faiss_idx': self.id_to_faiss_idx,
                'faiss_idx_to_id': self.faiss_idx_to_id,
                'next_faiss_idx': self.next_faiss_idx,
                'timestamp': datetime.now().isoformat()
            }
            
            with open('search_backup.pkl', 'wb') as f:
                pickle.dump(backup_data, f)
            
            # Guardar índice FAISS
            faiss.write_index(self.index, 'faiss_index.bin')
            print("💾 Backup guardado exitosamente")
            
        except Exception as e:
            print(f"❌ Error al guardar backup: {e}")
    
    def _load_backup(self):
        try:
            if os.path.exists('search_backup.pkl') and os.path.exists('faiss_index.bin'):
                with open('search_backup.pkl', 'rb') as f:
                    backup_data = pickle.load(f)
                
                self.productos = backup_data.get('productos', {})
                self.corpus = backup_data.get('corpus', {})
                self.id_to_faiss_idx = backup_data.get('id_to_faiss_idx', {})
                self.faiss_idx_to_id = backup_data.get('faiss_idx_to_id', {})
                self.next_faiss_idx = backup_data.get('next_faiss_idx', 0)
                
                self.index = faiss.read_index('faiss_index.bin')
                
                timestamp = backup_data.get('timestamp', 'desconocido')
                print(f"✅ Backup cargado exitosamente (creado: {timestamp})")
                print(f"📊 {len(self.productos)} productos en memoria")
                
        except Exception as e:
            print(f"⚠️  No se pudo cargar backup: {e}")
    
    def agregar_producto(self, producto_id: int) -> bool:
        #Add nuevo producto al índice
        try:
            producto = self._obtener_producto_desde_mysql(producto_id)
            if not producto:
                print(f"❌ No se encontró el producto {producto_id}")
                return False
            
            with self.lock:
                #act sì ya exist product
                if producto_id in self.productos:
                    return self.actualizar_producto(producto_id)
                
                #crea text y embed
                texto = self._crear_texto_producto(producto)
                embedding = self.model.encode([texto], normalize_embeddings=True)
                
                #add índice FAISS
                self.index.add(np.array(embedding, dtype=np.float32))
                
                #act map
                faiss_idx = self.next_faiss_idx
                self.productos[producto_id] = producto
                self.corpus[producto_id] = texto
                self.id_to_faiss_idx[producto_id] = faiss_idx
                self.faiss_idx_to_id[faiss_idx] = producto_id
                self.next_faiss_idx += 1
                
                print(f"✅ Producto {producto_id} agregado exitosamente")
                
                self._save_backup()
                return True
                
        except Exception as e:
            print(f"❌ Error al agregar producto {producto_id}: {e}")
            return False
    
    def actualizar_producto(self, producto_id: int) -> bool:
        try:
            if producto_id not in self.productos:
                #Add si no exist
                return self.agregar_producto(producto_id)
            
            producto = self._obtener_producto_desde_mysql(producto_id)
            if not producto:
                #si no esta en BD, borrar del índ
                return self.eliminar_producto(producto_id)
            
            with self.lock:
                #obtn índice FAISS producto
                faiss_idx = self.id_to_faiss_idx[producto_id]
                
                # text y embed
                nuevo_texto = self._crear_texto_producto(producto)
                nuevo_embedding = self.model.encode([nuevo_texto], normalize_embeddings=True)
                
                
                #reconst lazy(no reconstruir todo)
                self.productos[producto_id] = producto
                self.corpus[producto_id] = nuevo_texto
                
                #crear nuevo índice solo con los embeddings actualizados
                all_embeddings = []
                productos_ordenados = []
                
                for fid in range(self.next_faiss_idx):
                    if fid in self.faiss_idx_to_id:
                        pid = self.faiss_idx_to_id[fid]
                        if pid in self.corpus:
                            if pid == producto_id:
                                #new embedding
                                all_embeddings.append(nuevo_embedding[0])
                            else:
                                #re-encodificar text exist
                                existing_embedding = self.model.encode([self.corpus[pid]], normalize_embeddings=True)
                                all_embeddings.append(existing_embedding[0])
                            productos_ordenados.append(pid)
                
                if all_embeddings:
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.index.add(np.array(all_embeddings, dtype=np.float32))
                
                print(f"✅ Producto {producto_id} actualizado exitosamente")
                
                self._save_backup()
                return True
                
        except Exception as e:
            print(f"❌ Error al actualizar producto {producto_id}: {e}")
            return False
    
    def eliminar_producto(self, producto_id: int) -> bool:
        try:
            if producto_id not in self.productos:
                print(f"⚠️  Producto {producto_id} no existe en el índice")
                return True
            
            with self.lock:
                faiss_idx = self.id_to_faiss_idx[producto_id]
                
                del self.productos[producto_id]
                del self.corpus[producto_id]
                del self.id_to_faiss_idx[producto_id]
                del self.faiss_idx_to_id[faiss_idx]
                
                all_embeddings = []
                new_id_to_faiss = {}
                new_faiss_to_id = {}
                new_idx = 0
                
                for old_idx in range(self.next_faiss_idx):
                    if old_idx in self.faiss_idx_to_id:
                        pid = self.faiss_idx_to_id[old_idx]
                        if pid in self.corpus:
                            embedding = self.model.encode([self.corpus[pid]], normalize_embeddings=True)
                            all_embeddings.append(embedding[0])
                            new_id_to_faiss[pid] = new_idx
                            new_faiss_to_id[new_idx] = pid
                            new_idx += 1
                
                self.id_to_faiss_idx = new_id_to_faiss
                self.faiss_idx_to_id = new_faiss_to_id
                self.next_faiss_idx = new_idx
                
                self.index = faiss.IndexFlatIP(self.dimension)
                if all_embeddings:
                    self.index.add(np.array(all_embeddings, dtype=np.float32))
                
                print(f"✅ Producto {producto_id} eliminado exitosamente")
                
                self._save_backup()
                return True
                
        except Exception as e:
            print(f"❌ Error al eliminar producto {producto_id}: {e}")
            return False
    
    def buscar(self, query: str, threshold: float = 0.3) -> List[Tuple[int, float]]:
        try:
            with self.lock:
                if self.index.ntotal == 0:
                    return []
                
                query_vec = self.model.encode([query], normalize_embeddings=True)
                total_productos = self.index.ntotal
                
                D, I = self.index.search(np.array(query_vec, dtype=np.float32), total_productos)
                
                resultados = []
                for score, faiss_idx in zip(D[0], I[0]):
                    if faiss_idx in self.faiss_idx_to_id and score >= threshold:
                        #producto_id = self.agregarfaiss_idx_to_id[faiss_idx]
                        producto_id = self.faiss_idx_to_id[faiss_idx]
                        resultados.append((producto_id, float(score)))
                
                resultados.sort(key=lambda x: x[1], reverse=True)
                return resultados
                
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return []
    
    def buscar_hibrido(self, query: str, threshold: float = 0.3) -> List[Tuple[int, float]]:
        resultados = self.buscar(query, threshold)
        query_lower = query.lower()
        
        with self.lock:
            for producto_id, producto in self.productos.items():
                match_exacto = False
                score_exacto = 1.0
                
                nombre = producto.get('nombre', '') or ''
                descripcion = producto.get('descripcion', '') or ''
                tags = producto.get('tags', '') or ''
                variante_comb = producto.get('variante_comb', '') or ''
                
                if (#query_lower in nombre.lower() or
                    #query_lower in descripcion.lower() or
                    #query_lower in tags.lower() or
                    query_lower in variante_comb.lower()):
                    match_exacto = True
                
                campos_busqueda = [nombre, descripcion, tags, variante_comb]
                for campo in campos_busqueda:
                    if re.search(re.escape(query_lower), campo.lower()):
                        match_exacto = True
                        break
                
                if match_exacto:
                    if not any(result[0] == producto_id for result in resultados):
                        resultados.insert(0, (producto_id, score_exacto))
            
            resultados.sort(key=lambda x: x[1], reverse=True)
            return resultados
    
    def obtener_producto_por_id(self, producto_id: int) -> Optional[Dict]:
        with self.lock:
            return self.productos.get(producto_id)
    
    def get_stats(self) -> Dict:
        with self.lock:
            return {
                "total_productos": len(self.productos),
                "faiss_total": self.index.ntotal,
                "next_faiss_idx": self.next_faiss_idx,
                "dimension": self.dimension
            }

search_manager = SemanticSearchManager()

@app.get("/buscar")
def endpoint_buscar(query: str = Query(..., description="Texto a buscar"), threshold: float = 0.45):
    try:
        resultados = search_manager.buscar_hibrido(query, threshold)
        
        data = []
        for producto_id, score in resultados:
            producto = search_manager.obtener_producto_por_id(producto_id)
            
            if producto:
                data.append({
                    "id": producto["id"],
                    "nombre": producto["nombre"],
                    "descripcion": producto["descripcion"],
                    "variantes_comb": producto["variante_comb"],
                    "similitud": round(score, 3)
                })
        
        return JSONResponse(content={"query": query, "resultados": data})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/productos/{producto_id}/agregar")
def agregar_producto(producto_id: int):
    try:
        resultado = search_manager.agregar_producto(producto_id)
        if resultado:
            return JSONResponse(content={"mensaje": f"Producto {producto_id} agregado exitosamente"})
        else:
            raise HTTPException(status_code=404, detail=f"No se pudo agregar el producto {producto_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/productos/{producto_id}/actualizar")
def actualizar_producto(producto_id: int):
    try:
        resultado = search_manager.actualizar_producto(producto_id)
        if resultado:
            return JSONResponse(content={"mensaje": f"Producto {producto_id} actualizado exitosamente"})
        else:
            raise HTTPException(status_code=404, detail=f"No se pudo actualizar el producto {producto_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/productos/{producto_id}/eliminar")
def eliminar_producto(producto_id: int):
    try:
        resultado = search_manager.eliminar_producto(producto_id)
        if resultado:
            return JSONResponse(content={"mensaje": f"Producto {producto_id} eliminado exitosamente"})
        else:
            raise HTTPException(status_code=404, detail=f"No se pudo eliminar el producto {producto_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def obtener_estadisticas():
    try:
        stats = search_manager.get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindexar")
def reindexar_completo():
    try:
        search_manager._initial_load()
        stats = search_manager.get_stats()
        return JSONResponse(content={
            "mensaje": "Reindexación completa exitosa",
            "estadisticas": stats
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    