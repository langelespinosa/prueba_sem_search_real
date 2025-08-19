import time
import logging
import requests
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import meilisearch
from sentence_transformers import SentenceTransformer
import decimal

# =========================
# Configuración
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = "mysql+pymysql://web:password@localhost:3306/email"

Base = declarative_base()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Embeddings

class User(Base):
    __tablename__ = 'usuario'
    id = Column(Integer, primary_key=True)
    login = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    grupo = Column(String(255), nullable=False)
    identificacion = Column(String(255), nullable=False)

def setup_database():
    engine = create_engine(DATABASE_URL, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def wait_for_meilisearch(host="http://localhost:7700", max_attempts=10):
    for i in range(max_attempts):
        try:
            response = requests.get(f"{host}/health")
            if response.status_code == 200:
                logger.info(f"Meilisearch está listo: {response.text}")
                return True
        except requests.exceptions.RequestException:
            pass
        logger.info("Esperando a que Meilisearch esté listo...")
        time.sleep(2)
    return False

def setup_meilisearch():
    client = meilisearch.Client('http://localhost:7700', 'masterKey')
    
    if not wait_for_meilisearch():
        raise Exception("Meilisearch no está disponible")
    
    index = client.index('users')
    session = setup_database()
    all_users = session.query(User).all()
    
    # Indexar con embeddings
    docs = []
    for user in all_users:
        texto_busqueda = f"{user.login} {user.email} {user.grupo} {user.identificacion}"
        embedding = model.encode(texto_busqueda).tolist()
        
        docs.append({
            "id": user.id,
            "login": user.login,
            "email": user.email,
            "grupo": user.grupo,
            "identificacion": user.identificacion,
            "texto_busqueda": texto_busqueda,
            "_vectors": {"default": embedding}
        })
    
    index.add_documents(docs)
    
    # Configuración del índice
    settings = {
        "embedders": {
            "default": {
                "source": "userProvided",
                "dimensions": 384
            }
        },
        "searchableAttributes": ["login", "email", "grupo", "identificacion", "texto_busqueda"],
        "sortableAttributes": ["id"],
        "filterableAttributes": ["grupo"],
        "distinctAttribute": "email"
    }
    
    index.update_settings(settings)
    logger.info(f"Se indexaron {len(docs)} usuarios en Meilisearch con embeddings")
    
    return index, session

# =========================
# Flask App
# =========================
app = Flask(__name__)

index = None
db_session = None

@app.route('/buscar/semantica')
def buscar_semantica():
    query = request.args.get('q', '')
    threshold = float(request.args.get('threshold', 0.4))  # default 0.4
    
    try:
        query_embedding = model.encode(query).tolist()
        result = index.search(
            "",
            {
                "vector": query_embedding,
                "limit": 50,  # pedimos más y filtramos
                "hybrid": {"semanticRatio": 1.0}
            }
        )
        
        # Filtrar por umbral
        hits = [hit for hit in result["hits"] if hit.get("_semanticScore", 0) >= threshold]
        
        return jsonify(hits)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/buscar/hibrida')
def buscar_hibrida():
    query = request.args.get('q', '')
    ratio = float(request.args.get('ratio', 0.5))
    try:
        query_embedding = model.encode(query).tolist()
        result = index.search(
            query,
            {
                "vector": query_embedding,
                "limit": 20,
                "hybrid": {"semanticRatio": ratio}
            }
        )
        return jsonify(result["hits"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users')
def get_all_users():
    try:
        users = db_session.query(User).all()
        users_data = [{"id": u.id, "login": u.login, "email": u.email, "grupo": u.grupo, "identificacion": u.identificacion} for u in users]
        return jsonify(users_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        index, db_session = setup_meilisearch()
        logger.info("Servidor escuchando en http://localhost:3000")
        app.run(host='0.0.0.0', port=3000, debug=True)
    except Exception as e:
        logger.error(f"Error al inicializar la aplicación: {e}")
        raise
