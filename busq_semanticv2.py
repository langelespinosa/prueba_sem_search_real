#-------- 
import time
import logging
import requests
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import meilisearch

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la base de datos MySQL
# Formato: mysql+pymysql://usuario:contraseña@host:puerto/nombre_bd
#DATABASE_URL = "mysql+pymysql://root:pass@localhost:3306/mi_database"
DATABASE_URL = "mysql+pymysql://web:password@localhost:3306/email"

# Modelo de datos con SQLAlchemy
Base = declarative_base()

class User(Base):
    #__tablename__ = 'users'
    __tablename__ = 'usuario'
    
    id = Column(Integer, primary_key=True)
    login = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    grupo = Column(String(255), nullable=False)
    identificacion = Column(String(255), nullable=False)

def setup_database():
    """Configura la conexión a MySQL y crea las tablas"""
    engine = create_engine(DATABASE_URL, echo=True)
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Datos de ejemplo
    usuarios_ejemplo = [
        User(id=101, login="jlopez", email="jlopez@fc.com", grupo="administrador", identificacion="MX1234567890"),
        User(id=102, login="mgarcia", email="mgarcia@marketplace.com", grupo="ventas", identificacion="MX0987654321"),
        User(id=103, login="cfernandez", email="cfernandez@academy.com", grupo="soporte", identificacion="MX5678901234"),
        User(id=104, login="rnavarro", email="rnavarro@example.com", grupo="marketing", identificacion="MX1122334455"),
        User(id=105, login="lramirez", email="lramirez@fc.com", grupo="administrador", identificacion="MX5566778899"),
        User(id=106, login="cperez", email="cperez@fc.com", grupo="administrador", identificacion="MX1234567891"),
        User(id=107, login="acarrillo", email="acarillo@marketplace.com", grupo="ventas", identificacion="MX0987654322"),
        User(id=108, login="rgrajales", email="rgrajales@academy.com", grupo="soporte", identificacion="MX5678901235"),
        User(id=109, login="rnunez", email="rnunez@marketplace.com", grupo="marketing", identificacion="MX1122334456"),
        User(id=110, login="benriquez", email="benriquez@fc.com", grupo="administrador", identificacion="MX5566778900"),
    ]
    """
    # Insertar usuarios si no existen
    for usuario in usuarios_ejemplo:
        existing = session.query(User).filter_by(id=usuario.id).first()
        if not existing:
            session.add(usuario)
    """
    session.commit()
    return session

def wait_for_meilisearch(host="http://localhost:7700", max_attempts=10):
    """Espera a que Meilisearch esté disponible"""
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
    """Configura el cliente de Meilisearch y el índice"""
    # Configuración del cliente
    #client = meilisearch.Client('http://localhost:7700', 'mySuperSecretKey')
    client = meilisearch.Client('http://localhost:7700', 'masterKey')
    
    # Verificar que Meilisearch esté disponible
    if not wait_for_meilisearch():
        raise Exception("Meilisearch no está disponible después de múltiples intentos")
    
    # Obtener el índice
    index = client.index('users')
    
    # Obtener usuarios de la base de datos
    session = setup_database()
    all_users = session.query(User).all()
    
    # Convertir a formato para Meilisearch
    docs = []
    for user in all_users:
        docs.append({
            "id": user.id,
            "login": user.login,
            "email": user.email,
            "grupo": user.grupo,
            "identificacion": user.identificacion
        })
    
    # Agregar documentos al índice
    try:
        index.add_documents(docs)
        logger.info(f"Se agregaron {len(docs)} usuarios a Meilisearch")
    except Exception as e:
        logger.error(f"Error al agregar usuarios a Meilisearch: {e}")
        raise
    
    # Config rankingRules
    settings = {
        "searchableAttributes": ["login", "email", "grupo", "identificacion"],
        "sortableAttributes": ["id"],
        "rankingRules": [
            "words",           # words coincidentes
            "typo",            # errores tipográf
            "proximity",       # Proximidad entre términos de búsqueda
            "attribute",       # Orden de atributos
            "sort",            # Ordenamiento
            "exactness"        # exact coincidencia
        ],
        # config opc
        "stopWords": ["el", "la", "de", "en", "y", "a"],  # words vacías en esp
        "synonyms": {
            "admin": ["administrador", "administrator"],
            "support": ["soporte", "ayuda"]
        },
        "filterableAttributes": ["grupo"],  # filtro por grupo
        "distinctAttribute": "email"  # quitar dupl por email
    }
    
    try:
        index.update_settings(settings)
        logger.info("Configuración del índice actualizada")
    except Exception as e:
        logger.error(f"Error al configurar el índice: {e}")
        raise
    
    return index, session

# Crear la aplicación Flask
app = Flask(__name__)

# Variables globales
index = None
db_session = None

@app.route('/buscar')
def buscar():
    """Endpoint simple de búsqueda"""
    query = request.args.get('q', '')
    
    try:
        result = index.search(query, {
            'limit': 3
        })
        return jsonify(result['hits'])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search')
def api_search():
    """Endpoint avanzado de búsqueda"""
    query = request.args.get('q', '')
    grupo_filter = request.args.get('grupo', '')
    
    search_params = {
        'limit': 20,
        'attributesToRetrieve': ['id', 'login', 'email', 'grupo', 'identificacion']
    }
    
    # Agregar filtro por grupo si se especifica
    if grupo_filter:
        search_params['filter'] = f'grupo = {grupo_filter}'
    
    try:
        result = index.search(query, search_params)
        return jsonify(result['hits'])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users')
def get_all_users():
    """Endpoint para obtener todos los usuarios de MySQL"""
    try:
        users = db_session.query(User).all()
        users_data = []
        for user in users:
            users_data.append({
                'id': user.id,
                'login': user.login,
                'email': user.email,
                'grupo': user.grupo,
                'identificacion': user.identificacion
            })
        return jsonify(users_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        # Configurar Meilisearch
        index, db_session = setup_meilisearch()
        
        # Iniciar el servidor
        logger.info("Servidor escuchando en http://localhost:3000")
        app.run(host='0.0.0.0', port=3000, debug=True)
        
    except Exception as e:
        logger.error(f"Error al inicializar la aplicación: {e}")
        raise