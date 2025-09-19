from fastapi import FastAPI, Query
import mysql.connector
from typing import List
from pydantic import BaseModel
import uvicorn

app = FastAPI()

db_config = {
    'user': 'root',
    'password': 'pass',
    'host': 'localhost',
    'database': 'fireclub_back_pub'
}

class Product(BaseModel):
    """id: int
    name: str
    description: str
    price: float"""
    id: int
    nombre: str
    descripcion: str
    variantes_comb: str
    id_padre: int
    categoria: str
    marca: str
    
@app.get("/products")
def search_products(q: str = Query(None)):
    db = mysql.connector.connect(**db_config)
    cursor = db.cursor(dictionary=True)

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
    
    cursor.execute(query, (f'%{q}%', f'%{q}%'))
    results = cursor.fetchall()

    cursor.close()
    db.close()

    return [Product(**product) for product in results]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    