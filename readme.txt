python busq_milvus.py

sudo docker-compose up -d
---------------------------
para reindex2:
UPDATE tienda_catalogoproductopadre 
SET descripcion = 'Pantalla AMOLED 6.43" Full HD+.
Cámara cuádruple 64 Mpx.
Batería 5000 mAh con carga rápida',
nombre = 'Teléfono Alex' 
WHERE id = 275;

UPDATE tienda_catalogoproductopadre 
SET descripcion = 'Nueva pantalla Super AMOLED 6.5\".\nCámara cuádruple mejorada 108 Mpx.\nBatería 6000 mAh con carga ultra rápida.',
nombre = 'Auto mounstro de Alex' 
WHERE id = 275;

curl -X PUT "http://localhost:8000/productos/1045/actualizar"

curl -X DELETE "http://localhost:8000/productos/1045/eliminar"

curl -X POST "http://localhost:8000/productos/1045/agregar"
