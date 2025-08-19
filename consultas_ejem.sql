Hola, escribe una consulta para obtener toda la informaciòn relacionada a los productos(ignorar datos de usuarios, fechas, etc.) de manera ordenada, por favor:  
CREATE TABLE `tienda_catalogoproductopadre` (
  `id` bigint UNSIGNED NOT NULL,
  `id_empresa` bigint UNSIGNED NOT NULL,
  `id_usuario` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `id_categoria` bigint UNSIGNED NOT NULL,
  `id_marca` bigint UNSIGNED DEFAULT NULL,
  `id_producto_shopify` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `id_tipo` int UNSIGNED NOT NULL DEFAULT '0',
  `variantes` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin,
  `nombre` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `descripcion` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `paises_envio` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `fecha_creacion` datetime NOT NULL,
  `fecha_modificacion` datetime NOT NULL,
  `mostrar_en_portada` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0',
  `envio_internacional` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0',
  `tiene_devolucion` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0',
  `tiene_variantes` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0',
  `es_academy` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0',
  `es_fisico` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT '0' COMMENT '0=producto digital, 1= fisico',
  `destacado` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '1' COMMENT '0=no,1=si',
  `aprobado` enum('0','1','2') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0' COMMENT '0:Pendiente,1:Aprobado,2:Rechazado',
  `activo` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '1'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `tienda_catalogoproductos` (
  `id` bigint UNSIGNED NOT NULL COMMENT 'PK',
  `id_empresa` bigint UNSIGNED NOT NULL,
  `id_padre` bigint UNSIGNED NOT NULL DEFAULT '0' COMMENT 'ID del producto principal. 0=principal',
  `id_usuario` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'UID del vendedor',
  `id_variante_shopify` varchar(100) DEFAULT NULL,
  `sku` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'SKU del producto',
  `clave` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Clave del producto',
  `clave_proveedor` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `descripcion_larga` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Descripcion larga del producto',
  `modelo` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Modelo',
  `variante_comb` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin,
  `tags` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'Tags, separados por comas',
  `atributos` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin,
  `es_inventariable` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0' COMMENT '0= no se llevara control de inventario, 1 = se llevara control de inventario',
  `cantidad_minima` mediumint UNSIGNED NOT NULL COMMENT 'minimo de producto que debe haber',
  `codigobarras` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci COMMENT 'Codigo de barras opcional',
  `habilitado` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '1' COMMENT '0=no se puede vender, 1=habilitado para su venta',
  `vender_sin_existencia` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0' COMMENT '0=no,1=si',
  `variante_principal` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0',
  `precio` double NOT NULL,
  `comision_porcentaje` double NOT NULL DEFAULT '0',
  `comision_utilidad` double DEFAULT NULL,
  `precio_descuento` double DEFAULT NULL,
  `moneda` varchar(3) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'MXN',
  `dropshiping` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `costo_envio` double NOT NULL DEFAULT '0',
  `dias_entrega` smallint NOT NULL DEFAULT '0',
  `envio_incluido` enum('1','0') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0',
  `envio_solitario` enum('0','1') NOT NULL DEFAULT '0',
  `slug` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `precio_envio` double NOT NULL DEFAULT '0',
  `fecha_creacion` datetime NOT NULL COMMENT 'Fecha de creacion',
  `fecha_modificacion` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Fecha de modificacion',
  `fecha_baja` datetime DEFAULT NULL COMMENT 'Fecha de baja',
  `activo` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '1' COMMENT '0=no activo,1=activo'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `tienda_categoriasproductos` (
  `id` bigint UNSIGNED NOT NULL COMMENT 'PK',
  `id_padre` bigint UNSIGNED NOT NULL DEFAULT '0' COMMENT 'Categoria padre',
  `lft` bigint UNSIGNED DEFAULT NULL,
  `rht` bigint UNSIGNED DEFAULT NULL,
  `nombre` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'Nombre de la categoria',
  `descripcion` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `tags` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Tags separados por coma',
  `atributos` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin,
  `tipo` enum('1','2') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '1' COMMENT '1=entregable,2=no entregable',
  `imagen` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT 'Nombre del archivo de imagen de la categoria',
  `banner_img` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `slug` varchar(300) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `fecha_creacion` datetime NOT NULL,
  `fecha_modificacion` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `mostrar_en_menu` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0' COMMENT 'Mostrar en menu bar\r\n',
  `mostrar_en_portada` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0' COMMENT '0=no, 1=si',
  `activo` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '1' COMMENT '0=borrado,1=activo'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE `tienda_marcas` (
  `id` bigint UNSIGNED NOT NULL COMMENT 'PK',
  `id_usuario` char(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '' COMMENT 'UID de alta seguridad_usuarios.id',
  `nombre` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
  `imagen` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `slug` varchar(300) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `fecha_alta` datetime NOT NULL,
  `fecha_modificacion` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `portada` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0',
  `mostrar_en_portada` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '0',
  `activo` enum('0','1') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT '1' COMMENT '0=borrado,1=activo'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Catalogo de marcas';

-- Indices de la tabla `tienda_catalogoproductopadre`
ALTER TABLE `tienda_catalogoproductopadre`
  ADD PRIMARY KEY (`id`),
  ADD KEY `id_categoria` (`id_categoria`),
  ADD KEY `id_marca` (`id_marca`),
  ADD KEY `id_empresa` (`id_empresa`),
  ADD KEY `id_usuario` (`id_usuario`);

-- Indices de la tabla `tienda_catalogoproductos`
ALTER TABLE `tienda_catalogoproductos`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `sku` (`sku`),
  ADD KEY `id_usuario` (`id_usuario`),
  ADD KEY `id_padre` (`id_padre`),
  ADD KEY `id_empresa` (`id_empresa`),
  ADD KEY `clave` (`clave`);
ALTER TABLE `tienda_catalogoproductos` ADD FULLTEXT KEY `descripcion` (`descripcion_larga`);

-- Indices de la tabla `tienda_categoriasproductos`
ALTER TABLE `tienda_categoriasproductos`
  ADD PRIMARY KEY (`id`),
  ADD KEY `id_padre` (`id_padre`);

-- Indices de la tabla `tienda_marcas`
ALTER TABLE `tienda_marcas`
  ADD PRIMARY KEY (`id`),
  ADD KEY `id_usuario` (`id_usuario`);

-- AUTO_INCREMENT de la tabla `tienda_catalogoproductopadre`
ALTER TABLE `tienda_catalogoproductopadre`
  MODIFY `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3052;

-- AUTO_INCREMENT de la tabla `tienda_catalogoproductos`
ALTER TABLE `tienda_catalogoproductos`
  MODIFY `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'PK', AUTO_INCREMENT=3868;

-- AUTO_INCREMENT de la tabla `tienda_categoriasproductos`
ALTER TABLE `tienda_categoriasproductos`
  MODIFY `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'PK', AUTO_INCREMENT=19;

-- AUTO_INCREMENT de la tabla `tienda_marcas`
ALTER TABLE `tienda_marcas`
  MODIFY `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'PK', AUTO_INCREMENT=39;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*----------------------------------------------------------------------------------------------------------

SELECT 
    p.id AS id_producto,
    p.nombre AS nombre_producto,
    p.descripcion AS descripcion_producto,
    p.variantes,
    p.mostrar_en_portada,
    p.envio_internacional,
    p.tiene_devolucion,
    p.tiene_variantes,
    p.es_academy,
    p.es_fisico,
    p.destacado,
    p.aprobado,
    p.activo,
    
    v.id AS id_variante,
    v.sku,
    v.clave,
    v.clave_proveedor,
    v.descripcion_larga,
    v.modelo,
    v.tags,
    v.es_inventariable,
    v.cantidad_minima,
    v.codigobarras,
    v.habilitado,
    v.vender_sin_existencia,
    v.variante_principal,
    v.precio,
    v.comision_porcentaje,
    v.comision_utilidad,
    v.precio_descuento,
    v.moneda,
    v.dropshiping,
    v.costo_envio,
    v.dias_entrega,
    v.envio_incluido,
    v.envio_solitario,
    v.slug,
    v.precio_envio,
    v.activo AS variante_activa,
    
    c.nombre AS categoria,
    c.descripcion AS descripcion_categoria,
    c.slug AS slug_categoria,
    
    m.nombre AS marca,
    m.slug AS slug_marca

FROM tienda_catalogoproductopadre p
LEFT JOIN tienda_catalogoproductos v 
       ON v.id_padre = p.id
LEFT JOIN tienda_categoriasproductos c 
       ON c.id = p.id_categoria
LEFT JOIN tienda_marcas m 
       ON m.id = p.id_marca
ORDER BY p.nombre ASC, v.precio ASC;
---------------------------------------
SELECT 
    p.id AS id_producto,
    p.nombre AS nombre_producto,
    p.descripcion AS descripcion_producto,
    p.variantes,
    p.tiene_variantes,
    p.destacado,
    p.activo,
    
    v.id AS id_variante,
    v.descripcion_larga,
    v.modelo,
    v.tags,
    v.variante_principal,
    v.precio,
    v.slug,
    v.activo AS variante_activa,
    
    c.nombre AS categoria,
    c.descripcion AS descripcion_categoria,
    c.slug AS slug_categoria,
    
    m.nombre AS marca,
    m.slug AS slug_marca

FROM tienda_catalogoproductopadre p
LEFT JOIN tienda_catalogoproductos v 
       ON v.id_padre = p.id
LEFT JOIN tienda_categoriasproductos c 
       ON c.id = p.id_categoria
LEFT JOIN tienda_marcas m 
       ON m.id = p.id_marca
ORDER BY p.nombre ASC, v.precio ASC;

SELECT
    p.id AS id_producto,
    p.nombre AS nombre_producto,
    p.descripcion AS descripcion_producto,
    p.variantes,
    p.tiene_variantes,
    p.destacado,
    p.activo,
    
    v.id AS id_variante,
    v.descripcion_larga,
    v.modelo,
    v.tags,
    v.variante_principal,
    v.precio,
    v.slug,
    v.activo AS variante_activa,
    
    c.nombre AS categoria,
    c.descripcion AS descripcion_categoria,
    c.slug AS slug_categoria,
    
    m.nombre AS marca,
    m.slug AS slug_marca

FROM tienda_catalogoproductopadre p
LEFT JOIN tienda_catalogoproductos v 
       ON v.id_padre = p.id 
LEFT JOIN tienda_categoriasproductos c 
       ON c.id = p.id_categoria
LEFT JOIN tienda_marcas m 
       ON m.id = p.id_marca
WHERE p.activo = '1'       
ORDER BY p.nombre ASC, v.precio ASC;

Contenido de SELECT variantes FROM tienda_catalogoproductopadre;
