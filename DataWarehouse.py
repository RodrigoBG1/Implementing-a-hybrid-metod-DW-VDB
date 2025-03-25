# Implementación básica de Data Warehouse con Python
# Este código muestra cómo crear un data warehouse simple usando SQLAlchemy y pandas

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import random

# 1. Configuración de la base de datos
# Usamos SQLite para facilitar las pruebas, pero se puede cambiar a PostgreSQL o MySQL
engine = create_engine('sqlite:///mi_data_warehouse.db', echo=True)
Base = declarative_base()

# 2. Definir el modelo dimensional (Schema)
# Definimos dimensiones y hechos según el esquema estrella

# Dimensión de Tiempo
class DimTiempo(Base):
    __tablename__ = 'dim_tiempo'
    
    id_tiempo = Column(Integer, primary_key=True)
    fecha = Column(DateTime)
    dia = Column(Integer)
    mes = Column(Integer)
    anio = Column(Integer)
    trimestre = Column(Integer)
    dia_semana = Column(Integer)
    es_feriado = Column(Integer)
    
    def __repr__(self):
        return f"<Tiempo(fecha='{self.fecha}')>"

# Dimensión de Producto
class DimProducto(Base):
    __tablename__ = 'dim_producto'
    
    id_producto = Column(Integer, primary_key=True)
    codigo_producto = Column(String)
    nombre = Column(String)
    categoria = Column(String)
    subcategoria = Column(String)
    precio_unitario = Column(Float)
    
    def __repr__(self):
        return f"<Producto(nombre='{self.nombre}')>"

# Dimensión de Cliente
class DimCliente(Base):
    __tablename__ = 'dim_cliente'
    
    id_cliente = Column(Integer, primary_key=True)
    codigo_cliente = Column(String)
    nombre = Column(String)
    ciudad = Column(String)
    region = Column(String)
    segmento = Column(String)
    
    def __repr__(self):
        return f"<Cliente(nombre='{self.nombre}')>"

# Tabla de Hechos - Ventas
class FactVentas(Base):
    __tablename__ = 'fact_ventas'
    
    id_venta = Column(Integer, primary_key=True)
    id_tiempo = Column(Integer, ForeignKey('dim_tiempo.id_tiempo'))
    id_producto = Column(Integer, ForeignKey('dim_producto.id_producto'))
    id_cliente = Column(Integer, ForeignKey('dim_cliente.id_cliente'))
    cantidad = Column(Integer)
    precio_venta = Column(Float)
    descuento = Column(Float)
    costo = Column(Float)
    margen = Column(Float)
    
    def __repr__(self):
        return f"<Venta(id='{self.id_venta}')>"

# 3. Crear las tablas en la base de datos
Base.metadata.create_all(engine)

# 4. Función para generar datos de ejemplo (ETL simplificado)
def generar_datos_ejemplo():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Generar dimensión de tiempo (últimos 2 años)
    fechas = []
    fecha_actual = datetime.now()
    fecha_inicio = fecha_actual - timedelta(days=730)  # ~2 años
    
    fecha = fecha_inicio
    while fecha <= fecha_actual:
        dim_tiempo = DimTiempo(
            fecha=fecha,
            dia=fecha.day,
            mes=fecha.month,
            anio=fecha.year,
            trimestre=(fecha.month-1)//3 + 1,
            dia_semana=fecha.weekday(),
            es_feriado=1 if fecha.weekday() >= 5 else 0  # Simplificación: fin de semana = feriado
        )
        session.add(dim_tiempo)
        fechas.append(fecha)
        fecha = fecha + timedelta(days=1)
    
    # Generar dimensión de productos
    categorias = ['Electrónica', 'Ropa', 'Hogar', 'Deportes']
    subcategorias = {
        'Electrónica': ['Móviles', 'Computadoras', 'Audio', 'Accesorios'],
        'Ropa': ['Camisetas', 'Pantalones', 'Abrigos', 'Calzado'],
        'Hogar': ['Muebles', 'Decoración', 'Electrodomésticos', 'Jardinería'],
        'Deportes': ['Fútbol', 'Baloncesto', 'Fitness', 'Ciclismo']
    }
    
    productos = []
    for i in range(1, 51):  # 50 productos
        categoria = random.choice(categorias)
        subcategoria = random.choice(subcategorias[categoria])
        dim_producto = DimProducto(
            codigo_producto=f'PROD-{i:04d}',
            nombre=f'Producto {i}',
            categoria=categoria,
            subcategoria=subcategoria,
            precio_unitario=round(random.uniform(10.0, 1000.0), 2)
        )
        session.add(dim_producto)
        productos.append(dim_producto)
    
    # Generar dimensión de clientes
    ciudades = ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao', 'Lima', 'Bogotá', 'Ciudad de México', 'Buenos Aires']
    regiones = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    segmentos = ['Consumidor', 'Corporativo', 'PYME']
    
    clientes = []
    for i in range(1, 101):  # 100 clientes
        dim_cliente = DimCliente(
            codigo_cliente=f'CLI-{i:04d}',
            nombre=f'Cliente {i}',
            ciudad=random.choice(ciudades),
            region=random.choice(regiones),
            segmento=random.choice(segmentos)
        )
        session.add(dim_cliente)
        clientes.append(dim_cliente)
    
    # Confirmar cambios para obtener IDs
    session.commit()
    
    # Generar hechos de ventas (transacciones)
    # Simulamos 5000 transacciones aleatorias
    for i in range(1, 5001):
        tiempo = session.query(DimTiempo).offset(random.randint(0, len(fechas)-1)).first()
        producto = session.query(DimProducto).offset(random.randint(0, len(productos)-1)).first()
        cliente = session.query(DimCliente).offset(random.randint(0, len(clientes)-1)).first()
        
        cantidad = random.randint(1, 10)
        precio_unitario = producto.precio_unitario
        descuento = round(random.uniform(0, 0.3) * precio_unitario, 2)  # Descuento entre 0% y 30%
        precio_venta = precio_unitario - descuento
        costo = round(precio_unitario * 0.6, 2)  # Costo es 60% del precio regular
        margen = precio_venta - costo
        
        fact_venta = FactVentas(
            id_tiempo=tiempo.id_tiempo,
            id_producto=producto.id_producto,
            id_cliente=cliente.id_cliente,
            cantidad=cantidad,
            precio_venta=precio_venta,
            descuento=descuento,
            costo=costo,
            margen=margen
        )
        session.add(fact_venta)
        
        # Commit cada 1000 registros para evitar consumo excesivo de memoria
        if i % 1000 == 0:
            session.commit()
    
    # Commit final
    session.commit()
    session.close()
    
    print("Datos de ejemplo generados correctamente")

# 5. Consultas analíticas (ejemplos de uso del data warehouse)
def ejecutar_consultas_ejemplo():
    # Conectar a la base de datos
    engine = create_engine('sqlite:///mi_data_warehouse.db')
    
    # Ejemplo 1: Ventas totales por categoría de producto
    query1 = """
    SELECT 
        p.categoria,
        SUM(v.cantidad * v.precio_venta) as ventas_totales
    FROM 
        fact_ventas v
    JOIN
        dim_producto p ON v.id_producto = p.id_producto
    GROUP BY 
        p.categoria
    ORDER BY 
        ventas_totales DESC
    """
    
    # Ejemplo 2: Ventas mensuales por año
    query2 = """
    SELECT 
        t.anio,
        t.mes,
        SUM(v.cantidad * v.precio_venta) as ventas_totales
    FROM 
        fact_ventas v
    JOIN
        dim_tiempo t ON v.id_tiempo = t.id_tiempo
    GROUP BY 
        t.anio, t.mes
    ORDER BY 
        t.anio, t.mes
    """
    
    # Ejemplo 3: Top 10 productos más rentables
    query3 = """
    SELECT 
        p.nombre,
        p.categoria,
        p.subcategoria,
        SUM(v.margen * v.cantidad) as margen_total
    FROM 
        fact_ventas v
    JOIN
        dim_producto p ON v.id_producto = p.id_producto
    GROUP BY 
        p.id_producto
    ORDER BY 
        margen_total DESC
    LIMIT 10
    """
    
    # Ejemplo 4: Ventas por segmento de cliente y región
    query4 = """
    SELECT 
        c.segmento,
        c.region,
        SUM(v.cantidad * v.precio_venta) as ventas_totales
    FROM 
        fact_ventas v
    JOIN
        dim_cliente c ON v.id_cliente = c.id_cliente
    GROUP BY 
        c.segmento, c.region
    ORDER BY 
        ventas_totales DESC
    """
    
    # Ejecutar las consultas y mostrar resultados
    print("\n--- Ventas totales por categoría de producto ---")
    df1 = pd.read_sql(query1, engine)
    print(df1)
    
    print("\n--- Ventas mensuales por año ---")
    df2 = pd.read_sql(query2, engine)
    print(df2)
    
    print("\n--- Top 10 productos más rentables ---")
    df3 = pd.read_sql(query3, engine)
    print(df3)
    
    print("\n--- Ventas por segmento de cliente y región ---")
    df4 = pd.read_sql(query4, engine)
    print(df4)
    
    # Generar visualizaciones básicas
    try:
        import matplotlib.pyplot as plt
        
        # Gráfico de barras - Ventas por categoría
        plt.figure(figsize=(10, 6))
        plt.bar(df1['categoria'], df1['ventas_totales'])
        plt.title('Ventas Totales por Categoría')
        plt.xlabel('Categoría')
        plt.ylabel('Ventas ($)')
        plt.savefig('ventas_por_categoria.png')
        
        # Línea temporal - Ventas mensuales
        df_pivot = df2.pivot(index='mes', columns='anio', values='ventas_totales')
        plt.figure(figsize=(12, 6))
        for col in df_pivot.columns:
            plt.plot(df_pivot.index, df_pivot[col], marker='o', label=f'Año {col}')
        plt.title('Ventas Mensuales por Año')
        plt.xlabel('Mes')
        plt.ylabel('Ventas ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig('ventas_mensuales.png')
        
        print("\nGráficos generados: ventas_por_categoria.png, ventas_mensuales.png")
    except ImportError:
        print("\nMatplotlib no está instalado. No se generaron gráficos.")

# Función principal
def main():
    print("Implementación de Data Warehouse con Python")
    print("1. Generando esquema y datos de ejemplo...")
    generar_datos_ejemplo()
    
    print("\n2. Ejecutando consultas analíticas...")
    ejecutar_consultas_ejemplo()
    
    print("\nImplementación completada. El data warehouse está listo para su uso.")

if __name__ == "__main__":
    main()