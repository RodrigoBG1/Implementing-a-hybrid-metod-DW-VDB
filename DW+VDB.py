# Sistema Híbrido: Data Warehouse + Base de Datos Vectorial
# Esta implementación combina un data warehouse tradicional con capacidades 
# de búsqueda semántica vectorial para análisis de datos estructurados y no estructurados

import numpy as np
import pandas as pd
import sqlite3
import os
import faiss
import pickle
import json
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple, Optional, Union, Any
from sentence_transformers import SentenceTransformer
import time
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, ForeignKey, Text, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Configuración de rutas y directorios
BASE_DIR = "hybrid_analytics_system"
DW_DIR = os.path.join(BASE_DIR, "data_warehouse")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_db")
os.makedirs(DW_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ========================================================================
# PARTE 1: IMPLEMENTACIÓN DEL DATA WAREHOUSE
# ========================================================================

# Configuración de SQLAlchemy para el Data Warehouse
DW_DB_PATH = os.path.join(DW_DIR, "data_warehouse.db")
dw_engine = create_engine(f'sqlite:///{DW_DB_PATH}', echo=False)
Base = declarative_base()

# Definición del Esquema del Data Warehouse (Modelo Dimensional)
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
    
    # Relación con hechos
    ventas = relationship("FactVentas", back_populates="tiempo")
    comentarios = relationship("FactComentariosProducto", back_populates="tiempo")

class DimProducto(Base):
    __tablename__ = 'dim_producto'
    
    id_producto = Column(Integer, primary_key=True)
    codigo_producto = Column(String)
    nombre = Column(String)
    categoria = Column(String)
    subcategoria = Column(String)
    precio_unitario = Column(Float)
    descripcion = Column(Text)  # Descripciones largas de productos para vectorización
    
    # Relación con hechos
    ventas = relationship("FactVentas", back_populates="producto")
    comentarios = relationship("FactComentariosProducto", back_populates="producto")
    
    # Vector ID para enlace con la base de datos vectorial
    vector_id = Column(Integer, nullable=True)

class DimCliente(Base):
    __tablename__ = 'dim_cliente'
    
    id_cliente = Column(Integer, primary_key=True)
    codigo_cliente = Column(String)
    nombre = Column(String)
    ciudad = Column(String)
    region = Column(String)
    segmento = Column(String)
    perfil = Column(Text, nullable=True)  # Perfil de cliente para vectorización
    
    # Relación con hechos
    ventas = relationship("FactVentas", back_populates="cliente")
    comentarios = relationship("FactComentariosProducto", back_populates="cliente")
    
    # Vector ID para enlace con la base de datos vectorial
    vector_id = Column(Integer, nullable=True)

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
    
    # Relaciones
    tiempo = relationship("DimTiempo", back_populates="ventas")
    producto = relationship("DimProducto", back_populates="ventas")
    cliente = relationship("DimCliente", back_populates="ventas")

class FactComentariosProducto(Base):
    __tablename__ = 'fact_comentarios_producto'
    
    id_comentario = Column(Integer, primary_key=True)
    id_tiempo = Column(Integer, ForeignKey('dim_tiempo.id_tiempo'))
    id_producto = Column(Integer, ForeignKey('dim_producto.id_producto'))
    id_cliente = Column(Integer, ForeignKey('dim_cliente.id_cliente'))
    comentario = Column(Text)
    calificacion = Column(Integer)  # Por ejemplo, 1-5 estrellas
    sentimiento = Column(Float)     # Puntuación de sentimiento -1 a 1
    
    # Relaciones
    tiempo = relationship("DimTiempo", back_populates="comentarios")
    producto = relationship("DimProducto", back_populates="comentarios")
    cliente = relationship("DimCliente", back_populates="comentarios")
    
    # Vector ID para enlace con la base de datos vectorial
    vector_id = Column(Integer, nullable=True)

# Crear tablas en la base de datos
Base.metadata.create_all(dw_engine)

# Clase para gestionar operaciones del Data Warehouse
class DataWarehouseManager:
    def __init__(self, engine=dw_engine):
        self.engine = engine
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def close(self):
        self.session.close()
        
    def generate_sample_data(self, num_products=50, num_customers=100, num_days=365, num_comments=500):
        """Genera datos de ejemplo para el data warehouse"""
        print("Generando datos de ejemplo para el Data Warehouse...")
        
        # Generar dimensión de tiempo
        fechas = []
        fecha_actual = datetime.now()
        fecha_inicio = fecha_actual - timedelta(days=num_days)
        
        print("Generando dimensión de tiempo...")
        fecha = fecha_inicio
        while fecha <= fecha_actual:
            dim_tiempo = DimTiempo(
                fecha=fecha,
                dia=fecha.day,
                mes=fecha.month,
                anio=fecha.year,
                trimestre=(fecha.month-1)//3 + 1,
                dia_semana=fecha.weekday(),
                es_feriado=1 if fecha.weekday() >= 5 else 0
            )
            self.session.add(dim_tiempo)
            fechas.append(fecha)
            fecha = fecha + timedelta(days=1)
        
        self.session.commit()
        
        # Generar productos
        categorias = ['Electrónica', 'Ropa', 'Hogar', 'Deportes', 'Alimentos']
        subcategorias = {
            'Electrónica': ['Móviles', 'Computadoras', 'Audio', 'Accesorios'],
            'Ropa': ['Camisetas', 'Pantalones', 'Abrigos', 'Calzado'],
            'Hogar': ['Muebles', 'Decoración', 'Electrodomésticos', 'Jardinería'],
            'Deportes': ['Fútbol', 'Baloncesto', 'Fitness', 'Ciclismo'],
            'Alimentos': ['Frescos', 'Congelados', 'Bebidas', 'Snacks']
        }
        
        # Descripciones largas de ejemplo para productos
        descripciones_por_categoria = {
            'Electrónica': [
                "Producto de última generación con tecnología avanzada que ofrece rendimiento excepcional. Incluye características innovadoras y diseño elegante para usuarios exigentes.",
                "Dispositivo de alta calidad con componentes premium y excelente durabilidad. Perfecto para uso diario con batería de larga duración y funciones inteligentes.",
                "Tecnología de vanguardia en un diseño compacto y ligero. Conectividad mejorada y compatibilidad con múltiples plataformas para máxima versatilidad."
            ],
            'Ropa': [
                "Prenda confeccionada con materiales de primera calidad para máximo confort y durabilidad. Diseño versátil que combina con múltiples estilos y ocasiones.",
                "Tejido premium de alta resistencia con acabados de lujo y atención al detalle. Corte moderno y ajuste perfecto para todos los tipos de cuerpo.",
                "Combinación perfecta de estilo y funcionalidad con materiales sostenibles. Ideal para uso diario con fácil mantenimiento y durabilidad garantizada."
            ],
            'Hogar': [
                "Pieza esencial para el hogar moderno con diseño atemporal y acabados de calidad. Funcionalidad optimizada para maximizar el espacio y la comodidad.",
                "Producto duradero fabricado con materiales resistentes y acabados de alta calidad. Diseño que complementa cualquier decoración y estilo de hogar.",
                "Solución práctica para optimizar espacios con diseño elegante y construcción robusta. Fácil montaje y mantenimiento para años de uso sin problemas."
            ],
            'Deportes': [
                "Equipamiento deportivo de nivel profesional diseñado para maximizar el rendimiento. Materiales ligeros y resistentes para un uso intensivo y duradero.",
                "Producto desarrollado con tecnología avanzada para mejorar el desempeño deportivo. Diseño ergonómico que reduce la fatiga y previene lesiones.",
                "Artículo esencial para entusiastas del deporte con características premium y construcción duradera. Perfecto para uso regular en diversas condiciones."
            ],
            'Alimentos': [
                "Producto gourmet elaborado con ingredientes seleccionados de la más alta calidad. Sabor auténtico y preparación cuidadosa siguiendo métodos tradicionales.",
                "Alimento natural sin conservantes artificiales ni aditivos. Elaborado según estrictos estándares de calidad con ingredientes frescos y sostenibles.",
                "Opción saludable con perfil nutritivo equilibrado y sabor excepcional. Ideal para dietas especiales con información detallada de alérgenos."
            ]
        }
        
        print(f"Generando {num_products} productos...")
        for i in range(1, num_products + 1):
            categoria = random.choice(categorias)
            subcategoria = random.choice(subcategorias[categoria])
            descripcion = random.choice(descripciones_por_categoria[categoria])
            
            # Añadir algo de variación a las descripciones
            palabras_adicionales = [
                "Exclusivo", "Premium", "Económico", "Versátil", "Profesional", 
                "Básico", "Avanzado", "Recomendado", "Popular", "Innovador"
            ]
            
            descripcion_final = f"{random.choice(palabras_adicionales)} {descripcion} Código: {i:04d}"
            
            dim_producto = DimProducto(
                codigo_producto=f'PROD-{i:04d}',
                nombre=f'Producto {i} {subcategoria}',
                categoria=categoria,
                subcategoria=subcategoria,
                precio_unitario=round(random.uniform(10.0, 1000.0), 2),
                descripcion=descripcion_final
            )
            self.session.add(dim_producto)
        
        self.session.commit()
        
        # Generar clientes
        ciudades = ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao', 'Lima', 'Bogotá', 'Ciudad de México', 'Buenos Aires']
        regiones = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
        segmentos = ['Consumidor', 'Corporativo', 'PYME']
        
        # Perfiles de ejemplo para clientes
        perfiles_cliente = [
            "Cliente habitual que prefiere productos de calidad y está dispuesto a pagar por marcas premium. Suele hacer compras frecuentes de valor medio-alto.",
            "Comprador ocasional que busca ofertas y descuentos. Sensible al precio y normalmente compara opciones antes de decidir su compra.",
            "Cliente fiel a determinadas categorías donde prefiere calidad aunque compra otras categorías buscando el mejor precio.",
            "Entusiasta de la tecnología que busca los últimos lanzamientos y está dispuesto a invertir en productos innovadores.",
            "Cliente que valora la sostenibilidad y responsabilidad social de las marcas, priorizando productos ecológicos y empresas comprometidas.",
            "Comprador pragmático que busca durabilidad y funcionalidad por encima de tendencias o marcas. Valora la relación calidad-precio.",
            "Cliente premium que busca exclusividad y servicio personalizado. Realiza compras de alto valor y es menos sensible al precio."
        ]
        
        print(f"Generando {num_customers} clientes...")
        for i in range(1, num_customers + 1):
            dim_cliente = DimCliente(
                codigo_cliente=f'CLI-{i:04d}',
                nombre=f'Cliente {i}',
                ciudad=random.choice(ciudades),
                region=random.choice(regiones),
                segmento=random.choice(segmentos),
                perfil=random.choice(perfiles_cliente)
            )
            self.session.add(dim_cliente)
        
        self.session.commit()
        
        # Generar hechos de ventas
        print("Generando transacciones de ventas...")
        tiempos = self.session.query(DimTiempo).all()
        productos = self.session.query(DimProducto).all()
        clientes = self.session.query(DimCliente).all()
        
        num_ventas = min(num_days * 20, 10000)  # Aproximadamente 20 ventas por día
        print(f"Generando {num_ventas} transacciones...")
        
        for i in range(1, num_ventas + 1):
            tiempo = random.choice(tiempos)
            producto = random.choice(productos)
            cliente = random.choice(clientes)
            
            cantidad = random.randint(1, 10)
            precio_unitario = producto.precio_unitario
            descuento = round(random.uniform(0, 0.3) * precio_unitario, 2)
            precio_venta = precio_unitario - descuento
            costo = round(precio_unitario * 0.6, 2)
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
            self.session.add(fact_venta)
            
            if i % 1000 == 0:
                self.session.commit()
                print(f"  Procesadas {i} ventas...")
        
        self.session.commit()
        
        # Generar comentarios de productos
        print(f"Generando {num_comments} comentarios de productos...")
        
        # Plantillas de comentarios positivos, neutrales y negativos
        comentarios_templates = {
            "positivos": [
                "Excelente producto. {aspecto_positivo}. Totalmente recomendable.",
                "Muy satisfecho con mi compra. {aspecto_positivo} y {otro_aspecto_positivo}.",
                "Supera mis expectativas. {aspecto_positivo}, sin duda volvería a comprarlo.",
                "Gran relación calidad-precio. {aspecto_positivo} y entrega rápida.",
                "Producto de primera calidad. {aspecto_positivo}, estoy encantado."
            ],
            "neutrales": [
                "Producto correcto. {aspecto_positivo} pero {aspecto_negativo}.",
                "Cumple su función aunque {aspecto_negativo}. En general satisfecho.",
                "Esperaba algo más por el precio. {aspecto_positivo}, pero {aspecto_negativo}.",
                "Producto estándar, ni destaca ni decepciona. {aspecto_positivo} aunque podría mejorar.",
                "Aceptable. {aspecto_positivo} pero creo que hay mejores opciones en el mercado."
            ],
            "negativos": [
                "Decepcionado con el producto. {aspecto_negativo} y {otro_aspecto_negativo}.",
                "No lo recomendaría. {aspecto_negativo} a pesar del precio.",
                "No cumple con lo esperado. {aspecto_negativo}, no volvería a comprarlo.",
                "Mala experiencia. {aspecto_negativo} y el servicio postventa no ayudó.",
                "Calidad inferior a lo anunciado. {aspecto_negativo}, me siento estafado."
            ]
        }
        
        aspectos_positivos = [
            "Excelente acabado", 
            "Muy duradero", 
            "Funciona perfectamente", 
            "Diseño elegante",
            "Fácil de usar", 
            "Gran calidad de materiales", 
            "Precio inmejorable", 
            "Envío rápido",
            "Muy completo", 
            "Rendimiento excepcional"
        ]
        
        aspectos_negativos = [
            "Acabado mejorable", 
            "Poca durabilidad", 
            "Funcionalidad limitada", 
            "Diseño anticuado",
            "Complejo de utilizar", 
            "Materiales de baja calidad", 
            "Precio excesivo", 
            "Envío con retraso",
            "Faltan características básicas", 
            "Rendimiento por debajo de lo esperado"
        ]
        
        for i in range(1, num_comments + 1):
            tiempo = random.choice(tiempos)
            producto = random.choice(productos)
            cliente = random.choice(clientes)
            
            # Decidir tipo de comentario y calificación
            tipo_comentario = random.choices(
                ["positivos", "neutrales", "negativos"], 
                weights=[0.6, 0.25, 0.15], 
                k=1
            )[0]
            
            if tipo_comentario == "positivos":
                calificacion = random.randint(4, 5)
                sentimiento = random.uniform(0.5, 1.0)
            elif tipo_comentario == "neutrales":
                calificacion = random.randint(3, 4)
                sentimiento = random.uniform(-0.3, 0.5)
            else:
                calificacion = random.randint(1, 2)
                sentimiento = random.uniform(-1.0, -0.3)
            
            # Generar comentario a partir de template
            template = random.choice(comentarios_templates[tipo_comentario])
            aspecto_positivo = random.choice(aspectos_positivos)
            otro_aspecto_positivo = random.choice([ap for ap in aspectos_positivos if ap != aspecto_positivo])
            aspecto_negativo = random.choice(aspectos_negativos)
            otro_aspecto_negativo = random.choice([an for an in aspectos_negativos if an != aspecto_negativo])
            
            comentario = template.format(
                aspecto_positivo=aspecto_positivo,
                otro_aspecto_positivo=otro_aspecto_positivo,
                aspecto_negativo=aspecto_negativo,
                otro_aspecto_negativo=otro_aspecto_negativo
            )
            
            # Añadir algo de contexto relacionado con la categoría
            if random.random() > 0.5:
                categoria_frases = {
                    'Electrónica': [
                        "La batería dura todo el día.",
                        "La pantalla tiene una nitidez impresionante.",
                        "La velocidad de procesamiento es notable.",
                        "La conectividad es excelente."
                    ],
                    'Ropa': [
                        "La tela es muy cómoda y suave.",
                        "El tallaje es correcto.",
                        "Los acabados son de calidad.",
                        "Mantiene bien la forma después de lavar."
                    ],
                    'Hogar': [
                        "Encaja perfectamente en mi decoración.",
                        "Es muy fácil de limpiar.",
                        "Ocupa poco espacio y es funcional.",
                        "La construcción es sólida y estable."
                    ],
                    'Deportes': [
                        "Perfecto para mi rutina de ejercicios.",
                        "Ofrece buen soporte y comodidad.",
                        "Material transpirable y ligero.",
                        "Resistente al uso intensivo."
                    ],
                    'Alimentos': [
                        "El sabor es excelente y natural.",
                        "Ingredientes de calidad notables.",
                        "Perfecto para ocasiones especiales.",
                        "Buena relación cantidad-precio."
                    ]
                }
                
                frases_categoria = categoria_frases.get(producto.categoria, ["Muy recomendable."])
                comentario += f" {random.choice(frases_categoria)}"
            
            fact_comentario = FactComentariosProducto(
                id_tiempo=tiempo.id_tiempo,
                id_producto=producto.id_producto,
                id_cliente=cliente.id_cliente,
                comentario=comentario,
                calificacion=calificacion,
                sentimiento=sentimiento
            )
            self.session.add(fact_comentario)
            
            if i % 100 == 0:
                self.session.commit()
                print(f"  Procesados {i} comentarios...")
        
        self.session.commit()
        print("Generación de datos de ejemplo completada.")
    
    def run_sample_queries(self):
        """Ejecuta consultas de ejemplo en el data warehouse"""
        print("\nEjecutando consultas de ejemplo en el Data Warehouse:")
        
        queries = {
            "Ventas por categoría": """
                SELECT 
                    p.categoria,
                    COUNT(v.id_venta) as num_ventas,
                    SUM(v.cantidad) as unidades_vendidas,
                    SUM(v.cantidad * v.precio_venta) as ingresos_totales,
                    SUM(v.margen) as margen_total
                FROM 
                    fact_ventas v
                JOIN
                    dim_producto p ON v.id_producto = p.id_producto
                GROUP BY 
                    p.categoria
                ORDER BY 
                    ingresos_totales DESC
            """,
            
            "Ventas mensuales último año": """
                SELECT 
                    t.anio,
                    t.mes,
                    SUM(v.cantidad * v.precio_venta) as ventas_totales
                FROM 
                    fact_ventas v
                JOIN
                    dim_tiempo t ON v.id_tiempo = t.id_tiempo
                WHERE
                    t.fecha >= date('now', '-1 year')
                GROUP BY 
                    t.anio, t.mes
                ORDER BY 
                    t.anio, t.mes
            """,
            
            "Top 5 productos por margen": """
                SELECT 
                    p.nombre,
                    p.categoria,
                    SUM(v.cantidad) as unidades_vendidas,
                    SUM(v.margen) as margen_total
                FROM 
                    fact_ventas v
                JOIN
                    dim_producto p ON v.id_producto = p.id_producto
                GROUP BY 
                    p.id_producto
                ORDER BY 
                    margen_total DESC
                LIMIT 5
            """,
            
            "Sentimiento promedio por categoría": """
                SELECT 
                    p.categoria,
                    AVG(c.calificacion) as calificacion_promedio,
                    AVG(c.sentimiento) as sentimiento_promedio,
                    COUNT(c.id_comentario) as num_comentarios
                FROM 
                    fact_comentarios_producto c
                JOIN
                    dim_producto p ON c.id_producto = p.id_producto
                GROUP BY 
                    p.categoria
                ORDER BY 
                    sentimiento_promedio DESC
            """
        }
        
        for nombre, query in queries.items():
            print(f"\n--- {nombre} ---")
            result = pd.read_sql(text(query), self.engine)
            print(result)

# ========================================================================
# PARTE 2: IMPLEMENTACIÓN DE LA BASE DE DATOS VECTORIAL
# ========================================================================

class VectorDatabase:
    """Base de datos vectorial para análisis semántico de texto"""
    
    def __init__(self, embedding_model_name: str = 'paraphrase-MiniLM-L6-v2', dimension: int = 384):
        """
        Inicializa la base de datos vectorial.
        
        Args:
            embedding_model_name: Nombre del modelo de SentenceTransformer a utilizar
            dimension: Dimensión de los vectores de embedding
        """
        self.embedding_model_name = embedding_model_name
        self.dimension = dimension
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Inicializar índice FAISS
        self.index = faiss.IndexFlatL2(dimension)
        
        # Almacenamiento para metadatos
        self.documents = []
        self.document_embeddings = []
        self.id_to_index_map = {}
        self.next_id = 0
        
        print(f"Base de datos vectorial inicializada con modelo: {embedding_model_name}")
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[int]:
        """
        Añade documentos a la base de datos vectorial.
        
        Args:
            texts: Lista de textos a añadir
            metadatas: Lista opcional de metadatos asociados a cada texto
            
        Returns:
            Lista de IDs asignados a los documentos
        """
        if not texts:
            return []
            
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        if len(texts) != len(metadatas):
            raise ValueError("El número de textos y metadatos debe coincidir")
            
        # Generar embeddings para los textos
        start_time = time.time()
        embeddings = self.embedding_model.encode(texts)
        print(f"Generados {len(texts)} embeddings en {time.time() - start_time:.2f} segundos")
        
        # Asignar IDs y guardar documentos
        ids = []
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            doc_id = self.next_id
            self.id_to_index_map[doc_id] = len(self.documents)
            
            document = {
                "id": doc_id,
                "text": text,
                "metadata": metadata
            }
            
            self.documents.append(document)
            self.document_embeddings.append(embedding)
            ids.append(doc_id)
            self.next_id += 1
            
        # Actualizar el índice FAISS
        embeddings_array = np.array(self.document_embeddings[-len(texts):]).astype('float32')
        self.index.add(embeddings_array)
        
        print(f"Añadidos {len(texts)} documentos a la base de datos (Total: {len(self.documents)})")
        return ids
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Busca documentos similares a la consulta.
        
        Args:
            query: Texto de consulta
            top_k: Número de resultados a devolver
            
        Returns:
            Lista de documentos más similares con sus scores
        """
        # Generar embedding para la consulta
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Realizar búsqueda
        top_k = min(top_k, len(self.documents))
        if top_k == 0:
            return []
            
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            top_k
        )
        
        # Preparar resultados
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # -1 significa que no se encontró resultado
                document = self.documents[idx]
                results.append({
                    "id": document["id"],
                    "text": document["text"],
                    "metadata": document["metadata"],
                    "score": float(1.0 / (1.0 + distance))  # Convertir distancia a similitud
                })
                
        return results
    
    def save(self, directory: str) -> None:
        """
        Guarda la base de datos en disco.
        
        Args:
            directory: Directorio donde guardar los archivos
        """
        os.makedirs(directory, exist_ok=True)
        
        # Guardar índice FAISS
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))
        
        # Guardar metadatos y embeddings
        with open(os.path.join(directory, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "document_embeddings": self.document_embeddings,
                "id_to_index_map": self.id_to_index_map,
                "next_id": self.next_id,
                "embedding_model_name": self.embedding_model_name,
                "dimension": self.dimension
            }, f)
            
        print(f"Base de datos vectorial guardada en {directory}")
        
    @classmethod
    def load(cls, directory: str) -> 'VectorDatabase':
        """
        Carga la base de datos desde disco.
        
        Args:
            directory: Directorio donde están los archivos guardados
            
        Returns:
            Instancia de VectorDatabase cargada
        """
        # Cargar metadatos
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
            
        # Crear instancia
        db = cls(
            embedding_model_name=metadata["embedding_model_name"],
            dimension=metadata["dimension"]
        )
        
        # Cargar índice FAISS
        db.index = faiss.read_index(os.path.join(directory, "faiss_index.bin"))
        
        # Restaurar estado
        db.documents = metadata["documents"]
        db.document_embeddings = metadata["document_embeddings"]
        db.id_to_index_map = metadata["id_to_index_map"]
        db.next_id = metadata["next_id"]
        
        print(f"Base de datos vectorial cargada desde {directory} con {len(db.documents)} documentos")
        return db
        
# ========================================================================
# PARTE 3: SISTEMA HÍBRIDO - INTEGRACIÓN DW + DB VECTORIAL
# ========================================================================

class HybridAnalyticsSystem:
    """
    Sistema híbrido que integra data warehouse y base de datos vectorial
    para proporcionar análisis tanto estructurado como semántico.
    """
    
    def __init__(self, base_dir=BASE_DIR):
        """
        Inicializa el sistema híbrido.
        
        Args:
            base_dir: Directorio base para guardar los datos
        """
        self.base_dir = base_dir
        self.dw_dir = os.path.join(base_dir, "data_warehouse")
        self.vector_dir = os.path.join(base_dir, "vector_db")
        
        # Inicializar componentes
        self.dw = DataWarehouseManager()
        self.vector_db = VectorDatabase()
        
        # Mapeo entre IDs de vector y IDs del data warehouse
        self.vector_to_dw_map = {
            'product': {},  # vector_id -> producto_id
            'customer': {},  # vector_id -> cliente_id
            'comment': {}   # vector_id -> comentario_id
        }
        
        self.dw_to_vector_map = {
            'product': {},  # producto_id -> vector_id
            'customer': {}, # cliente_id -> vector_id
            'comment': {}   # comentario_id -> vector_id
        }
        
        print("Sistema híbrido de análisis inicializado")
    
    def initialize_with_sample_data(self):
        """
        Inicializa el sistema con datos de ejemplo y construye la conexión
        entre el data warehouse y la base de datos vectorial.
        """
        # 1. Generar datos para el data warehouse
        self.dw.generate_sample_data()
        
        # 2. Vectorizar descripciones de productos
        self._vectorize_product_descriptions()
        
        # 3. Vectorizar perfiles de clientes
        self._vectorize_customer_profiles()
        
        # 4. Vectorizar comentarios de productos
        self._vectorize_product_comments()
        
        # 5. Guardar estado
        self.save()
        
        print("Sistema híbrido inicializado con datos de ejemplo")
    
    def _vectorize_product_descriptions(self):
        """Vectoriza las descripciones de productos y las añade a la base vectorial"""
        print("\nVectorizando descripciones de productos...")
        
        # Obtener productos del data warehouse
        result = self.dw.session.execute(text("""
            SELECT id_producto, nombre, descripcion, categoria, subcategoria
            FROM dim_producto
            WHERE descripcion IS NOT NULL
        """))
        
        productos = []
        for row in result:
            productos.append({
                'id_producto': row[0],
                'nombre': row[1],
                'descripcion': row[2],
                'categoria': row[3],
                'subcategoria': row[4]
            })
        
        if not productos:
            print("No hay descripciones de productos para vectorizar")
            return
        
        # Preparar datos para vectorización
        textos = [p['descripcion'] for p in productos]
        metadatos = [{
            'tipo': 'producto',
            'id_producto': p['id_producto'],
            'nombre': p['nombre'],
            'categoria': p['categoria'],
            'subcategoria': p['subcategoria']
        } for p in productos]
        
        # Añadir a la base de datos vectorial
        vector_ids = self.vector_db.add_documents(textos, metadatos)
        
        # Actualizar mapeos
        for i, p in enumerate(productos):
            vector_id = vector_ids[i]
            producto_id = p['id_producto']
            
            self.vector_to_dw_map['product'][vector_id] = producto_id
            self.dw_to_vector_map['product'][producto_id] = vector_id
            
            # Actualizar el ID vectorial en el data warehouse
            self.dw.session.execute(text(f"""
                UPDATE dim_producto
                SET vector_id = {vector_id}
                WHERE id_producto = {producto_id}
            """))
        
        self.dw.session.commit()
        print(f"Vectorizadas {len(productos)} descripciones de productos")
    
    def _vectorize_customer_profiles(self):
        """Vectoriza los perfiles de clientes y los añade a la base vectorial"""
        print("\nVectorizando perfiles de clientes...")
        
        # Obtener clientes del data warehouse
        result = self.dw.session.execute(text("""
            SELECT id_cliente, nombre, ciudad, region, segmento, perfil
            FROM dim_cliente
            WHERE perfil IS NOT NULL
        """))
        
        clientes = []
        for row in result:
            clientes.append({
                'id_cliente': row[0],
                'nombre': row[1],
                'ciudad': row[2],
                'region': row[3],
                'segmento': row[4],
                'perfil': row[5]
            })
        
        if not clientes:
            print("No hay perfiles de clientes para vectorizar")
            return
        
        # Preparar datos para vectorización
        textos = [c['perfil'] for c in clientes]
        metadatos = [{
            'tipo': 'cliente',
            'id_cliente': c['id_cliente'],
            'nombre': c['nombre'],
            'ciudad': c['ciudad'],
            'region': c['region'],
            'segmento': c['segmento']
        } for c in clientes]
        
        # Añadir a la base de datos vectorial
        vector_ids = self.vector_db.add_documents(textos, metadatos)
        
        # Actualizar mapeos
        for i, c in enumerate(clientes):
            vector_id = vector_ids[i]
            cliente_id = c['id_cliente']
            
            self.vector_to_dw_map['customer'][vector_id] = cliente_id
            self.dw_to_vector_map['customer'][cliente_id] = vector_id
            
            # Actualizar el ID vectorial en el data warehouse
            self.dw.session.execute(text(f"""
                UPDATE dim_cliente
                SET vector_id = {vector_id}
                WHERE id_cliente = {cliente_id}
            """))
        
        self.dw.session.commit()
        print(f"Vectorizados {len(clientes)} perfiles de clientes")
    
    def _vectorize_product_comments(self):
        """Vectoriza los comentarios de productos y los añade a la base vectorial"""
        print("\nVectorizando comentarios de productos...")
        
        # Obtener comentarios del data warehouse
        result = self.dw.session.execute(text("""
            SELECT 
                c.id_comentario, c.comentario, c.calificacion, c.sentimiento,
                p.id_producto, p.nombre as producto_nombre,
                cl.id_cliente, cl.nombre as cliente_nombre
            FROM fact_comentarios_producto c
            JOIN dim_producto p ON c.id_producto = p.id_producto
            JOIN dim_cliente cl ON c.id_cliente = cl.id_cliente
            WHERE c.comentario IS NOT NULL
        """))
        
        comentarios = []
        for row in result:
            comentarios.append({
                'id_comentario': row[0],
                'comentario': row[1],
                'calificacion': row[2],
                'sentimiento': row[3],
                'id_producto': row[4],
                'producto_nombre': row[5],
                'id_cliente': row[6],
                'cliente_nombre': row[7]
            })
        
        if not comentarios:
            print("No hay comentarios para vectorizar")
            return
        
        # Preparar datos para vectorización
        textos = [c['comentario'] for c in comentarios]
        metadatos = [{
            'tipo': 'comentario',
            'id_comentario': c['id_comentario'],
            'calificacion': c['calificacion'],
            'sentimiento': c['sentimiento'],
            'id_producto': c['id_producto'],
            'producto_nombre': c['producto_nombre'],
            'id_cliente': c['id_cliente'],
            'cliente_nombre': c['cliente_nombre']
        } for c in comentarios]
        
        # Añadir a la base de datos vectorial
        vector_ids = self.vector_db.add_documents(textos, metadatos)
        
        # Actualizar mapeos
        for i, c in enumerate(comentarios):
            vector_id = vector_ids[i]
            comentario_id = c['id_comentario']
            
            self.vector_to_dw_map['comment'][vector_id] = comentario_id
            self.dw_to_vector_map['comment'][comentario_id] = vector_id
            
            # Actualizar el ID vectorial en el data warehouse
            self.dw.session.execute(text(f"""
                UPDATE fact_comentarios_producto
                SET vector_id = {vector_id}
                WHERE id_comentario = {comentario_id}
            """))
        
        self.dw.session.commit()
        print(f"Vectorizados {len(comentarios)} comentarios de productos")
    
    def save(self):
        """Guarda el estado completo del sistema híbrido"""
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Guardar mapeos
        with open(os.path.join(self.base_dir, "mappings.json"), "w") as f:
            json.dump({
                'vector_to_dw_map': self.vector_to_dw_map,
                'dw_to_vector_map': self.dw_to_vector_map
            }, f)
        
        # Guardar base de datos vectorial
        self.vector_db.save(self.vector_dir)
        
        print(f"Sistema híbrido guardado en {self.base_dir}")
    
    @classmethod
    def load(cls, base_dir=BASE_DIR):
        """Carga el sistema híbrido desde el directorio especificado"""
        system = cls(base_dir)
        
        # Verificar que existen los directorios
        if not os.path.exists(os.path.join(base_dir, "mappings.json")):
            print("No se encontró un sistema guardado. Devolviendo instancia vacía.")
            return system
        
        # Cargar mapeos
        with open(os.path.join(base_dir, "mappings.json"), "r") as f:
            mappings = json.load(f)
            system.vector_to_dw_map = mappings['vector_to_dw_map']
            system.dw_to_vector_map = mappings['dw_to_vector_map']
        
        # Cargar base de datos vectorial
        system.vector_db = VectorDatabase.load(system.vector_dir)
        
        print(f"Sistema híbrido cargado desde {base_dir}")
        return system
    
    def hybrid_search(self, query, top_k=5):
        """
        Realiza una búsqueda híbrida que combina resultados de la base
        vectorial con datos estructurados del data warehouse.
        
        Args:
            query: Consulta de texto
            top_k: Número máximo de resultados
            
        Returns:
            Lista de resultados enriquecidos
        """
        # 1. Buscar en la base de datos vectorial
        vector_results = self.vector_db.search(query, top_k=top_k*2)  # Buscar el doble para filtrar
        
        # 2. Enriquecer resultados con datos del data warehouse
        hybrid_results = []
        
        for result in vector_results:
            metadata = result['metadata']
            tipo = metadata.get('tipo')
            
            enriched_result = {
                'texto': result['text'],
                'score': result['score'],
                'tipo': tipo
            }
            
            # Enriquecer según tipo de documento
            if tipo == 'producto':
                # Obtener datos del producto
                id_producto = metadata.get('id_producto')
                producto_row = self.dw.session.execute(text(f"""
                    SELECT p.nombre, p.categoria, p.subcategoria, p.precio_unitario,
                        COUNT(v.id_venta) as num_ventas,
                        SUM(v.cantidad) as unidades_vendidas,
                        AVG(v.precio_venta) as precio_promedio,
                        SUM(v.margen) as margen_total
                    FROM dim_producto p
                    LEFT JOIN fact_ventas v ON p.id_producto = v.id_producto
                    WHERE p.id_producto = {id_producto}
                    GROUP BY p.id_producto
                """)).fetchone()
                
                if producto_row:
                    enriched_result.update({
                        'nombre': producto_row[0],
                        'categoria': producto_row[1],
                        'subcategoria': producto_row[2],
                        'precio': producto_row[3],
                        'estadisticas_ventas': {
                            'num_ventas': producto_row[4] or 0,
                            'unidades_vendidas': producto_row[5] or 0,
                            'precio_promedio': producto_row[6] or 0,
                            'margen_total': producto_row[7] or 0
                        }
                    })
                
                # Obtener sentimiento promedio de comentarios
                sentimiento_row = self.dw.session.execute(text(f"""
                    SELECT 
                        AVG(c.calificacion) as calificacion_promedio,
                        AVG(c.sentimiento) as sentimiento_promedio,
                        COUNT(c.id_comentario) as num_comentarios
                    FROM fact_comentarios_producto c
                    WHERE c.id_producto = {id_producto}
                """)).fetchone()
                
                if sentimiento_row and sentimiento_row[2] > 0:
                    enriched_result['sentimiento'] = {
                        'calificacion_promedio': float(sentimiento_row[0]),
                        'sentimiento_promedio': float(sentimiento_row[1]),
                        'num_comentarios': sentimiento_row[2]
                    }
            
            elif tipo == 'cliente':
                # Obtener datos del cliente
                id_cliente = metadata.get('id_cliente')
                cliente_row = self.dw.session.execute(text(f"""
                    SELECT c.nombre, c.ciudad, c.region, c.segmento,
                        COUNT(v.id_venta) as num_compras,
                        SUM(v.cantidad * v.precio_venta) as gasto_total
                    FROM dim_cliente c
                    LEFT JOIN fact_ventas v ON c.id_cliente = v.id_cliente
                    WHERE c.id_cliente = {id_cliente}
                    GROUP BY c.id_cliente
                """)).fetchone()
                
                if cliente_row:
                    enriched_result.update({
                        'nombre': cliente_row[0],
                        'ciudad': cliente_row[1],
                        'region': cliente_row[2],
                        'segmento': cliente_row[3],
                        'estadisticas_compras': {
                            'num_compras': cliente_row[4] or 0,
                            'gasto_total': cliente_row[5] or 0
                        }
                    })
                
                # Obtener categorías más compradas
                categorias_row = self.dw.session.execute(text(f"""
                    SELECT 
                        p.categoria,
                        COUNT(v.id_venta) as num_compras
                    FROM fact_ventas v
                    JOIN dim_producto p ON v.id_producto = p.id_producto
                    WHERE v.id_cliente = {id_cliente}
                    GROUP BY p.categoria
                    ORDER BY num_compras DESC
                    LIMIT 3
                """)).fetchall()
                
                if categorias_row:
                    enriched_result['categorias_preferidas'] = [
                        {'categoria': row[0], 'num_compras': row[1]}
                        for row in categorias_row
                    ]
            
            elif tipo == 'comentario':
                # Obtener datos del comentario
                id_comentario = metadata.get('id_comentario')
                comentario_row = self.dw.session.execute(text(f"""
                    SELECT 
                        c.calificacion, c.sentimiento,
                        p.nombre as producto_nombre, p.categoria,
                        cl.nombre as cliente_nombre,
                        t.fecha
                    FROM fact_comentarios_producto c
                    JOIN dim_producto p ON c.id_producto = p.id_producto
                    JOIN dim_cliente cl ON c.id_cliente = cl.id_cliente
                    JOIN dim_tiempo t ON c.id_tiempo = t.id_tiempo
                    WHERE c.id_comentario = {id_comentario}
                """)).fetchone()
                
                if comentario_row:
                    enriched_result.update({
                        'calificacion': comentario_row[0],
                        'sentimiento': comentario_row[1],
                        'producto': {
                            'nombre': comentario_row[2],
                            'categoria': comentario_row[3]
                        },
                        'cliente': comentario_row[4],
                        'fecha': str(comentario_row[5])
                    })
            
            hybrid_results.append(enriched_result)
            
            # Limitar a top_k resultados
            if len(hybrid_results) >= top_k:
                break
        
        return hybrid_results
    

    def analyze_sentiment_by_category(self):
        """
        Análisis de sentimiento por categoría de producto.
        Combina datos estructurados con análisis semántico.
        """
        print("\nAnálisis de sentimiento por categoría:")
        
        # Consulta SQL para obtener sentimiento promedio por categoría
        result = pd.read_sql(text("""
            SELECT 
                p.categoria,
                AVG(c.calificacion) as calificacion_promedio,
                AVG(c.sentimiento) as sentimiento_promedio,
                COUNT(c.id_comentario) as num_comentarios
            FROM 
                fact_comentarios_producto c
            JOIN
                dim_producto p ON c.id_producto = p.id_producto
            GROUP BY 
                p.categoria
            ORDER BY 
                sentimiento_promedio DESC
        """), self.dw.engine)
        
        print(result)
        
        # Para cada categoría, encontrar comentarios representativos
        for categoria in result['categoria']:
            print(f"\nComentarios representativos para categoría: {categoria}")
            
            # Buscar comentarios positivos
            pos_query = f"comentarios positivos sobre productos de {categoria}"
            pos_results = self.hybrid_search(pos_query, top_k=2)
            
            print("  Comentarios positivos:")
            for i, res in enumerate(pos_results):
                if res['tipo'] == 'comentario' and res['score'] > 0.5:
                    print(f"    - {res['texto'][:100]}... (Score: {res['score']:.2f}, " + 
                        f"Sentimiento: {res['sentimiento']:.2f})")
            
            # Buscar comentarios negativos
            neg_query = f"comentarios negativos sobre productos de {categoria}"
            neg_results = self.hybrid_search(neg_query, top_k=2)
            
            print("  Comentarios negativos:")
            for i, res in enumerate(neg_results):
                if res['tipo'] == 'comentario' and res['score'] > 0.5:
                    print(f"    - {res['texto'][:100]}... (Score: {res['score']:.2f}, " + 
                        f"Sentimiento: {res['sentimiento']:.2f})")

    # In HybridAnalyticsSystem.find_similar_products method:
    def find_similar_products(self, product_id):
        """
        Encuentra productos similares basados en sus descripciones vectorizadas.
        
        Args:
            product_id: ID del producto en el data warehouse
            
        Returns:
            Lista de productos similares con datos enriquecidos
        """
        # Verificar que el producto existe y tiene vector_id
        product_row = self.dw.session.execute(text(f"""
            SELECT id_producto, nombre, descripcion, vector_id 
            FROM dim_producto 
            WHERE id_producto = {product_id}
        """)).fetchone()
        
        if not product_row or not product_row[3]:
            print(f"Producto con ID {product_id} no encontrado o no vectorizado")
            return []
        
        vector_id = product_row[3]
        desc = product_row[2]
        
        print(f"\nBuscando productos similares a: {product_row[1]}")
        print(f"Descripción: {desc[:100]}...")
        
        # Buscar productos similares usando la descripción como query
        similar_products = self.hybrid_search(desc, top_k=5)
        
        # Filtrar para incluir solo productos y excluir el producto original
        result = [p for p in similar_products 
                if p['tipo'] == 'producto' and p.get('nombre') != product_row[1]]
        
        return result

    # In HybridAnalyticsSystem.customer_insights method:
    def customer_insights(self, customer_id):
        """
        Genera insights sobre un cliente combinando datos estructurados
        con análisis semántico de sus comentarios.
        
        Args:
            customer_id: ID del cliente en el data warehouse
            
        Returns:
            Diccionario con insights del cliente
        """
        # Verificar que el cliente existe
        customer_row = self.dw.session.execute(text(f"""
            SELECT id_cliente, nombre, perfil, vector_id 
            FROM dim_cliente 
            WHERE id_cliente = {customer_id}
        """)).fetchone()
        
        if not customer_row:
            print(f"Cliente con ID {customer_id} no encontrado")
            return {}
        
        print(f"\nGenerando insights para cliente: {customer_row[1]}")
        
        # Datos básicos y comportamiento de compra
        basic_data = self.dw.session.execute(text(f"""
            SELECT 
                c.nombre, c.ciudad, c.region, c.segmento,
                COUNT(v.id_venta) as num_compras,
                SUM(v.cantidad * v.precio_venta) as gasto_total,
                AVG(v.cantidad * v.precio_venta) as ticket_promedio,
                MAX(t.fecha) as ultima_compra
            FROM dim_cliente c
            LEFT JOIN fact_ventas v ON c.id_cliente = v.id_cliente
            LEFT JOIN dim_tiempo t ON v.id_tiempo = t.id_tiempo
            WHERE c.id_cliente = {customer_id}
            GROUP BY c.id_cliente
        """)).fetchone()
        
        insights = {
            'cliente': customer_row[1],
            'perfil': customer_row[2],
            'estadisticas': {
                'ciudad': basic_data[1],
                'region': basic_data[2],
                'segmento': basic_data[3],
                'num_compras': basic_data[4] or 0,
                'gasto_total': float(basic_data[5] or 0),
                'ticket_promedio': float(basic_data[6] or 0),
                'ultima_compra': str(basic_data[7]) if basic_data[7] else None
            }
        }
        
        # Categorías preferidas
        cat_rows = self.dw.session.execute(text(f"""
            SELECT 
                p.categoria,
                COUNT(v.id_venta) as num_compras,
                SUM(v.cantidad) as unidades
            FROM fact_ventas v
            JOIN dim_producto p ON v.id_producto = p.id_producto
            WHERE v.id_cliente = {customer_id}
            GROUP BY p.categoria
            ORDER BY num_compras DESC
        """)).fetchall()
        
        insights['categorias_preferidas'] = [
            {'categoria': row[0], 'compras': row[1], 'unidades': row[2]}
            for row in cat_rows
        ]
        
        # Análisis de sentimiento en comentarios
        sent_row = self.dw.session.execute(text(f"""
            SELECT 
                AVG(c.calificacion) as calificacion_media,
                AVG(c.sentimiento) as sentimiento_medio,
                COUNT(c.id_comentario) as num_comentarios
            FROM fact_comentarios_producto c
            WHERE c.id_cliente = {customer_id}
        """)).fetchone()
        
        if sent_row and sent_row[2] > 0:
            insights['analisis_comentarios'] = {
                'calificacion_media': float(sent_row[0]),
                'sentimiento_medio': float(sent_row[1]),
                'num_comentarios': sent_row[2]
            }
            
            # Obtener comentarios representativos
            comentarios = self.dw.session.execute(text(f"""
                SELECT 
                    c.comentario, c.calificacion, c.sentimiento,
                    p.nombre as producto
                FROM fact_comentarios_producto c
                JOIN dim_producto p ON c.id_producto = p.id_producto
                WHERE c.id_cliente = {customer_id}
                ORDER BY c.calificacion DESC
            """)).fetchall()
            
            if comentarios:
                # Comentario más positivo
                insights['comentario_mas_positivo'] = {
                    'texto': comentarios[0][0],
                    'calificacion': comentarios[0][1],
                    'producto': comentarios[0][3]
                }
                
                # Comentario más negativo (si existe)
                if comentarios[-1][1] < 4:
                    insights['comentario_mas_negativo'] = {
                        'texto': comentarios[-1][0],
                        'calificacion': comentarios[-1][1],
                        'producto': comentarios[-1][3]
                    }
        
        # Clientes similares (si el cliente está vectorizado)
        if customer_row[3]:  # tiene vector_id
            perfil = customer_row[2]
            if perfil:
                print("Buscando clientes con perfil similar...")
                similar_results = self.hybrid_search(perfil, top_k=3)
                
                similar_customers = [
                    {
                        'nombre': r['nombre'],
                        'segmento': r['segmento'],
                        'ciudad': r['ciudad'],
                        'similaridad': r['score']
                    }
                    for r in similar_results
                    if r['tipo'] == 'cliente' and r['nombre'] != customer_row[1]
                ]
                
                insights['clientes_similares'] = similar_customers
        
        return insights

# ========================================================================
# FUNCIÓN PRINCIPAL Y APLICACIÓN DE EJEMPLO
# ========================================================================

def main():
    """Función principal que demuestra el sistema híbrido"""
    print("=" * 80)
    print("SISTEMA HÍBRIDO DE ANÁLISIS: DATA WAREHOUSE + BASE DE DATOS VECTORIAL")
    print("=" * 80)
    
    # Comprobar si ya existe un sistema guardado
    if os.path.exists(os.path.join(BASE_DIR, "mappings.json")):
        print("\nCargando sistema existente...")
        system = HybridAnalyticsSystem.load()
    else:
        print("\nInicializando nuevo sistema con datos de ejemplo...")
        system = HybridAnalyticsSystem()
        system.initialize_with_sample_data()
    
    # Menú de demostración
    while True:
        print("\n" + "=" * 50)
        print("MENÚ DE DEMOSTRACIÓN DEL SISTEMA HÍBRIDO")
        print("=" * 50)
        print("1. Ejecutar consultas tradicionales de Data Warehouse")
        print("2. Realizar búsqueda semántica")
        print("3. Análisis de sentimiento por categoría")
        print("4. Encontrar productos similares")
        print("5. Análisis de clientes")
        print("6. Salir")
        
        opcion = input("\nSeleccione una opción (1-6): ")
        
        if opcion == "1":
            # Consultas tradicionales del Data Warehouse
            system.dw.run_sample_queries()
            
        elif opcion == "2":
            # Búsqueda semántica
            query = input("\nIngrese su consulta de búsqueda: ")
            top_k = int(input("Número de resultados a mostrar: "))
            
            print(f"\nResultados para la consulta: '{query}'")
            results = system.hybrid_search(query, top_k)
            
            for i, res in enumerate(results):
                print(f"\nResultado {i+1} (Score: {res['score']:.4f}):")
                print(f"Tipo: {res['tipo']}")
                
                if res['tipo'] == 'producto':
                    print(f"Producto: {res['nombre']}")
                    print(f"Categoría: {res['categoria']} > {res.get('subcategoria', '')}")
                    print(f"Precio: ${res['precio']}")
                    if 'estadisticas_ventas' in res:
                        stats = res['estadisticas_ventas']
                        print(f"Ventas: {stats['num_ventas']} transacciones, {stats['unidades_vendidas']} unidades")
                        print(f"Margen total: ${stats['margen_total']:.2f}")
                    if 'sentimiento' in res:
                        sent = res['sentimiento']
                        print(f"Sentimiento: {sent['calificacion_promedio']:.1f}/5 ({sent['num_comentarios']} comentarios)")
                
                elif res['tipo'] == 'cliente':
                    print(f"Cliente: {res['nombre']}")
                    print(f"Segmento: {res['segmento']}")
                    print(f"Ubicación: {res['ciudad']}, {res['region']}")
                    if 'estadisticas_compras' in res:
                        stats = res['estadisticas_compras']
                        print(f"Compras: {stats['num_compras']} transacciones, ${stats['gasto_total']:.2f} total")
                    if 'categorias_preferidas' in res:
                        print("Categorías preferidas:", ", ".join([c['categoria'] for c in res['categorias_preferidas']]))
                
                elif res['tipo'] == 'comentario':
                    print(f"Comentario sobre: {res['producto']['nombre']} ({res['producto']['categoria']})")
                    print(f"Cliente: {res['cliente']}")
                    print(f"Calificación: {res['calificacion']}/5 (sentimiento: {res['sentimiento']:.2f})")
                    print(f"Fecha: {res['fecha']}")
                
                print(f"\nTexto: {res['texto']}")
        
        elif opcion == "3":
            # Análisis de sentimiento por categoría
            system.analyze_sentiment_by_category()
        
        elif opcion == "4":
            # Encontrar productos similares
            # Primero mostrar algunos productos como ejemplo
            productos = system.dw.session.execute("""
                SELECT id_producto, nombre, categoria
                FROM dim_producto
                LIMIT 10
            """).fetchall()
            
            print("\nEjemplos de productos disponibles:")
            for p in productos:
                print(f"ID: {p[0]} - {p[1]} ({p[2]})")
            
            product_id = int(input("\nIngrese el ID del producto para buscar similares: "))
            similar_products = system.find_similar_products(product_id)
            
            if similar_products:
                print("\nProductos similares encontrados:")
                for i, p in enumerate(similar_products):
                    print(f"\n{i+1}. {p['nombre']} (Score: {p['score']:.4f})")
                    print(f"   Categoría: {p['categoria']}")
                    print(f"   Precio: ${p['precio']}")
                    if 'estadisticas_ventas' in p:
                        stats = p['estadisticas_ventas']
                        print(f"   Ventas: {stats['unidades_vendidas']} unidades, ${stats['margen_total']:.2f} margen")
                    print(f"   Descripción: {p['texto'][:100]}...")
            else:
                print("\nNo se encontraron productos similares.")
        
        elif opcion == "5":
            # Análisis de clientes
            # Primero mostrar algunos clientes como ejemplo
            clientes = system.dw.session.execute(text("""
                SELECT id_cliente, nombre, segmento
                FROM dim_cliente
                LIMIT 10
            """)).fetchall()
            
            print("\nEjemplos de clientes disponibles:")
            for c in clientes:
                print(f"ID: {c[0]} - {c[1]} ({c[2]})")
            
            customer_id = int(input("\nIngrese el ID del cliente para análisis: "))
            insights = system.customer_insights(customer_id)
            
            if insights:
                print(f"\nInsights para el cliente: {insights['cliente']}")
                print(f"\nPerfil: {insights['perfil']}")
                
                print("\nEstadísticas de compra:")
                stats = insights['estadisticas']
                print(f"  Segmento: {stats['segmento']}")
                print(f"  Ubicación: {stats['ciudad']}, {stats['region']}")
                print(f"  Compras: {stats['num_compras']} transacciones")
                print(f"  Gasto total: ${stats['gasto_total']:.2f}")
                print(f"  Ticket promedio: ${stats['ticket_promedio']:.2f}")
                if stats['ultima_compra']:
                    print(f"  Última compra: {stats['ultima_compra']}")
                
                print("\nCategorías preferidas:")
                for cat in insights.get('categorias_preferidas', []):
                    print(f"  {cat['categoria']}: {cat['compras']} compras, {cat['unidades']} unidades")
                
                if 'analisis_comentarios' in insights:
                    com = insights['analisis_comentarios']
                    print(f"\nAnálisis de comentarios ({com['num_comentarios']} comentarios):")
                    print(f"  Calificación media: {com['calificacion_media']:.1f}/5")
                    print(f"  Sentimiento medio: {com['sentimiento_medio']:.2f}")
                    
                    if 'comentario_mas_positivo' in insights:
                        pos = insights['comentario_mas_positivo']
                        print(f"\nComentario más positivo ({pos['calificacion']}/5) sobre {pos['producto']}:")
                        print(f"  \"{pos['texto']}\"")
                    
                    if 'comentario_mas_negativo' in insights:
                        neg = insights['comentario_mas_negativo']
                        print(f"\nComentario más negativo ({neg['calificacion']}/5) sobre {neg['producto']}:")
                        print(f"  \"{neg['texto']}\"")
                
                if 'clientes_similares' in insights:
                    print("\nClientes con perfil similar:")
                    for cliente in insights['clientes_similares']:
                        print(f"  {cliente['nombre']} - {cliente['segmento']} en {cliente['ciudad']} " + 
                              f"(Similaridad: {cliente['similaridad']:.2f})")
            else:
                print("\nNo se encontraron datos para el cliente especificado.")
        
        elif opcion == "6":
            print("\nSaliendo del sistema...")
            break
        
        else:
            print("\nOpción no válida. Por favor, seleccione una opción del 1 al 6.")
        
        input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    main()