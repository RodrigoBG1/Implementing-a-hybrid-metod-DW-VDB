# Implementación de una Base de Datos Vectorial Simple con Python
# Este ejemplo usa FAISS para la indexación vectorial y Sentence Transformers para generar embeddings

import numpy as np
import pandas as pd
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
import time
from typing import List, Dict, Tuple, Optional, Union

class VectorDatabase:
    """
    Una implementación simple de base de datos vectorial usando FAISS para indexación
    y Sentence Transformers para la generación de embeddings.
    """
    
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
        self.index = faiss.IndexFlatL2(dimension)  # Índice L2 (distancia euclidiana)
        
        # Almacenamiento para metadatos
        self.documents = []
        self.document_embeddings = []
        self.id_to_index_map = {}
        self.next_id = 0
        
        print(f"Base de datos vectorial inicializada con modelo: {embedding_model_name}")
        print(f"Dimensión de vectores: {dimension}")
    
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
    
    def delete_document(self, doc_id: int) -> bool:
        """
        Elimina un documento de la base de datos.
        
        Nota: FAISS no admite eliminaciones, por lo que esta implementación
        reconstruye el índice. Para una base de datos real, se podrían usar
        otras estrategias como marcar como eliminados.
        
        Args:
            doc_id: ID del documento a eliminar
            
        Returns:
            True si el documento fue eliminado, False si no se encontró
        """
        if doc_id not in self.id_to_index_map:
            return False
            
        idx = self.id_to_index_map[doc_id]
        
        # Eliminar documento
        del self.documents[idx]
        del self.document_embeddings[idx]
        
        # Actualizar mapeo de IDs
        self.id_to_index_map = {}
        for i, doc in enumerate(self.documents):
            self.id_to_index_map[doc["id"]] = i
            
        # Reconstruir índice
        self.index = faiss.IndexFlatL2(self.dimension)
        if self.document_embeddings:
            self.index.add(np.array(self.document_embeddings).astype('float32'))
            
        print(f"Documento con ID {doc_id} eliminado. Total de documentos: {len(self.documents)}")
        return True
    
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
            
        print(f"Base de datos guardada en {directory}")
        
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
        
        print(f"Base de datos cargada desde {directory} con {len(db.documents)} documentos")
        return db

# Ejemplo de uso
def ejemplo_completo():
    """Ejemplo completo del uso de la base de datos vectorial"""
    
    # Datos de ejemplo (artículos de noticias)
    articulos = [
        "El presidente anuncia nuevas medidas económicas para impulsar el crecimiento",
        "Científicos descubren un nuevo tratamiento contra el cáncer",
        "El equipo local gana el campeonato nacional de fútbol",
        "Nueva exposición de arte contemporáneo llega al museo de la ciudad",
        "Innovaciones tecnológicas revolucionan la industria automotriz",
        "Estudio revela beneficios de la meditación para la salud mental",
        "Aumenta la preocupación por el cambio climático tras recientes catástrofes",
        "El banco central decide mantener las tasas de interés sin cambios",
        "Reconocido chef abre nuevo restaurante con concepto innovador",
        "Investigadores desarrollan inteligencia artificial capaz de predecir enfermedades"
    ]
    
    # Metadatos asociados a cada artículo
    metadatos = [
        {"categoria": "Política", "fecha": "2023-03-15", "fuente": "Diario Nacional"},
        {"categoria": "Ciencia", "fecha": "2023-03-12", "fuente": "Revista Científica"},
        {"categoria": "Deportes", "fecha": "2023-03-14", "fuente": "Deportes Hoy"},
        {"categoria": "Cultura", "fecha": "2023-03-10", "fuente": "Arte & Cultura"},
        {"categoria": "Tecnología", "fecha": "2023-03-08", "fuente": "Tech News"},
        {"categoria": "Salud", "fecha": "2023-03-05", "fuente": "Salud al Día"},
        {"categoria": "Medio Ambiente", "fecha": "2023-03-11", "fuente": "Planeta Verde"},
        {"categoria": "Economía", "fecha": "2023-03-16", "fuente": "Economía Global"},
        {"categoria": "Gastronomía", "fecha": "2023-03-09", "fuente": "Guía Gourmet"},
        {"categoria": "Tecnología", "fecha": "2023-03-07", "fuente": "Innovación Diaria"}
    ]
    
    # Inicializar la base de datos vectorial
    print("Inicializando base de datos vectorial...")
    db = VectorDatabase()
    
    # Añadir documentos
    print("\nAñadiendo documentos...")
    ids = db.add_documents(articulos, metadatos)
    
    # Realizar búsquedas de ejemplo
    print("\nRealizando búsquedas de ejemplo:")
    
    # Búsqueda relacionada con tecnología
    query1 = "Avances tecnológicos e innovación"
    print(f"\nBúsqueda: '{query1}'")
    resultados1 = db.search(query1, top_k=3)
    for i, res in enumerate(resultados1):
        print(f"  {i+1}. [{res['metadata']['categoria']}] {res['text'][:50]}... (Score: {res['score']:.4f})")
    
    # Búsqueda relacionada con salud
    query2 = "Mejoras en la salud y tratamientos médicos"
    print(f"\nBúsqueda: '{query2}'")
    resultados2 = db.search(query2, top_k=3)
    for i, res in enumerate(resultados2):
        print(f"  {i+1}. [{res['metadata']['categoria']}] {res['text'][:50]}... (Score: {res['score']:.4f})")
    
    # Búsqueda relacionada con economía
    query3 = "Economía y finanzas"
    print(f"\nBúsqueda: '{query3}'")
    resultados3 = db.search(query3, top_k=3)
    for i, res in enumerate(resultados3):
        print(f"  {i+1}. [{res['metadata']['categoria']}] {res['text'][:50]}... (Score: {res['score']:.4f})")
    
    # Eliminar un documento
    doc_id_to_delete = ids[2]  # Eliminar el artículo de deportes
    print(f"\nEliminando documento con ID {doc_id_to_delete}...")
    db.delete_document(doc_id_to_delete)
    
    # Búsqueda después de eliminar
    query4 = "Deportes y competiciones"
    print(f"\nBúsqueda después de eliminar: '{query4}'")
    resultados4 = db.search(query4, top_k=3)
    for i, res in enumerate(resultados4):
        print(f"  {i+1}. [{res['metadata']['categoria']}] {res['text'][:50]}... (Score: {res['score']:.4f})")
    
    # Guardar la base de datos
    print("\nGuardando base de datos...")
    db.save("./vector_db_demo")
    
    # Cargar la base de datos
    print("\nCargando base de datos desde disco...")
    db_cargada = VectorDatabase.load("./vector_db_demo")
    
    # Verificar que funciona correctamente después de cargar
    query5 = "Arte y cultura"
    print(f"\nBúsqueda después de cargar: '{query5}'")
    resultados5 = db_cargada.search(query5, top_k=3)
    for i, res in enumerate(resultados5):
        print(f"  {i+1}. [{res['metadata']['categoria']}] {res['text'][:50]}... (Score: {res['score']:.4f})")
    
    print("\nEjemplo completo finalizado")

# Aplicación práctica: Sistema de búsqueda de documentos
def aplicacion_busqueda_documentos():
    """
    Aplicación práctica que simula un sistema de búsqueda de documentos
    utilizando la base de datos vectorial.
    """
    print("=== SISTEMA DE BÚSQUEDA SEMÁNTICA DE DOCUMENTOS ===")
    
    # Inicializar base de datos
    db = VectorDatabase()
    
    # Datos de ejemplo (más extensos)
    documentos = [
        "Los algoritmos de aprendizaje profundo han revolucionado el campo de la inteligencia artificial, permitiendo avances significativos en reconocimiento de imágenes, procesamiento del lenguaje natural y sistemas de recomendación.",
        "El calentamiento global es un fenómeno causado principalmente por la emisión de gases de efecto invernadero. Los científicos advierten que, sin medidas urgentes, las consecuencias para los ecosistemas serán devastadoras.",
        "La arquitectura sostenible busca minimizar el impacto ambiental de los edificios mediante el uso eficiente de energía, materiales ecológicos y diseños que se integran con el entorno natural.",
        "Los mercados financieros experimentaron una volatilidad significativa durante la pandemia, con sectores como la tecnología y la salud registrando ganancias históricas mientras otros como el turismo y la hostelería sufrieron pérdidas récord.",
        "La nutrición personalizada es una tendencia creciente que utiliza datos genéticos y biomarcadores para crear planes alimenticios adaptados a las necesidades específicas de cada individuo.",
        "El blockchain no solo ha permitido el surgimiento de criptomonedas como Bitcoin, sino que también tiene aplicaciones en cadenas de suministro, contratos inteligentes y sistemas de votación segura.",
        "La realidad virtual está transformando sectores como la educación, permitiendo experiencias inmersivas que mejoran la retención de conocimientos y facilitan el aprendizaje de habilidades complejas.",
        "Los avances en medicina regenerativa ofrecen esperanza para tratar enfermedades degenerativas mediante el uso de células madre y terapias génicas que pueden reparar tejidos dañados.",
        "El comercio electrónico ha experimentado un crecimiento exponencial, cambiando los hábitos de consumo y obligando a las tiendas físicas a reinventarse para ofrecer experiencias únicas que no pueden replicarse online.",
        "La agricultura vertical utiliza tecnología para cultivar alimentos en capas apiladas verticalmente, reduciendo la necesidad de tierra y agua mientras permite la producción local de alimentos en entornos urbanos.",
        "Los sistemas de energía renovable como la solar y eólica están alcanzando paridad de red en muchas regiones, convirtiéndose en alternativas económicamente viables a los combustibles fósiles.",
        "La psicología positiva estudia los factores que contribuyen al bienestar y la felicidad, centrándose en fortalecer las cualidades positivas en lugar de solo tratar trastornos mentales.",
        "El desarrollo de ciudades inteligentes integra tecnologías IoT para optimizar servicios urbanos, reducir el consumo de recursos y mejorar la calidad de vida de sus habitantes.",
        "Las técnicas modernas de análisis de datos permiten a las empresas predecir tendencias de consumo y personalizar sus ofertas, aumentando la satisfacción del cliente y la eficiencia operativa.",
        "La impresión 3D está revolucionando la manufactura, permitiendo la producción personalizada y reduciendo el desperdicio de materiales en industrias que van desde la aeroespacial hasta la medicina."
    ]
    
    # Metadatos
    metadatos = [
        {"categoria": "Tecnología", "autor": "A. Johnson", "fecha": "2023-01-15"},
        {"categoria": "Medio Ambiente", "autor": "B. Smith", "fecha": "2023-01-20"},
        {"categoria": "Arquitectura", "autor": "C. Davis", "fecha": "2023-01-25"},
        {"categoria": "Economía", "autor": "D. Wilson", "fecha": "2023-02-01"},
        {"categoria": "Salud", "autor": "E. Brown", "fecha": "2023-02-05"},
        {"categoria": "Tecnología", "autor": "F. Taylor", "fecha": "2023-02-10"},
        {"categoria": "Educación", "autor": "G. Miller", "fecha": "2023-02-15"},
        {"categoria": "Medicina", "autor": "H. Anderson", "fecha": "2023-02-20"},
        {"categoria": "Negocios", "autor": "I. Thomas", "fecha": "2023-03-01"},
        {"categoria": "Agricultura", "autor": "J. White", "fecha": "2023-03-05"},
        {"categoria": "Energía", "autor": "K. Harris", "fecha": "2023-03-10"},
        {"categoria": "Psicología", "autor": "L. Martin", "fecha": "2023-03-15"},
        {"categoria": "Urbanismo", "autor": "M. Thompson", "fecha": "2023-03-20"},
        {"categoria": "Análisis de Datos", "autor": "N. Garcia", "fecha": "2023-03-25"},
        {"categoria": "Manufactura", "autor": "O. Rodriguez", "fecha": "2023-04-01"}
    ]
    
    # Añadir documentos
    print("\nCargando documentos en la base de datos...")
    db.add_documents(documentos, metadatos)
    
    # Interfaz de usuario simple
    print("\nBase de datos lista. Puede realizar búsquedas semánticas.")
    
    while True:
        print("\n" + "-"*50)
        consulta = input("Ingrese su consulta (o 'salir' para terminar): ")
        
        if consulta.lower() == 'salir':
            break
            
        # Realizar búsqueda
        resultados = db.search(consulta, top_k=5)
        
        # Mostrar resultados
        if resultados:
            print(f"\nResultados más relevantes para: '{consulta}'")
            for i, res in enumerate(resultados):
                print(f"\n{i+1}. [{res['metadata']['categoria']}] - Autor: {res['metadata']['autor']}")
                print(f"   {res['text']}")
                print(f"   Relevancia: {res['score']:.4f}")
        else:
            print("\nNo se encontraron resultados para su consulta.")
            
    print("\nGracias por utilizar el sistema de búsqueda semántica.")

# Programa principal
if __name__ == "__main__":
    print("Seleccione una opción:")
    print("1. Ejecutar ejemplo básico")
    print("2. Ejecutar aplicación de búsqueda de documentos")
    
    opcion = input("Opción (1/2): ")
    
    if opcion == "1":
        ejemplo_completo()
    elif opcion == "2":
        aplicacion_busqueda_documentos()
    else:
        print("Opción no válida")