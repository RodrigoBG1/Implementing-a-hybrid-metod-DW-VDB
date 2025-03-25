# Implementing-a-hybrid-metod-DW-VDB
This is a implementation between a data warehouse and a vectorial database, looking to hav e the best part of both the analitics from the DW and the semantic seach in the VDB.                                         
# How it works:                                                
For this research we implement the three diferent methods of managuing the data                    
# 1. Data warehouse:
      This code implements a basic data warehouse using Python, SQLAlchemy and pandas. It creates a star schema data warehouse using SQLite as the 
      database backend (though it's noted that PostgreSQL or MySQL could be used instead). The schema consists of three dimension tables (DimTiempo for 
      time, DimProducto for products, and DimCliente for customers) and one fact table (FactVentas for sales transactions). The implementation includes
      functions to generate sample data for all tables, with around 5,000 random sales transactions spanning a two-year period. It also demonstrates
      analytical capabilities by executing four example SQL queries that analyze sales by product category, monthly sales trends, most profitable products,
      and sales by customer segment and region. Finally, the code includes basic data visualization using matplotlib to generate bar charts and line graphs of sales data.
# 1. Vector data base:
    This code implements a simple vector database system using Python, leveraging FAISS for vector indexing and Sentence Transformers for generating text embeddings.
    It creates a VectorDatabase class that allows users to add documents with associated metadata, search for semantically similar content using vector similarity, 
    delete documents, and save/load the database from disk. The implementation uses the L2 (Euclidean distance) metric for similarity searching and includes a complete 
    example with news articles and a practical application for semantic document search. The system converts text into numerical vector representations (embeddings) 
    using a pretrained language model ('paraphrase-MiniLM-L6-v2'), which enables searching for conceptually related content rather than just keyword matching. The code 
    demonstrates both a basic usage flow and a more interactive document search application that lets users perform semantic queries against a collection of documents.
# 1. Hybrid implementation:
    This code implements a hybrid analytics system in Python that combines a traditional data warehouse with a vector database for enhanced data analysis. It integrates 
    structured data processing (using SQLAlchemy and SQLite) with semantic search capabilities (using FAISS and Sentence Transformers). The system creates a star schema 
    data warehouse with dimension tables for time, products, and customers, plus fact tables for sales and product comments. It then enhances this model by vectorizing 
    product descriptions, customer profiles, and product comments, allowing for semantic similarity searches. The implementation includes functionality to generate sample 
    data, perform traditional SQL-based analytics, conduct semantic searches, analyze sentiment by product category, find similar products based on descriptions, and generate
    customer insights. The code demonstrates how combining traditional structured data analysis with vector-based semantic analysis can provide richer insights by bridging 
    the gap between structured and unstructured data.

# Requirments to run tue code:
 - Python 3.12 installed 
 - For this implementations as mention before are some libraries need it, use the next command in the cmd to install them in case you don have it 
                pip install faiss-cpu numpy pandas sentence-transformers
    
     
