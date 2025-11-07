"""
Vector store building and population.
"""
import logging
from qdrant_client import models
from qdrant_client.models import PointStruct, PayloadSchemaType

from .chunk_generator import generate_table_ingestion_chunks
from .utils import stable_id


def feed_vector_store(dico_api, qdrant_client, embeddings, collection_name):
    """
    Feed mode: Fetch DICO data, create chunks, and populate vector store.
    
    Args:
        dico_api: DicoAPI instance
        qdrant_client: QdrantClient instance
        embeddings: OpenAIEmbeddings instance
        collection_name: Name of the Qdrant collection
        
    Returns:
        bool: True if successful, False otherwise
    """
    logging.info("=== FEED MODE: Building Vector Store ===")
    
    logging.info("Fetching database schema from DICO API...")
    dico_data = dico_api.fetch_database_schema()
    if not dico_data:
        logging.error("Failed to fetch DICO data")
        return False
    
    logging.info("Generating table chunks...")
    table_chunks = generate_table_ingestion_chunks(dico_data)
    logging.info(f"Generated {len(table_chunks)} table chunks for ingestion.")

    existing_collections = [col.name for col in qdrant_client.get_collections().collections]
    if collection_name not in existing_collections: 
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(embeddings.embed_query("Hello world")),
                distance=models.Distance.COSINE,
            ),
        )
        logging.info(f"Created new collection: {collection_name}")
    else: 
        logging.info(f"Using existing collection: {collection_name}")

    logging.info("Creating embeddings and points...")
    
    table_points = []
    for chunk in table_chunks:
        chunk_id = stable_id(
            chunk.metadata['chunk_type'],
            chunk.metadata['primary_table'],
            chunk.metadata['table_code']
        )

        embedding = embeddings.embed_query(chunk.page_content)

        point = PointStruct(
            id=chunk_id,
            vector=embedding,
            payload={
                'content': chunk.page_content,
                'chunk_type': chunk.metadata['chunk_type'],
                'primary_table': chunk.metadata['primary_table'],
                'table_code': chunk.metadata['table_code'],
                'table_kind': chunk.metadata['table_kind'],
                'field_count': chunk.metadata['field_count'],
                'metadata': chunk.metadata
            }
        )
        table_points.append(point)

    try:
        result = qdrant_client.upsert(
            collection_name=collection_name, 
            points=table_points
        )
        logging.info(f"Successfully upserted {len(table_points)} table chunks")
        collection_info = qdrant_client.get_collection(collection_name)
        logging.info(f"Collection now contains {collection_info.points_count} points.")
    except Exception as e:
        logging.error(f"Error during upsert: {str(e)}")
        return False

    try:
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="chunk_type",
            field_schema=PayloadSchemaType.KEYWORD
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="primary_table",
            field_schema=PayloadSchemaType.KEYWORD
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="table_kind",
            field_schema=PayloadSchemaType.KEYWORD
        )
        logging.info('Payload indexes created successfully.')
    except Exception as e:
        logging.error(f"Error creating payload indexes: {str(e)}")
    
    logging.info("=== FEED MODE COMPLETED SUCCESSFULLY ===")
    return True
