"""
init_milvus.py - One-Time Milvus Vector Database Initialization

Creates the 'motion_corrections' collection in the local Milvus instance.
This collection stores historical motion correction rules as 384-dim embeddings
so the pipeline's retrieve_context_node (graph.py) can query for past failures
and apply learned fixes during motion planning.

Prerequisites:
  - Milvus running via Docker Compose (docker-compose.yml) on localhost:19530
  - pip install pymilvus

Run once after first deploying Milvus:
    python init_milvus.py
"""

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

connections.connect("default", host="localhost", port="19530")

if not utility.has_collection("motion_corrections"):
    # Collection schema: auto-ID primary key, 384-dim float vector for semantic
    # similarity search, and a text field storing the human-readable correction rule.
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
    ]

    schema = CollectionSchema(fields, description="Stores motion correction rules")
    collection = Collection("motion_corrections", schema)

    # IVF_FLAT index with L2 distance for fast approximate nearest-neighbor search
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
    collection.create_index(field_name="embedding", index_params=index_params)

    print("Success! The 'motion_corrections' collection is ready.")
else:
    print("Collection already exists!")