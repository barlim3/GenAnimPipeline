"""
init_milvus.py - Milvus Vector Database Initialisation (Hybrid Multi-Tenant)

Rebuilds the 'motion_corrections' collection with a hybrid multi-tenant schema:
  - Physical partitions isolate broad project categories (ProjectA, ProjectB,
    Global_Rules) for coarse-grained routing and parallel search.
  - A 'style' scalar field enables fine-grained metadata filtering within a
    partition (e.g. style == 'combat' or style == 'physics').

Schema
------
  id        INT64 (primary, auto-ID)
  embedding FLOAT_VECTOR (384-dim, all-MiniLM-L6-v2)
  text      VARCHAR (max 1000 chars) -- human-readable correction rule
  style     VARCHAR (max 100 chars)  -- animation style tag

Partitions
----------
  ProjectA      -- corrections scoped to Project A animations
  ProjectB      -- corrections scoped to Project B animations
  Global_Rules  -- universal physics / kinematics rules shared across projects

Prerequisites:
  - Milvus running via Docker Compose (docker-compose.yml) on localhost:19530
  - pip install pymilvus

WARNING: This script drops and recreates the collection. All existing data
will be lost. Run once during initial setup or when resetting the database.

    python init_milvus.py
"""

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

connections.connect("default", host="localhost", port="19530")

# Drop existing collection so we always start from a clean, correct schema.
if utility.has_collection("motion_corrections"):
    utility.drop_collection("motion_corrections")
    print("Existing 'motion_corrections' collection dropped.")

# ── Schema ────────────────────────────────────────────────────────────────────
fields = [
    FieldSchema(name="id",        dtype=DataType.INT64,         is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,  dim=384),
    FieldSchema(name="text",      dtype=DataType.VARCHAR,        max_length=1000),
    FieldSchema(name="style",     dtype=DataType.VARCHAR,        max_length=100),
]

schema = CollectionSchema(fields, description="Hybrid multi-tenant motion correction rules")
collection = Collection("motion_corrections", schema)
print("Collection 'motion_corrections' created with hybrid schema.")

# ── Physical Partitions ───────────────────────────────────────────────────────
for partition in ("ProjectA", "ProjectB", "Global_Rules"):
    collection.create_partition(partition)
    print(f"  Partition '{partition}' created.")

# ── Index ─────────────────────────────────────────────────────────────────────
# IVF_FLAT with L2 distance for fast approximate nearest-neighbour search.
index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
collection.create_index(field_name="embedding", index_params=index_params)
print("IVF_FLAT index applied to 'embedding' field.")

print("\nSuccess! Hybrid multi-tenant 'motion_corrections' collection is ready.")
