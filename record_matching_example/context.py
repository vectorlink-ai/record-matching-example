import datafusion as df
import pyarrow as pa

RECORD_SCHEMA = pa.schema(
    [
        pa.field("TID", pa.int64(), nullable=False),
        pa.field("CID", pa.int64(), nullable=False),
        pa.field("CTID", pa.int64(), nullable=False),
        pa.field("SourceID", pa.int64(), nullable=False),
        pa.field("id", pa.string(), nullable=True),
        pa.field("number", pa.string(), nullable=True),
        pa.field("title", pa.string(), nullable=True),
        pa.field("length", pa.string(), nullable=True),
        pa.field("artist", pa.string(), nullable=True),
        pa.field("album", pa.string(), nullable=True),
        pa.field("year", pa.string(), nullable=True),
        pa.field("language", pa.string(), nullable=True),
    ]
)

TEMPLATED_SCHEMA = pa.schema(
    [
        pa.field("TID", pa.int64(), nullable=False),
        pa.field("templated", pa.string_view(), nullable=False),
        pa.field("hash", pa.string_view(), nullable=False),
    ]
)

DEDUP_SCHEMA = pa.schema(
    [
        pa.field("hash", pa.string_view(), nullable=False),
        pa.field("templated", pa.string_view(), nullable=False),
    ]
)

INDEX_MAP_SCHEMA = pa.schema(
    [
        pa.field("vector_id", pa.int64(), nullable=False),
        pa.field("TID", pa.int64(), nullable=False),
        pa.field("hash", pa.string_view(), nullable=False),
    ]
)

MODEL = "text-embedding-3-large"
EMBEDDING_SIZE = 3072  # 1536

VECTORS_SCHEMA = pa.schema(
    [
        pa.field("hash", pa.string_view(), nullable=False),
        pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_SIZE), nullable=False),
    ]
)

ANN_SCHEMA = pa.schema(
    [
        pa.field("vector_id", pa.int64(), nullable=False),
        pa.field("beams", pa.list_(pa.int32()), nullable=False),
        pa.field("distances", pa.list_(pa.float32()), nullable=False),
    ]
)

TRAINING_SET_SCHEMA = pa.schema(
    [
        pa.field("left", pa.int64(), nullable=False),
        pa.field("right", pa.int64(), nullable=False),
        pa.field("match", pa.bool_(), nullable=False),
    ]
)

FIELD_DISTANCES_SCHEMA = pa.schema(
    [
        pa.field("left_tid", pa.int64(), nullable=False),
        pa.field("right_tid", pa.int64(), nullable=False),
        pa.field("key", pa.string_view(), nullable=False),
        pa.field("distance", pa.float32(), nullable=False),
    ]
)

WEIGHTS_SCHEMA = pa.schema(
    [
        pa.field("intercept", pa.float32(), nullable=False),
        pa.field("album", pa.float32(), nullable=False),
        pa.field("artist", pa.float32(), nullable=False),
        pa.field("composite", pa.float32(), nullable=False),
        pa.field("language", pa.float32(), nullable=False),
        pa.field("length", pa.float32(), nullable=False),
        pa.field("number", pa.float32(), nullable=False),
        pa.field("title", pa.float32(), nullable=False),
        pa.field("year", pa.float32(), nullable=False),
    ]
)

FILTERED_SCHEMA = pa.schema(
    [
        pa.field("left", pa.int64(), nullable=False),
        pa.field("right", pa.int64(), nullable=False),
    ]
)

PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("left", pa.int64(), nullable=False),
        pa.field("right", pa.int64(), nullable=False),
        pa.field("prediction", pa.float32(), nullable=False),
    ]
)

CLUSTERS_SCHEMA = pa.schema(
    [
        pa.field("cluster_id", pa.int64(), nullable=False),
        pa.field("cluster_element", pa.int64(), nullable=False),
    ]
)

VECTOR_AVERAGES_SCHEMA = pa.schema(
    [
        pa.field("template", pa.string_view(), nullable=False),
        pa.field("average", pa.list_(pa.float32(), EMBEDDING_SIZE), nullable=False),
    ]
)


def build_session_context(location="output/") -> df.SessionContext:
    ctx = df.SessionContext()
    ctx.register_parquet("records", f"{location}records/", schema=RECORD_SCHEMA)
    ctx.register_parquet(
        f"templated",
        f"{location}templated/",
        schema=TEMPLATED_SCHEMA,
        table_partition_cols=[("key", "string")],
    )

    ctx.register_parquet("dedup", f"{location}dedup/", schema=DEDUP_SCHEMA)
    ctx.register_parquet("index_map", f"{location}index_map/", schema=INDEX_MAP_SCHEMA)
    ctx.register_parquet("vectors", f"{location}vectors/", schema=VECTORS_SCHEMA)
    ctx.register_parquet("ann", f"{location}ann/", schema=ANN_SCHEMA)
    ctx.register_parquet(
        "training_set", f"{location}training_set/", schema=TRAINING_SET_SCHEMA
    )
    ctx.register_parquet(
        "field_distances",
        f"{location}field_distances/",
        schema=FIELD_DISTANCES_SCHEMA,
    )
    ctx.register_parquet(
        "match_field_distances",
        f"{location}match_field_distances/",
        schema=FIELD_DISTANCES_SCHEMA,
    )

    ctx.register_parquet("weights", f"{location}weights/", schema=WEIGHTS_SCHEMA)

    ctx.register_parquet("filtered", f"{location}filtered/", schema=FILTERED_SCHEMA)

    ctx.register_parquet(
        "prediction", f"{location}prediction/", schema=PREDICTION_SCHEMA
    )

    ctx.register_parquet("clusters", f"{location}clusters/", schema=CLUSTERS_SCHEMA)

    ctx.register_parquet(
        "vector_averages", f"{location}vector_averages/", schema=VECTOR_AVERAGES_SCHEMA
    )

    ctx.sql(
        'CREATE VIEW templated_vectors AS SELECT templated."TID",templated.key,templated.templated,vectors.embedding FROM templated LEFT JOIN vectors ON (templated.hash = vectors.hash)'
    )

    ctx.sql(
        'CREATE VIEW index_vectors AS SELECT index_map.vector_id,index_map."TID", vectors.embedding FROM index_map LEFT JOIN vectors ON (index_map.hash = vectors.hash)'
    )

    ctx.sql(
        'CREATE VIEW total_ann AS SELECT index_vectors.vector_id,index_vectors."TID", index_vectors.embedding, ann.beams, ann.distances FROM index_vectors LEFT JOIN ann ON (index_vectors.vector_id = ann.vector_id)'
    )

    return ctx
