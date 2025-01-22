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

EMBEDDING_SIZE = 1536

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

# todo this is now in both context and match_records. it'd be weird to have the templates here but we do need to know the names if we're not able to partition
template_names = ["title", "artist", "album", "year", "language", "composite"]


def build_session_context(location="output/") -> df.SessionContext:
    ctx = df.SessionContext()
    ctx.register_parquet("records", f"{location}records/", schema=RECORD_SCHEMA)
    for template_name in template_names:
        ctx.register_parquet(
            f"templated_{template_name}",
            f"{location}templated/{template_name}/",
            schema=TEMPLATED_SCHEMA,
        )

    ctx.register_parquet("dedup", f"{location}dedup/", schema=DEDUP_SCHEMA)
    ctx.register_parquet("index_map", f"{location}index_map/", schema=INDEX_MAP_SCHEMA)
    ctx.register_parquet("vectors", f"{location}vectors/", schema=VECTORS_SCHEMA)
    ctx.register_parquet("ann", f"{location}ann/", schema=ANN_SCHEMA)

    return ctx
