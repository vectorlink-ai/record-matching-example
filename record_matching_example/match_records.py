import datafusion as df
import pyarrow as pa
from vectorlink_py import template as tpl, dedup, embed, records
import sys
import torch

from vectorlink_gpu.ann import ANN
from vectorlink_gpu.datafusion import dataframe_to_tensor, tensor_to_arrow

"""
Matching process

1. Vectorize fields from records
2. Index vectors
3. Train classifier on record data or use LLM to classify
   a. If using LLM, find likely and unlikely matches using index
4. Use neighbors within threshold as filter for match candidates
5. Run trained classifier on records
"""


INPUT_CSV_PATH = "musicbrainz-20-A01.csv.dapo"

templates = {
    "title": "{{#if title}}title: {{title}}\n{{/if}}",
    "artist": "{{#if artist}}artist: {{artist}}\n{{/if}}",
    "album": "{{#if album}}album: {{album}}\n{{/if}}",
    "year": "{{#if year}}year: {{year}}\n{{/if}}",
    "language": "{{#if language}}language: {{language}}\n{{/if}}",
}

templates["composite"] = (
    f"{templates['title']}{templates['artist']}{templates['album']}{templates['year']}{templates['language']}"
)

INPUT_CSV_SCHEMA = pa.schema(
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


def eprintln(string):
    print(string, file=sys.stderr)


def ingest_csv():
    eprintln("ingesting csv to parquet...")
    sc = df.SessionConfig().with_batch_size(10)
    ctx = df.SessionContext(config=sc)
    dataframe = ctx.read_csv(
        INPUT_CSV_PATH, file_extension=".dapo", schema=INPUT_CSV_SCHEMA
    )

    dataframe.write_parquet("output/records/")


def template_records():
    sc = df.SessionConfig().with_batch_size(10)
    ctx = df.SessionContext(config=sc)
    dataframe = ctx.read_parquet("output/records/")

    eprintln("templating...")
    tpl.write_templated_fields(
        dataframe,
        templates,
        "output/templated/",
        id_column='"TID"',
        columns_of_interest=[
            "number",
            "title",
            "length",
            "artist",
            "album",
            "year",
            "language",
        ],
    )


def dedup_records():
    sc = df.SessionConfig().with_batch_size(10)
    ctx = df.SessionContext(config=sc)

    eprintln("dedupping...")
    for key in templates.keys():
        dedup.dedup_from_into(ctx, f"output/templated/{key}/", "output/dedup/")


def vectorize_records():
    sc = df.SessionConfig().with_batch_size(10)
    ctx = df.SessionContext(config=sc)

    eprintln("vectorizing...")
    embed.vectorize(ctx, "output/dedup/", "output/vectors/")


def average_fields():
    sc = df.SessionConfig().with_batch_size(10)
    ctx = df.SessionContext(config=sc)

    eprintln("averaging (for imputation)...")
    for key in templates.keys():
        records.write_field_averages(
            ctx, "output/templated", key, "output/vectors/", "output/vector_averages"
        )


def build_index_map():
    ctx = df.SessionContext()

    ctx.read_parquet("output/templated/composite").sort(df.col("hash")).select(
        (df.functions.row_number() - 1).alias("vector_id"),
        df.col('"TID"'),
        df.col("hash"),
    ).write_parquet("output/index_map/")


def load_vectors(ctx: df.SessionContext):
    index_map = ctx.read_parquet("output/index_map/")
    count = index_map.count()
    embeddings = (
        index_map.with_column_renamed("hash", "map_hash")
        .join(
            ctx.read_parquet("output/vectors/"),
            left_on="map_hash",
            right_on="hash",
            how="left",
        )
        .sort(df.col("vector_id"))
        .select("embedding")
    )

    vectors = torch.empty((count, 1536), dtype=torch.float32, device="cuda")
    dataframe_to_tensor(embeddings, vectors)

    return vectors


def index_field():
    ctx = df.SessionContext()
    vectors = load_vectors(ctx)

    ann = ANN(vectors, beam_size=32)
    print(ann.dump_logs())

    distances = tensor_to_arrow(ann.distances)
    beams = tensor_to_arrow(ann.beams)
    table = pa.Table.from_pydict({"beams": beams, "distances": distances})
    # todo this should not be plural names
    ctx.from_arrow_table(table).select(
        (df.functions.row_number() - 1).alias("vector_id"),
        df.col("beams"),
        df.col("distances"),
    ).write_parquet("output/ann/")


def load_ann():
    ctx = df.SessionContext()

    print("loading vectors..")
    vectors = load_vectors(ctx)
    ann_data = ctx.read_parquet("output/ann/")
    count = ann_data.count()
    beams = torch.empty((count, 32), dtype=torch.int32, device="cuda")
    distances = torch.empty((count, 32), dtype=torch.float32, device="cuda")
    print("loading beams..")
    dataframe_to_tensor(ann_data.select(df.col("beams")), beams)
    print("loading distances..")
    dataframe_to_tensor(ann_data.select(df.col("distances")), distances)

    print("loading ann..")
    ann = ANN(vectors, beams=beams, distances=distances, beam_size=32)
    print(ann.dump_logs())


def main():
    ingest_csv()
    template_records()
    dedup_records()
    vectorize_records()
    average_fields()
    build_index_map()


if __name__ == "__main__":
    main()
