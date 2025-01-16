import datafusion as df
import pyarrow as pa
from vectorlink_py import template as tpl, dedup, embed, records
import sys

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

templates[
    "composite"
] = f"{templates['title']}{templates['artist']}{templates['album']}{templates['year']}{templates['language']}"

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


def main():
    sc = df.SessionConfig().with_batch_size(10)
    ctx = df.SessionContext(config=sc)
    dataframe = ctx.read_csv(
        INPUT_CSV_PATH, file_extension=".dapo", schema=INPUT_CSV_SCHEMA
    )

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

    eprintln("dedupping...")
    for key in templates.keys():
        dedup.dedup_from_into(ctx, f"output/templated/{key}/", "output/dedup/")

    eprintln("vectorizing...")
    embed.vectorize(ctx, "output/dedup/", "output/vectors/")

    eprintln("averaging (for imputation)...")
    for key in templates.keys():
        records.write_field_averages(
            ctx, "output/templated", key, "output/vectors/", "output/vector_averages"
        )

    eprintln("index field...")


if __name__ == "__main__":
    main()
