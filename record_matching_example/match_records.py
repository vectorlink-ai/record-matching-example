import datafusion as df
import pyarrow as pa
from vectorlink_py import template as tpl, dedup, embed, records
import sys
import torch
import scipy
from openai import OpenAI
import re
import argparse

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


def get_vectors_dataframe(ctx: df.SessionContext) -> df.DataFrame:
    index_map = ctx.read_parquet("output/index_map/")
    return index_map.with_column_renamed("hash", "map_hash").join(
        ctx.read_parquet("output/vectors/"),
        left_on="map_hash",
        right_on="hash",
        how="left",
    )


def load_vectors(ctx: df.SessionContext) -> torch.Tensor:
    embeddings = (
        get_vectors_dataframe(ctx).sort(df.col("vector_id")).select(df.col("embedding"))
    )
    count = embeddings.count()

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
    return ann


def discover_training_set():
    """
    1. Loads the existing ANN
    2. Finds the first peak of the first derivative
    3. Keeping a tally to balance, guesses on the left or right of the first peak threshold
    4. Sends records (in toto) to LLM
    4. Writes each match or non-match in a training table
    """

    # 1.
    ann = load_ann()
    distances = ann.distances
    (vector_count, beam_size) = distances.size()

    # 2.
    sample_size = 1000
    sample_size = min(vector_count, sample_size)
    all_distances = distances[0:sample_size].flatten().sort()
    (length,) = all_distances.size()
    tail = all_distances[1:length]
    head = all_distances[0 : length - 1]
    diff = tail - head
    # maybe use smoothing (savitzky_golay?) for smoothing first to remove jitter?
    (peaks, _) = scipy.signal.find_peaks(diff.numpy)
    # Assume first peak is good for now
    first_peak = peaks[0]
    threshold = all_distances[first_peak]

    # 3.
    ctx = df.SessionContext()
    candidate_size = 1000  # increase for better training
    same = 0
    different = 0
    record = []
    candidates = []
    for i in range(0, candidate_size):
        beam = ann.beams[i]
        distance = ann.distances[i]
        indices = (distance > threshold).nonzero()
        (count, _) = indices.size()
        if indices == 0:
            continue
        pivot = indices[0][0]
        total = same + different
        if same_count > total / 2 and pivot < len(distance):
            last = len(beam) - 1
            j = beam[last]
        else:
            j = beam[first]

        (answer, id1, id2) = ask_oracle(ctx, i, j)
        if answer == True:
            same += 1
        else:
            different += 1

        record = {"match": answer, "left": id1, "right": id2}
        candidates.append(record)
    candidates_pd = pd.DataFrame(candidates)
    ctx.from_pandas(candidates_pd).write_parquet("output/training_set/")


def get_record_from_vid(ctx, vid) -> str:
    templated_df = ctx.read_parquet("output/templated/composite").select(
        df.col("templated"), df.col('"TID"').alias("tid")
    )
    result = (
        ctx.read_parquet("output/index_map/")
        .filter(vid == df.col("vector_id"))
        .join(templated_df, left_on='"TID"', right_on="tid", how="inner")
        .select(df.col('"TID"'), df.col("templated"))
        .limit(1)
    )
    return result.to_pylist()[0]


def check_y_or_n(string):
    if re.search(r".*[Y|y]es\W*$", string, re.MULTILINE) is None:
        return False
    else:
        return True


def ask_oracle(ctx, vid1, vid2):
    """
    1. Map from vid to record id
    2. load record 1 and 2
    3. Ask LLM for match of record 1 and 2
    """
    record1 = get_record_from_vid(ctx, vid1)
    record2 = get_record_from_vid(ctx, vid2)

    subject = "pieces of music"
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a classifier deciding if two songs are a match or not.",
            },
            {
                "role": "user",
                "content": f"""Tell me whether the following two records are referring to the same entity or a different entity using a chain of reasoning followed by a single yes or no answer on a single line, without any formatting.

1:  {record1['templated']}

2:  {record2['templated']}
""",
            },
        ],
    )
    content = completion.choices[0].message.content
    print(content)
    return (
        check_y_or_n(completion.choices[0].message.content),
        record1['"TID"'],
        record2['"TID"'],
    )


def load_ann() -> ANN:
    ctx = df.SessionContext()

    print("loading vectors..")
    vectors = load_vectors(ctx)
    ann_data = ctx.read_parquet("output/ann/")

    vectors_dataframe = get_vectors_dataframe(ctx)
    combined_dataframe = vectors_dataframe.with_column_renamed(
        "vector_id", "v_vector_id"
    ).join(ann_data, left_on="v_vector_id", right_on="vector_id", how="left")
    print("loading ann..")
    ann = ANN.load_from_dataframe(combined_dataframe)

    print(ann.dump_logs())
    return ann


def train_weights():
    ctx = SessionContext()
    keys = ["__INTERCEPT__"]
    weights = [0.0]
    records = ctx.read_parquet("output/records/")

    for key in templates.keys():
        keys.append(key)
        weights.append(1.0)
        field = ctx.read_parquet("output/templated/{key}").select(
            df.col('"TID"'), df.col("hash"), df.col("templated").alias(key)
        )

    keys = list(d.keys())
    candidates = ctx.read_parquet("output/training_set/")
    templates = ctx.read_parquet("output/index_map/")
    vectors = ctx.read_parquet("output/vectors/")
    # left = candidates.join(records, how="left", left_on="left", right_on='"TID"').join(index_map,


def search():
    parser = argparse.ArgumentParser(usage="search [query] [options]")
    parser.add_argument("query", help="The query to search for")
    args = parser.parse_args()
    ann = load_ann()
    ctx = df.SessionContext()
    result = (
        ann_search(ctx, ann, args.query)
        .sort(df.col("distance"))
        .select(
            df.col("distance"),
            df.col("title"),
            df.col("artist"),
            df.col("album"),
            df.col("year"),
            df.col("language"),
            df.col("length"),
        )
    )

    result.show()


def ann_search(ctx: df.SessionContext, ann: ANN, query_string: str) -> df.DataFrame:
    client = OpenAI()

    response = client.embeddings.create(
        input=query_string, model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    query_tensor = torch.tensor(embedding, dtype=torch.float32, device="cuda").reshape(
        (1, 1536)
    )

    result = ann.search(query_tensor)
    matches = pa.array(result.indices.flatten().cpu().numpy())
    distances = pa.array(result.distances.flatten().cpu().numpy())
    results = ctx.from_arrow(
        pa.RecordBatch.from_arrays([matches, distances], ["match", "distance"])
    )

    records = ctx.read_parquet("output/records/")
    index_map = ctx.read_parquet("output/index_map")

    return (
        results.join(index_map, left_on="match", right_on="vector_id")
        .with_column_renamed('"TID"', "match_tid")
        .join(records, left_on="match_tid", right_on='"TID"')
    )


def main():
    ingest_csv()
    template_records()
    dedup_records()
    vectorize_records()
    average_fields()
    build_index_map()
    index_field()
    find_together_and_apart()
    train_weights()


if __name__ == "__main__":
    main()
