from enum import StrEnum
from pathlib import Path

import psycopg2

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            database=self.db_config["database"],
            user=self.db_config["user"],
            password=self.db_config["password"],
        )

    # TODO:
    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and bool of the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be casted to vector ({embeddings}::vector)
    def process_text_file(
        self,
        file_name: str,
        chunk_size: int = 500,
        overlap: int = 50,
        dimensions: int = 1536,
        truncate_table: bool = False,
    ):
        content = Path(file_name).read_text(encoding="utf-8")
        chunks = chunk_text(content, chunk_size, overlap)
        embeddings_dict = self.embeddings_client.get_embeddings(chunks, dimensions)
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                if truncate_table:
                    cursor.execute("TRUNCATE TABLE vectors;")
                for index, chunk in enumerate(chunks):
                    embedding = embeddings_dict[index]
                    cursor.execute(
                        "INSERT INTO vectors (text, embedding) VALUES (%s, %s::vector);",
                        (chunk, embedding),
                    )
            conn.commit()

    # TODO:
    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`
    def search(
        self,
        search_mode: SearchMode,
        user_request: str,
        top_k: int = 5,
        min_score_threshold: float = 0.5,
        dimensions: int = 1536,
    ) -> list[str]:
        embeddings_dict = self.embeddings_client.get_embeddings([user_request], dimensions)
        query_embedding = embeddings_dict[0]

        distance_operator = "<=>" if search_mode == SearchMode.COSINE_DISTANCE else "<->"

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT text, embedding {distance_operator} %s::vector AS distance
                    FROM vectors
                    WHERE embedding {distance_operator} %s::vector < %s
                    ORDER BY distance
                    LIMIT %s;
                    """,
                    (query_embedding, query_embedding, min_score_threshold, top_k),
                )
                results = cursor.fetchall()
                context_texts = [row[0] for row in results]
        return context_texts
