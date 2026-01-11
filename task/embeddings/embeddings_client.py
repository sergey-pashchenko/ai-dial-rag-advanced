from collections.abc import Iterable

import requests

DIAL_EMBEDDINGS = "https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings"


# TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)
#
# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }


class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str):
        self._deployment_name = deployment_name
        self._api_key = api_key

    def get_embeddings(self, input_texts: Iterable[str], dimensions: int = 1536) -> dict[int, list[float]]:
        response = requests.post(
            url=DIAL_EMBEDDINGS.format(model=self._deployment_name),
            headers={"Api-Key": self._api_key},
            json={"input": input_texts, "dimensions": dimensions},
        )
        response.raise_for_status()
        response_json = response.json()

        embeddings_dict = {item["index"]: item["embedding"] for item in response_json.get("data", [])}
        return embeddings_dict
