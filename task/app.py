# TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope

from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import SearchMode
from task.embeddings.text_processor import TextProcessor
from task.models.message import Message
from task.models.role import Role

SYSTEM_PROMPT = """You are a RAG powered assistant.

# User message structure:
- `RAG CONTEXT` - Retrieved documents relevant to the query.
- `USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering `USER QUESTION`.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

# TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """## RAG CONTEXT:
{context}


## USER QUESTION:
{query}
"""

# TODO:
# - create embeddings client with 'text-embedding-3-small-1' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)

embeddings_client = DialEmbeddingsClient(
    deployment_name="text-embedding-3-small-1",
    api_key=API_KEY,
)
chat_client = DialChatCompletionClient(
    deployment_name="gpt-4o",
    api_key=API_KEY,
)
text_processor = TextProcessor(
    embeddings_client=embeddings_client,
    db_config={
        "host": "localhost",
        "port": 5433,
        "database": "vectordb",
        "user": "postgres",
        "password": "postgres",
    },
)


def run_console_chat():
    print("Welcome to the RAG-powered console chat! Type 'exit' to quit.")

    text_processor.process_text_file(
        file_name="task/embeddings/microwave_manual.txt",
        # chunk_size=500,
        # overlap=50,
        # dimensions=1536,
        truncate_table=True,
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        # Retrieve context from the database based on user input
        context = text_processor.search(
            search_mode=SearchMode.COSINE_DISTANCE,
            user_request=user_input,
            top_k=3,
            min_score_threshold=0.5,
            dimensions=1536,
        )

        # Prepare the augmented user prompt
        augmented_prompt = USER_PROMPT.format(context=context, query=user_input)

        # Create messages for chat completion
        messages = [
            Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
            Message(role=Role.USER, content=augmented_prompt),
        ]

        # Get completion from chat client
        response_message = chat_client.get_completion(messages, print_request=False)

        # Display the AI's response
        print(f"AI: {response_message.content}\n")


if __name__ == "__main__":
    # TODO:
    #  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
    #  RUN docker-compose.yml

    run_console_chat()
