import os
from pathlib import Path
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from config import ASSET_PATH, get_logger, enable_telemetry
from search_resumes import search_resumes
from azure.ai.inference.prompts import PromptTemplate
import json

# initialize logging and tracing objects
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# create a project client using environment variables loaded from the .env file
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# create a chat client we can use for testing
chat_client = project.inference.get_chat_completions_client()

from azure.ai.inference.prompts import PromptTemplate

# Parameter messages: e.g. [{"role": "user", "content": "I need to hire a new data scientist, what would you recommend?"}
@tracer.start_as_current_span(name="chat_with_resumes")
def chat_with_resumes(messages: list, context: dict = None) -> dict:
    if context is None:
        context = {}

    try:
        documents = search_resumes(messages, context)

        # do a grounded chat call using the search results
        grounded_chat_prompt = PromptTemplate.from_prompty(Path(ASSET_PATH) / "grounded_chat.prompty")

        system_message = grounded_chat_prompt.create_messages(documents=documents, context=context)

        logger.info(f" system_message: \n{system_message}")
        logger.info(f" messages: \n{messages}")
        
        
        try:
            response = chat_client.complete(
                model=os.environ["CHAT_MODEL"],
                messages=system_message + messages,
                **grounded_chat_prompt.parameters,
            )
        except project.exceptions.HttpResponseError as http_error:
            logger.error(f"[chat_with_resumes] HTTP response error occurred: {http_error}")
            raise
        except project.exceptions.ServiceRequestError as request_error:
            logger.error(f"[chat_with_resumes]Service request error occurred: {request_error}")
            raise
        except Exception as e:
            logger.error(f"[chat_with_resumes] An unexpected error occurred during the HTTP request: {e}")
            raise
        
        logger.info(f"[chat_with_resumes] 💬 Response: \n {json.dumps(response.choices[0].message.as_dict(), indent=4, ensure_ascii=False).replace('\\n', '\n')}")

        # Validate if the response is valid JSON
        try:
            json_response = response.choices[0].message.as_dict()
            json.dumps(json_response)  # Ensure it can be serialized to JSON
        except (TypeError, ValueError) as e:
            logger.error(f"[chat_with_resumes] Invalid JSON response: {e}")
            raise ValueError("[chat_with_resumes] The response is not a valid JSON format") from e
        
        # Return a chat protocol compliant response
        return response

    except KeyError as e:
        logger.error(f"[chat_with_resumes] Missing key in context or environment variables: {e}")
        raise
    except Exception as e:
        logger.error(f"[chat_with_resumes] An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    import argparse

    # load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        help="Query to use to search product",
        default="I need to hire a new data scientist, what would you recommend?",
    )
    parser.add_argument(
        "--enable-telemetry",
        action="store_true",
        help="Enable sending telemetry back to the project",
    )
    args = parser.parse_args()
    if args.enable_telemetry:
        enable_telemetry(True)

    # run chat with products
    response = chat_with_resumes(messages=[{"role": "user", "content": args.query}])