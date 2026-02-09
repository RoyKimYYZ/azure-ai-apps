### Setup sample resources for embedding chunks

from openai import AzureOpenAI
from azure.identity import get_bearer_token_provider
from config import azure_openai_key, azure_openai_embedding_deployment_id, azure_openai_endpoint, search_endpoint, search_credential, blob_connection_string, blob_container
import os

from azure.storage.blob import BlobServiceClient

def open_blob_client():
    # Set max_block_size and max_single_put_size due to large PDF transfers
    # See https://learn.microsoft.com/azure/storage/blobs/storage-blobs-tune-upload-download-python
    if not blob_connection_string.startswith("ResourceId"):
        return BlobServiceClient.from_connection_string(
            blob_connection_string,
            max_block_size=1024*1024*8, # 8 MiB
            max_single_put_size=1024*1024*8 # 8 MiB
        )
    return BlobServiceClient(
        account_url=blob_account_url,
        credential=DefaultAzureCredential(),
        max_block_size=1024*1024*8, # 8 MiB
        max_single_put_size=1024*1024*8 # 8 MiB
    )

blob_client = open_blob_client()
container_client = blob_client.get_container_client(blob_container)
if not container_client.exists():
    container_client.create_container()

file_path = os.path.join("resume-pdfs", "resume-1.pdf")
file_path = os.path.join("resume-pdfs", "resume-2.pdf")
resume_id = "2";
print(f'file_path: ' + file_path)

blob_name = os.path.basename(file_path)
print(f'blob_name: ' + blob_name)

blob_client = container_client.get_blob_client(blob_name)
if not blob_client.exists():
    with open(file_path, "rb") as f:
        print(f'blob_name: ' + blob_name)
        blob_client.upload_blob(data=f, overwrite=True, metadata={"index_id": resume_id, "filename": blob_name})