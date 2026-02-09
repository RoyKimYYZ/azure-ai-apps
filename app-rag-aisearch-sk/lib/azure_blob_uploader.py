from dotenv import load_dotenv
import os
from azure.storage.blob import BlobServiceClient, BlobClient
from pathlib import Path
load_dotenv()
secret_key = os.getenv("blobaccesskey")
print(secret_key)

account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
container_name = os.getenv("AZURE_CONTAINER_NAME")
connection_string = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={account_name};"
    f"AccountKey={account_key};"
    f"EndpointSuffix=core.windows.net"
)



local_folder_path = Path("Resumes")

# Loop through all files in the folder
for file_path in local_folder_path.glob("*.*"):  
    if file_path.is_file():
        blob_name = f"{local_folder_path.name}/{file_path.name}"  # optional subfolder in blob

        try:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            print(f"Uploaded '{file_path}' to blob '{blob_name}'.")

        except Exception as e:
            print(f" Error uploading '{file_path}': {e}")