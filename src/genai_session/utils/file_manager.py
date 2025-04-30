import mimetypes
from io import BytesIO
from typing import IO

import aiohttp
from genai_session.utils.exceptions import (
    FailedFileUploadException,
    FileNotFoundException,
    IncorrectFileInputException,
)


class FileManager:
    """
    Handles file upload, download, and metadata retrieval operations for a GenAI session.

    Attributes:
        api_base_url (str): Base URL of the file service API.
        session_id (str): ID representing the current session.
        request_id (str): ID representing the current request.
    """

    def __init__(self, api_base_url: str, session_id: str, request_id: str, jwt_token: str) -> None:
        self.session_id = session_id
        self.request_id = request_id
        self.file_service_url = api_base_url
        self.jwt_token = jwt_token

    async def save(self, content: bytes, filename: str) -> str:
        """
        Uploads a file to the file service.

        Args:
            content (bytes): The binary content of the file.
            filename (str): The name of the file to upload.

        Returns:
            str: The ID of the uploaded file.

        Raises:
            FailedFileUploadException: If the upload fails.
        """
        if not isinstance(content, bytes):
            raise IncorrectFileInputException("Content must be of type bytes.")

        data = aiohttp.FormData()
        mime_type = mimetypes.guess_type(url=filename)
        data.add_field(
            "file",
            content,
            filename=filename,
            content_type=mime_type[0] if mime_type else "application/octet-stream",
        )
        data.add_field("request_id", self.request_id)
        data.add_field("session_id", self.session_id)

        try:
            headers = {
                "Authorization": f"Bearer {self.jwt_token}"
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(f"{self.file_service_url}/files", data=data) as resp:
                    resp.raise_for_status()
                    json_resp = await resp.json()
                    return json_resp.get("id")
        except Exception as e:
            raise FailedFileUploadException(f"Failed to upload file: {e}")

    async def get_by_id(self, file_id: str) -> IO[bytes]:
        """
        Downloads a file from the file service using its ID.

        Args:
            file_id (str): The ID of the file to retrieve.

        Returns:
            IO[bytes]: A byte-stream of the file content.

        Raises:
            FileNotFoundException: If the file cannot be found or retrieved.
        """
        url = f"{self.file_service_url}/files/{file_id}"
        try:
            headers = {
                "Authorization": f"Bearer {self.jwt_token}"
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    content = await resp.read()
                    return BytesIO(content)
        except Exception as e:
            raise FileNotFoundException(f"Failed to retrieve file: {e}")

    async def get_metadata_by_id(self, file_id: str) -> dict[str, str]:
        """
        Retrieves metadata of a file by its ID.

        Args:
            file_id (str): The ID of the file whose metadata is to be fetched.

        Returns:
            dict[str, str]: A dictionary containing the file's metadata.

        Raises:
            FileNotFoundException: If the metadata cannot be retrieved.
        """
        url = f"{self.file_service_url}/files/{file_id}/metadata"
        try:
            headers = {
                "Authorization": f"Bearer {self.jwt_token}"
            }
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    content = await resp.json()
                    return content
        except Exception as e:
            raise FileNotFoundException(f"Failed to retrieve file: {e}")
