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
        session_id (str): Identifier of the current session.
        request_id (str): Identifier of the current request.
        jwt_token (str): JWT token used for authenticated requests.
    """

    def __init__(self, api_base_url: str, session_id: str, request_id: str, jwt_token: str) -> None:
        """
        Initializes a FileManager instance.

        Args:
            api_base_url (str): Base URL of the file service API.
            session_id (str): ID of the current session.
            request_id (str): ID of the current request.
            jwt_token (str): JWT token used for authentication.
        """
        self.session_id = session_id
        self.request_id = request_id
        self.file_service_url = api_base_url
        self.jwt_token = jwt_token

    async def save(self, content: bytes, filename: str) -> str:
        """
        Uploads a file to the file service.

        Args:
            content (bytes): Binary content of the file.
            filename (str): Name of the file to upload.

        Returns:
            str: ID of the uploaded file.

        Raises:
            IncorrectFileInputException: If content is not of type bytes.
            FailedFileUploadException: If the file upload fails.
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        elif not isinstance(content, bytes):
            raise IncorrectFileInputException("Content must be of type bytes.")

        data = aiohttp.FormData()

        try:
            mime_type = mimetypes.guess_type(url=filename)[0]
        except IndexError:
            mime_type = None

        data.add_field(
            "file",
            content,
            filename=filename,
            content_type=mime_type or "application/octet-stream",
        )
        data.add_field("request_id", self.request_id)
        data.add_field("session_id", self.session_id)

        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(f"{self.file_service_url}/files", data=data) as resp:
                    resp.raise_for_status()
                    json_resp = await resp.json()
                    return json_resp.get("id")
        except Exception as e:
            raise FailedFileUploadException(f"Failed to upload file: {e}")

    async def get_by_id(self, file_id: str) -> IO[bytes]:
        """
        Downloads a file from the file service by its ID.

        Args:
            file_id (str): ID of the file to download.

        Returns:
            IO[bytes]: A byte-stream of the downloaded file content.

        Raises:
            FileNotFoundException: If the file cannot be found or retrieved.
        """
        url = f"{self.file_service_url}/files/{file_id}"
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    content = await resp.read()
                    return BytesIO(content)
        except Exception as e:
            raise FileNotFoundException(f"Failed to retrieve file: {e}")

    async def get_metadata_by_id(self, file_id: str) -> dict[str, str]:
        """
        Retrieves metadata for a file by its ID.

        Args:
            file_id (str): ID of the file whose metadata is being requested.

        Returns:
            dict[str, str]: A dictionary containing file metadata.

        Raises:
            FileNotFoundException: If the metadata cannot be found or retrieved.
        """
        url = f"{self.file_service_url}/files/{file_id}/metadata"
        try:
            headers = {"Authorization": f"Bearer {self.jwt_token}"}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    content = await resp.json()
                    return content
        except Exception as e:
            raise FileNotFoundException(f"Failed to retrieve metadata: {e}")
