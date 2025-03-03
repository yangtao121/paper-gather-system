import os
import json
from typing import Any, Dict, Generator, List, Optional, Union
from requests import Session, Response
from requests.adapters import HTTPAdapter
from requests.auth import AuthBase, HTTPBasicAuth
from urllib3.util.retry import Retry


class PaperlessAPIError(Exception):
    """Base exception for Paperless API errors"""

    def __init__(self, message: str, response: Optional[Response] = None):
        super().__init__(message)
        self.response = response


class PaperlessClient:
    def __init__(
        self,
        base_url: str,
        auth: Union[AuthBase, str, tuple] = None,
        api_version: int = 7,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize Paperless-ngx API client
        
        :param base_url: Base URL of Paperless instance (e.g., "http://localhost:8000")
        :param auth: Authentication method. Can be:
            - requests.auth.AuthBase instance
            - API token string
            - (username, password) tuple for basic auth or token generation
        :param api_version: API version to use
        :param timeout: Default request timeout in seconds
        :param max_retries: Maximum number of retries for failed requests
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.api_version = api_version

        # Configure session with retries
        self.session = Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=['GET', 'POST', 'PUT', 'DELETE']
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retry))
        self.session.mount('http://', HTTPAdapter(max_retries=retry))

        # Set authentication
        self._setup_auth(auth)

        # Set default headers
        self.session.headers.update({
            'Accept': f'application/json; version={self.api_version}',
            'Content-Type': 'application/json'
        })

    def _setup_auth(self, auth: Union[AuthBase, str, tuple]) -> None:
        """Configure authentication based on provided credentials"""
        if isinstance(auth, AuthBase):
            self.session.auth = auth
        elif isinstance(auth, str):
            self.session.headers['Authorization'] = f'Token {auth}'
        elif isinstance(auth, tuple) and len(auth) == 2:
            # Auto-detect authentication type
            try:
                # First try token authentication
                self._get_auth_token(auth[0], auth[1])
            except PaperlessAPIError:
                # Fall back to basic auth
                self.session.auth = HTTPBasicAuth(auth[0], auth[1])
        elif auth is not None:
            raise ValueError("Invalid authentication credentials")

    def _get_auth_token(self, username: str, password: str) -> None:
        """Obtain and set API token using credentials"""
        token_url = f"{self.base_url}/api/token/"
        response = self._request('POST', token_url, data={
                                 'username': username, 'password': password})
        self.session.headers['Authorization'] = f'Token {response.json()["token"]}'

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Union[Dict, List]] = None,
        files: Optional[Dict] = None
    ) -> Response:
        """
        Execute API request with error handling
        
        :raises PaperlessAPIError: For API errors
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data if not files else None,
                data=data if files else None,
                files=files,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response
        except Exception as e:
            error_msg = f"API request failed: {str(e)}"
            if isinstance(e, requests.HTTPError):
                try:
                    error_detail = response.json()
                    error_msg += f" | Details: {error_detail}"
                except json.JSONDecodeError:
                    error_msg += f" | Response: {response.text[:200]}"
            raise PaperlessAPIError(
                error_msg, getattr(e, 'response', None)) from e

    # Document Operations ------------------------------------------------------

    def upload_document(
        self,
        file_path: str,
        title: Optional[str] = None,
        created: Optional[str] = None,
        correspondent_id: Optional[int] = None,
        document_type_id: Optional[int] = None,
        storage_path_id: Optional[int] = None,
        tag_ids: Optional[List[int]] = None,
        archive_serial_number: Optional[int] = None,
        custom_field_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Upload a document to Paperless
        
        :param file_path: Path to document file
        :param metadata: Additional document metadata
        :return: Consumption task information
        """
        endpoint = "/api/documents/post_document/"
        metadata = {
            'title': title,
            'created': created,
            'correspondent': correspondent_id,
            'document_type': document_type_id,
            'storage_path': storage_path_id,
            'tags': tag_ids,
            'archive_serial_number': archive_serial_number,
            'custom_fields': custom_field_ids
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        with open(file_path, 'rb') as f:
            files = {'document': f}
            response = self._request(
                'POST', endpoint, data=metadata, files=files)

        return response.json()

    def get_documents(
        self,
        query: Optional[str] = None,
        page: int = 1,
        page_size: int = 100,
        filter_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get paginated list of documents with optional filtering
        
        :param query: Full text search query
        :param page: Page number
        :param page_size: Items per page (max 1000)
        :param filter_params: Additional filter parameters (e.g., {'correspondent__id': 1})
        :return: Dictionary with results and pagination info
        """
        endpoint = "/api/documents/"
        params = {
            'query': query,
            'page': page,
            'page_size': min(page_size, 1000),
            **(filter_params or {})
        }
        response = self._request('GET', endpoint, params=params)
        return response.json()

    def iterate_all_documents(
        self,
        query: Optional[str] = None,
        page_size: int = 100,
        filter_params: Optional[Dict] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that yields all documents matching criteria
        
        Usage:
            for doc in client.iterate_all_documents(query="invoice"):
                process_document(doc)
        """
        page = 1
        while True:
            result = self.get_documents(
                query=query,
                page=page,
                page_size=page_size,
                filter_params=filter_params
            )
            for doc in result['results']:
                yield doc
            if not result['next']:
                break
            page += 1

    # Bulk Operations ----------------------------------------------------------

    def bulk_edit_documents(
        self,
        document_ids: List[int],
        operation: str,
        parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Perform bulk operation on documents
        
        :param document_ids: List of document IDs to affect
        :param operation: One of supported operations (set_correspondent, add_tag, etc.)
        :param parameters: Operation-specific parameters
        :return: Operation result
        """
        VALID_OPERATIONS = {
            'set_correspondent', 'set_document_type', 'set_storage_path',
            'add_tag', 'remove_tag', 'modify_tags', 'delete', 'reprocess',
            'set_permissions', 'merge', 'split', 'rotate', 'delete_pages',
            'modify_custom_fields'
        }

        if operation not in VALID_OPERATIONS:
            raise ValueError(
                f"Invalid operation. Valid options: {VALID_OPERATIONS}")

        endpoint = "/api/documents/bulk_edit/"
        payload = {
            "documents": document_ids,
            "method": operation,
            "parameters": parameters or {}
        }
        response = self._request('POST', endpoint, data=payload)
        return response.json()

    # Custom Fields ------------------------------------------------------------

    def filter_by_custom_field(
        self,
        field_name: str,
        operator: str,
        value: Union[str, List, bool]
    ) -> List[Dict[str, Any]]:
        """
        Filter documents using custom field query
        
        :param field_name: Name of custom field
        :param operator: Query operator (exact, contains, range, etc.)
        :param value: Query value
        :return: List of matching documents
        """
        query = json.dumps([field_name, operator, value])
        result = self._request('GET', "/api/documents/",
                               params={'custom_field_query': query})
        return result.json()['results']

    # Task Management ----------------------------------------------------------

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of an asynchronous task"""
        response = self._request('GET', "/api/tasks/",
                                 params={'task_id': task_id})
        return response.json()

# Helper function for environment-based configuration


def from_environment() -> PaperlessClient:
    """Create client using environment variables"""
    return PaperlessClient(
        base_url=os.environ['PAPERLESS_URL'],
        auth=(
            os.environ.get('PAPERLESS_USERNAME'),
            os.environ.get('PAPERLESS_PASSWORD')
        ),
        api_version=int(os.environ.get('PAPERLESS_API_VERSION', 7))
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Environment-configured client
    client = from_environment()

    # Example 2: Upload document with metadata
    try:
        task = client.upload_document(
            file_path="/path/to/invoice.pdf",
            title="2023 Annual Invoice",
            correspondent_id=5,
            tag_ids=[1, 3],
            document_type_id=2
        )
        print(f"Document consumption started. Task ID: {task['task_id']}")
    except PaperlessAPIError as e:
        print(f"Upload failed: {str(e)}")

    # Example 3: Process all documents matching a query
    for doc in client.iterate_all_documents(query="financial", page_size=50):
        print(f"Processing document {doc['id']}: {doc['title']}")
        if doc['correspondent'] is None:
            client.bulk_edit_documents(
                document_ids=[doc['id']],
                operation='set_correspondent',
                parameters={'correspondent': 5}
            )

    # Example 4: Complex custom field query
    projects = client.filter_by_custom_field(
        field_name="Project",
        operator="in",
        value=["Alpha", "Beta"]
    )
    print(f"Found {len(projects)} documents in target projects")
