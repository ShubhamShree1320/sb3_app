import json
import logging

import boto3
from botocore.exceptions import ClientError
from langchain_core.documents import Document

from sb3_api.enums.document import DocumentType
from sb3_api.settings import ServiceSettings

logger = logging.getLogger(__name__)


class S3Loader:
    def __init__(self, settings: ServiceSettings) -> None:
        self.bucket = settings.S3_BUCKET
        self.acronyms_key = settings.ACRONYMS_KEY
        self.kpi_prefix = settings.KPI_PREFIX
        self.table_prefix = settings.TABLE_PREFIX
        self.query_prefix = settings.QUERY_PREFIX

        self.prefix_map: dict[DocumentType, str] = {
            DocumentType.KPI: self.kpi_prefix,
            DocumentType.TABLE: self.table_prefix,
            DocumentType.QUERY: self.query_prefix,
        }

        self.s3_client = boto3.client("s3")

    def list_s3_files(self, prefix: str, extension: str = "") -> list[str]:
        keys = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith(extension):
                        keys.append(key)
        except ClientError:
            logger.exception("Failed to list objects for prefix {prefix}.", extra={"prefix": prefix})
        return keys

    def read_s3_file(self, key: str, max_size_mb: int = 10, encoding: str = "utf-8") -> str:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)

            # Check file size
            content_length = response.get("ContentLength", 0)
            max_size_bytes = max_size_mb * 1024 * 1024
            if content_length > max_size_bytes:
                logger.error("File %s exceeds size limit (%d MB)", key, max_size_mb)
                return ""

            return response["Body"].read().decode(encoding)
        except ClientError:
            logger.exception("Failed to read object {key}.", extra={"key": key})
            return ""

    def read_s3_text_file(self, key: str) -> str:
        return self.read_s3_file(key)

    def read_s3_json_file(self, key: str, max_size_mb: int = 10) -> list[dict]:
        try:
            content = self.read_s3_file(key, max_size_mb)
            if not content:
                return []

            data = json.loads(content)

            # Validate it's a list
            if not isinstance(data, list):
                logger.error("JSON file %s is not a list", key)
                return []
            logger.info("Successfully read JSON file %s", key)
            return data  # noqa: TRY300
        except json.JSONDecodeError:
            logger.exception("Failed to parse JSON from {key}.", extra={"key": key})
            return []

    def load_documents_from_s3(self, prefix: str) -> list[Document]:
        documents = []
        files = self.list_s3_files(prefix)
        if prefix != self.query_prefix:
            for file in files:
                content = self.read_s3_text_file(file)
                if content.strip():
                    documents.append(Document(page_content=content, metadata={"source": file}))
        else:
            for file in files:
                content = self.read_s3_json_file(file)  # type: ignore[assignment]
                # json file content (question-sql pair) is processed and validated in processor.
                documents.append(
                    Document(
                        page_content="",
                        metadata={"source": file, "query_pairs": content},
                    )
                )
        return documents
