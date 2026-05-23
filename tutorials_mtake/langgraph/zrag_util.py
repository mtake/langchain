import json
from typing import override, Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class ZRAGRetriever(BaseRetriever):
    """Retriever for Z-related documents"""
    query_dict: dict[str, list[Document]] = {}

    @classmethod
    def from_dict(cls, query_dict: dict[str, list[Document]], **kwargs: Any) -> "ZRAGRetriever":
        zrag_retriever = cls()
        zrag_retriever.query_dict.update(query_dict)
        return zrag_retriever

    @classmethod
    def from_json(cls, input_file: str, **kwargs: Any) -> "ZRAGRetriever":
        query_dict = convert_json_to_dict(input_file)
        return cls.from_dict(query_dict, **kwargs)

    def add_dict(self, query_dict: dict[str, list[Document]], **kwargs: Any):
        self.query_dict.update(query_dict)

    def add_query(self, query: str, documents: list[Document], **kwargs: Any):
        self.query_dict[query] = documents

    @override
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.query_dict.get(query, [])


def convert_json_to_dict(input_file: str) -> dict[str, list[Document]]:
    query_dict: dict[str, list[Document]] = {}
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            query = obj["query"]
            documents = obj["documents"]
            lc_documents: list[Document] = []
            for document in documents:
                page_content = document.pop("content")
                metadata = document
                lc_document = Document(page_content=page_content, metadata=metadata)
                lc_documents.append(lc_document)
            query_dict[query] = lc_documents
    return query_dict
