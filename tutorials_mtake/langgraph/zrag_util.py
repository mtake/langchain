from typing import override, Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from langchain_core.retrievers import BaseRetriever


class ZRAGRetriever(BaseRetriever):
    """Retriever for Z-related documents"""
    _documents_dict: dict[str, list[Document]] = {}

    @classmethod
    def from_dict(cls, documents_dict: dict[str, list[Document]], **kwargs: Any) -> "ZRAGRetriever":
        zrag_retriever = cls()
        zrag_retriever._documents_dict.update(documents_dict)
        return zrag_retriever

    def add_dict(self, documents_dict: dict[str, list[Document]], **kwargs: Any):
        self._documents_dict.update(documents_dict)

    def add_query(self, query: str, documents: list[Document], **kwargs: Any):
        self._documents_dict[query] = documents

    @override
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self._documents_dict.get(query, [])


import json

def convert_json_to_dict(input_file: str) -> dict[str, list[Document]]:

    documents_dict: dict[str, list[Document]] = {}

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
            documents_dict[query] = lc_documents

    return documents_dict
