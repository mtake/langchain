import json
from typing import override, Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class ZRAGRetriever(BaseRetriever):
    """Retriever for Z-related documents"""
    query_dict: dict[str, list[Document]] = {}

    def add_query(self, query: str, documents: list[Document], **kwargs: Any):
        self.query_dict[query] = documents

    @override
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return self.query_dict.get(query, [])

    def add_dict(self, query_dict: dict[str, list[Document]], **kwargs: Any):
        self.query_dict.update(query_dict)

    @classmethod
    def from_dict(cls, query_dict: dict[str, list[Document]], **kwargs: Any) -> "ZRAGRetriever":
        zrag_retriever = cls()
        zrag_retriever.add_dict(query_dict, **kwargs)
        return zrag_retriever

    @classmethod
    def from_json(cls, query_file: str, **kwargs: Any) -> "ZRAGRetriever":
        query_dict = convert_json_to_dict(query_file)
        return cls.from_dict(query_dict, **kwargs)


def convert_json_to_dict(query_file: str) -> dict[str, list[Document]]:
    query_dict: dict[str, list[Document]] = {}
    with open(query_file, "r", encoding="utf-8") as fin:
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


#
# Experimental
#
template = "[Rank {metadata[rank]}] Title: {metadata[title]}\nURL: {metadata[url]}\nScore: {metadata[score]}\nContent: {content}"

def format_document(doc: Document) -> str:
    content = doc.page_content
    metadata = doc.metadata
    formatted_doc = template.format(content=content, metadata=metadata)
    # print(formatted_doc)
    return formatted_doc


def format_documents(docs: list[Document]) -> str:
    return "\n\n".join(map(format_document, docs))
