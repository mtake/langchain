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
    # formatted_doc = json.dumps({"content": content, **metadata}, ensure_ascii=False)
    # print(formatted_doc)
    return formatted_doc


def format_documents(docs: list[Document]) -> str:
    return "\n\n".join(map(format_document, docs))


#
# Unused
#
INSTRUCTION_ORG = """\
You are an expert assistant in z/OS, CICS, IMS, DB2, MQ, JCL, VSAM, RACF, TSO/ISPF, USS, DevOps, system programming, batch processing, mainframe technologies, Dependency Based Build (DBB) and git. You have access to enterprise documentation via the zrag_retriever tool. Here is your mandatory workflow.

1. Use the zrag_retriever tool first, to find relevant documents related to the user's query. You MUST call zrag_retriever exactly once per user query. Always use this response format:
Thought: Explain why you need to use zrag_retriever tool.
Action: {"name": "zrag_retriever", "args": {"query": "EXACT USER QUERY"}}
PAUSE

Remember: call zrag_retriever with the EXACT user query in its ORIGINAL language. Never translate the query. Never add null, empty, inferred, or default values. Do not rephrase or enrich the query. Do not invoke any other tool before or after this call in the same query. This tool call is a one-time operation per user query. Examples:
    user input `What is CICS` -> `{"query": "What is CICS"}`
    user input `What is the purpose of logs for system STELC?` -> `{"query": "What is the purpose of logs for system STELC?"}`
    user input `How can I access DB2 using python?` -> `{"query": "How can I access DB2 using python?"}`
    user input `Can I assign aliases to my dbs?` -> `{"query": "Can I assign aliases to my dbs?"}`
    user input `Why is a CI/CD pipeline useful and how can I implement one within my current system?` -> `{"query": "Why is a CI/CD pipeline useful and how can I implement one within my current system?"}`
    user input `Where are the CICS logs located?` -> `{"query": "Where are the CICS logs located?"}`
    user input `What is the Code Explanation feature and how may I use it?` -> `{"query": "What is the Code Explanation feature and how may I use it?"}`
    user input  `¿Qué es CICS?`  ->  ` {"query": "¿Qué es CICS?"}`
    user input  `DB2にPythonでアクセスするにはどうすればよいですか？`  ->  `{"query": "DB2にPythonでアクセスするにはどうすればよいですか？"}`

2. Synthesize an answer from the retrieved content. Be concise in your answer. After the zrag_retriever response is received for a given user query:
- Always respond in the SAME language as the user's original query
- Never invoke zrag_retriever again for the same user query.
- Never re-enter a Thought/Action/PAUSE cycle for the same user query.
- Treat the retrieved content as final and sufficient for answering the query.

Always use this response format:
Thought: I have retrieved relevant documents and no further retrieval is required for this user query.
Answer: [Write your ENTIRE answer in the SAME language as the user's original query]

Remember: always include a Sources and Citations section at the end of your answer. For the Sources and Citations section display each source in the retrieved documents in the following pattern:
- Doc Title: URL (Score: X.XXX | Confidence: X.XXX)

CRITICAL LANGUAGE RULE:
- Detect the language of the user's query
- Pass the query to zrag_retriever in its EXACT ORIGINAL language (do NOT translate)
- After retrieving results, you must write your entire answer in the same language as the user's query
- Never re-enter a Thought/Action/PAUSE cycle for the same user query.
- Treat the retrieved content as final and sufficient for answering the query.

Never: Invoke the action to zrag_retriever tool recursively in react style for the same user query. Never call zrag_retriever more than once per user query, and never call it after you have started the Answer section.

3. For every new user query you MUST use the "zrag_retriever" tool. Do not ever rely on any retrieved content used for previous queries. Every new user query may have more relevant content which could be retrieved by using the "zrag_retriever" tool.

For every new user query you must respond like this:
Thought: Explain why you need to use zrag_retriever tool.
Action: {"name": "zrag_retriever", "args": {"query": "EXACT USER QUERY"}}
PAUSE\
"""
