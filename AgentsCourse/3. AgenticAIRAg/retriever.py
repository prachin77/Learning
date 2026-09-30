"""Build a BM25 retriever over the gala guest dataset."""

import datasets
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.tools import Tool


def load_guest_documents() -> list[Document]:
    """Load the course guest dataset and convert rows to LangChain documents."""
    guest_dataset = datasets.load_dataset(
        "agents-course/unit3-invitees",
        split="train",
    )
    return [
        Document(
            page_content="\n".join(
                [
                    f"Name: {guest['name']}",
                    f"Relation: {guest['relation']}",
                    f"Description: {guest['description']}",
                    f"Email: {guest['email']}",
                ]
            ),
            metadata={"name": guest["name"]},
        )
        for guest in guest_dataset
    ]


def create_guest_info_tool() -> Tool:
    """Create a tool that finds relevant guest records with BM25."""
    retriever = BM25Retriever.from_documents(load_guest_documents())

    def extract_text(query: str) -> str:
        results = retriever.invoke(query)
        if not results:
            return "No matching guest information found."
        return "\n\n".join(document.page_content for document in results[:3])

    return Tool(
        name="guest_info_retriever",
        func=extract_text,
        description=(
            "Retrieves detailed information about gala guests based on their "
            "name, relation, or description."
        ),
    )


guest_info_tool = create_guest_info_tool()