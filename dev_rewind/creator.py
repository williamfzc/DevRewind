from git import Commit
from langchain.schema import Document


class Creator(object):
    """
    creator was designed for converting commits to documents
    """

    def create_doc_from_commit(self, file_path: str, commit: Commit):
        content = commit.message
        return Document(page_content=content, metadata={"sha": commit.hexsha, "file": file_path})
