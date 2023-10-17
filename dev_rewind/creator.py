from git import Commit
from langchain.schema import Document

from dev_rewind.context import FileContext


class Creator(object):
    """
    creator was designed for converting commits to documents
    """

    def create_doc(self, file_ctx: FileContext, commit: Commit):
        page_content = commit.message
        metadata = {
            "sha": commit.hexsha,
            "file": file_ctx.name,
            "author": str(commit.author),
        }

        return Document(page_content=page_content, metadata=metadata)
