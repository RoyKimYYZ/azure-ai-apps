from dataclasses import dataclass, field
from typing import Annotated
from uuid import uuid4

from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel


@vectorstoremodel(collection_name="resume-index")
@dataclass
class ResumeModel:
    """model to store some text with a ID."""

    parent_id: Annotated[str, VectorStoreField("key")] # = field(default_factory=lambda: str(uuid4()))
    id: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    title: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    firstName: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    lastName: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    profession: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    resume: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    filepath: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    content: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    resumeContent: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    
    resumeVector: Annotated[list[float] | str | None, VectorStoreField("vector", dimensions=1536)] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = self.text