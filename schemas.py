"""Pydantic bodies for the Mealbox API.

Same role as schemas.py in the FastAPI database tutorial: the shapes
/docs shows, and the checks that reject a bad body before the route runs.
"""

from pydantic import BaseModel, Field, field_validator


def _required_text(value: str, label: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} is empty.")
    return cleaned


class QueryRequest(BaseModel):
    question: str = Field(description="Customer question.")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        return _required_text(value, "Question")


class EntryCreate(BaseModel):
    id: str = Field(description="Stable entry id, for example faq-099.")
    kind: str = Field(description="faq or ticket.")
    title: str
    body: str
    category: str = Field(description="account, billing, subscription, delivery, orders, or bugs.")
    last_updated: str = Field(description="Date as YYYY-MM-DD.")

    @field_validator("id")
    @classmethod
    def id_not_blank(cls, value: str) -> str:
        return _required_text(value, "Entry id")

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        return _required_text(value, "Title")

    @field_validator("body")
    @classmethod
    def body_not_blank(cls, value: str) -> str:
        return _required_text(value, "Body")


class IngestResponse(BaseModel):
    entry_id: str
    point_id: str
    already_present: bool
    point_count: int
