from pydantic import BaseModel


class UserCreate(BaseModel):
    name: str
    age: int | None = None
    language: str
    interaction_mode: str | None = None