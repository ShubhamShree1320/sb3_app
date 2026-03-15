from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Enum, Text, Uuid, func
from sqlalchemy.orm import DeclarativeBase

from sb3_api.enums.evaluation import Evaluation


class BaseFeedback(DeclarativeBase):
    __abstract__ = True
    id = Column(Uuid, primary_key=True)

    session_id = Column(Uuid, nullable=False)
    query = Column(Text, nullable=False)
    result = Column(Text, nullable=False)
    evaluation = Column(Enum(Evaluation), nullable=False)  # type: ignore[var-annotated]
    updated_at = Column(DateTime(timezone=True), server_default=func.now())


_model_cache: dict[str, type[BaseFeedback]] = {}


def create_model(table_name: str, embeddings_size: int) -> type[BaseFeedback]:
    if table_name in _model_cache:
        return _model_cache[table_name]

    class FeedbackModel(BaseFeedback):  # table-specific dynamic model
        __tablename__ = table_name
        embedding = Column(Vector(dim=embeddings_size), nullable=False)

    _model_cache[table_name] = FeedbackModel
    return FeedbackModel
