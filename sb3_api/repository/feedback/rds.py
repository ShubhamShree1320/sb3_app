import logging
import uuid
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from sb3_api.enums.encoder import EncoderModel
from sb3_api.enums.tables import TableName
from sb3_api.exceptions.exceptions import FeedbackNotFoundError
from sb3_api.repository.feedback.base import FeedbackRepository
from sb3_api.repository.feedback.model import BaseFeedback, create_model

log = logging.getLogger(__name__)


class RDSFeedbackRepository(FeedbackRepository):
    def __init__(
        self,
        db_url: str,
        encoder_model: EncoderModel,
        engine_kwargs: dict | None = None,
    ) -> None:
        _engine = create_async_engine(db_url, **(engine_kwargs or {}))
        _session_factory = async_sessionmaker(bind=_engine)

        self.engine = _engine
        self.dialect = _engine.dialect.name
        self.session_factory = _session_factory
        self._encoder_model = encoder_model

        # # Query records: historical queries with positive feedback used for similarity search
        # # Feedback table: all queries with feedback (positive and negative) for evaluation purposes
        # # TODO: remove the embeddings_size argument from feedback table when the it does not store embeddings anymore
        self.tables = {
            table.value: create_model(table_name=table.value, embeddings_size=encoder_model.embeddings_size)
            for table in TableName
        }

    @property
    def encoder_model(self) -> EncoderModel:
        return self._encoder_model

    async def create_schema(self) -> None:
        # TODO: Remove this once alembic is set up
        if self.dialect == "postgresql":
            async with self.engine.begin() as session:
                # Allow the pgvector extension for similarity search
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                await session.commit()

        async with self.engine.begin() as session:
            await session.run_sync(self.tables[TableName.QUERY_RECORDS].metadata.create_all)

    async def add_feedback(self, values: dict, table_name: TableName = TableName.FEEDBACK) -> UUID:
        values["id"] = uuid.uuid4()
        values["updated_at"] = datetime.now(UTC)

        table = self.tables[table_name]
        async with self.session_factory() as session:
            stmt = insert(table).values(values)
            await session.execute(stmt)
            await session.commit()

        return values["id"]

    async def get_feedback(self, feedback_id: UUID, table_name: TableName = TableName.FEEDBACK) -> BaseFeedback:
        table = self.tables[table_name]

        async with self.session_factory() as session:
            query = select(table).where(table.id == feedback_id)
            result = await session.execute(query)
            data = result.scalar()

        if not data:
            raise FeedbackNotFoundError(feedback_id=feedback_id)

        return data

    async def get_feedback_by_session(
        self, session_id: UUID, table_name: TableName = TableName.FEEDBACK
    ) -> list[BaseFeedback]:
        table = self.tables[table_name]

        async with self.session_factory() as session:
            query = select(table).where(table.session_id == session_id)
            result = await session.execute(query)
            data = result.scalars().all()

        return list(data)

    async def update_feedback(
        self, feedback_id: UUID, values: dict, table_name: TableName = TableName.FEEDBACK
    ) -> None:
        table = self.tables[table_name]
        values["updated_at"] = datetime.now(UTC)

        async with self.session_factory() as session:
            stmt = update(table).where(table.id == feedback_id).values(values)
            await session.execute(stmt)
            await session.commit()

    async def delete_feedback(self, feedback_id: UUID, table_name: TableName = TableName.FEEDBACK) -> None:
        table = self.tables[table_name]

        async with self.session_factory() as session:
            query = select(table).where(table.id == feedback_id)
            result = await session.execute(query)
            data = result.scalar()

            if data:
                await session.delete(data)
                await session.commit()
