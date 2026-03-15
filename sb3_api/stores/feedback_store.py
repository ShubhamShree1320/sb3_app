from uuid import UUID

from sb3_api.encoder import BedrockEncoder
from sb3_api.enums.tables import TableName
from sb3_api.models.feedback import FeedbackBase, FeedbackRecord, FeedbackRecordOutput
from sb3_api.repository.feedback.base import FeedbackRepository


class FeedbackStore:
    def __init__(self, repository: FeedbackRepository, encoder: BedrockEncoder) -> None:
        self.repository = repository
        self.encoder = encoder

    async def save_feedback(self, data: FeedbackRecord, *, query_cache: bool = False) -> FeedbackRecordOutput:
        """Save feedback data to the repository.
        If query_cache is True, the feedback is saved to a cache for retrieval.
        """
        values = data.model_dump()

        # Texts that are being searched over in vector search should be embedded as documents
        # embedding = await self.encoder.embed_documents([data.query])  # noqa: ERA001
        # values["embedding"] = embedding[0]  # noqa: ERA001
        # TODO: revert this change after SB3-90 is done.
        values["embedding"] = [0.0] * 1024  # Fixed default embedding vector to fix issue on prod

        table_name = TableName.QUERY_RECORDS if query_cache else TableName.FEEDBACK
        id_ = await self.repository.add_feedback(values=values, table_name=table_name)
        record = await self.repository.get_feedback(feedback_id=id_, table_name=table_name)
        return FeedbackRecordOutput.from_data_model(record)

    async def update_feedback(self, feedback_id: UUID, data: FeedbackBase) -> FeedbackRecordOutput:
        values = data.model_dump()

        await self.repository.update_feedback(feedback_id=feedback_id, values=values)
        record = await self.repository.get_feedback(feedback_id=feedback_id)

        return FeedbackRecordOutput.from_data_model(record)

    async def get_feedback(self, feedback_id: UUID, *, query_cache: bool = False) -> FeedbackRecordOutput:
        table_name = TableName.QUERY_RECORDS if query_cache else TableName.FEEDBACK

        record = await self.repository.get_feedback(feedback_id=feedback_id, table_name=table_name)
        return FeedbackRecordOutput.from_data_model(record)

    async def get_session_feedback(self, session_id: UUID) -> list[FeedbackRecordOutput]:
        feedback_data = await self.repository.get_feedback_by_session(session_id=session_id)
        return [FeedbackRecordOutput.from_data_model(data) for data in feedback_data]

    async def delete_feedback(self, feedback_id: UUID, *, query_cache: bool = False) -> None:
        table_name = TableName.QUERY_RECORDS if query_cache else TableName.FEEDBACK
        return await self.repository.delete_feedback(feedback_id=feedback_id, table_name=table_name)
