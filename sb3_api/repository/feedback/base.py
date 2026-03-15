from abc import ABC, abstractmethod
from uuid import UUID

from sb3_api.enums.encoder import EncoderModel
from sb3_api.enums.tables import TableName
from sb3_api.repository.feedback.model import BaseFeedback


class FeedbackRepository(ABC):
    @abstractmethod
    async def create_schema(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def add_feedback(self, values: dict, table_name: TableName = TableName.FEEDBACK) -> UUID:
        raise NotImplementedError

    @abstractmethod
    async def update_feedback(
        self, feedback_id: UUID, values: dict, table_name: TableName = TableName.FEEDBACK
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_feedback(self, feedback_id: UUID, table_name: TableName = TableName.FEEDBACK) -> BaseFeedback:
        raise NotImplementedError

    @abstractmethod
    async def get_feedback_by_session(
        self, session_id: UUID, table_name: TableName = TableName.FEEDBACK
    ) -> list[BaseFeedback]:
        raise NotImplementedError

    @abstractmethod
    async def delete_feedback(self, feedback_id: UUID, table_name: TableName = TableName.FEEDBACK) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def encoder_model(self) -> EncoderModel:
        pass
