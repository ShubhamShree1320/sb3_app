"""
This integrates DynamoDB session storage with Strands' SessionManager interface.
"""

import json
import logging
from datetime import datetime
from typing import Any

from strands.session import SessionManager
from strands.types.content import Message

from sb3_api.repository.session.dynamodb import DynamoDBSessionRepository

logger = logging.getLogger(__name__)


class DynamoDBStrandsSessionManager(SessionManager):
    """Strands SessionManager implementation using DynamoDB.
    
    Bridges existing DynamoDB session storage with Strands' SessionManager interface,
    maintaining compatibility with current infrastructure.
    
    Args:
        session_id: Unique session identifier
        dynamodb_repository: existing DynamoDB session repository
        actor_id: Optional actor/user identifier
        
    Example:
        >>> from sb3_api.repository.session.dynamodb import DynamoDBSessionRepository
        >>> repo = DynamoDBSessionRepository(
        ...     table_name="sessions",
        ...     endpoint_url=None  # or local endpoint for testing
        ... )
        >>> session_manager = DynamoDBStrandsSessionManager(
        ...     session_id="abc123",
        ...     dynamodb_repository=repo,
        ...     actor_id="user@example.com"
        ... )
        >>> # Now use with Strands Agent
        >>> agent = Agent(model=..., session_manager=session_manager, ...)
    """
    
    def __init__(
        self,
        session_id: str,
        dynamodb_repository: DynamoDBSessionRepository,
        actor_id: str | None = None,
    ) -> None:
        """Initialize DynamoDB session manager."""
        super().__init__(session_id=session_id)
        self.repository = dynamodb_repository
        self.actor_id = actor_id or "default"
        logger.info(f"Initialized DynamoDB session manager for session: {session_id}")
    
    def load_messages(self) -> list[Message]:
        """Load conversation messages from DynamoDB.
        
        Returns:
            List of Strands Message objects from stored conversation history
        """
        try:
            # Load session from your DynamoDB repository
            session_data = self.repository.get_session(
                session_id=self.session_id,
                actor_id=self.actor_id
            )
            
            if not session_data or not session_data.get('messages'):
                logger.info(f"No existing messages for session {self.session_id}")
                return []
            
            # Convert stored messages to Strands Message format
            messages = []
            for msg_data in session_data['messages']:
                message = self._convert_to_strands_message(msg_data)
                if message:
                    messages.append(message)
            
            logger.info(f"Loaded {len(messages)} messages from DynamoDB")
            return messages
            
        except Exception as e:
            logger.error(f"Error loading messages from DynamoDB: {e}")
            return []
    
    def save_messages(self, messages: list[Message]) -> None:
        """Save conversation messages to DynamoDB.
        
        Args:
            messages: List of Strands Message objects to persist
        """
        try:
            # Convert Strands messages to storage format
            stored_messages = []
            for msg in messages:
                stored_msg = self._convert_from_strands_message(msg)
                if stored_msg:
                    stored_messages.append(stored_msg)
            
            # Save to DynamoDB using your repository
            self.repository.save_session(
                session_id=self.session_id,
                actor_id=self.actor_id,
                messages=stored_messages,
                metadata={
                    'last_updated': datetime.utcnow().isoformat(),
                    'message_count': len(stored_messages)
                }
            )
            
            logger.info(f"Saved {len(stored_messages)} messages to DynamoDB")
            
        except Exception as e:
            logger.error(f"Error saving messages to DynamoDB: {e}")
            raise
    
    def _convert_to_strands_message(self, msg_data: dict) -> Message | None:
        """Convert DynamoDB message format to Strands Message."""
        try:
            role = msg_data.get('role', 'user')
            content = msg_data.get('content', '')
            
            # Strands Message expects content as list of dicts
            if isinstance(content, str):
                content = [{"text": content}]
            elif isinstance(content, dict):
                content = [content]
            elif not isinstance(content, list):
                content = [{"text": str(content)}]
            
            return Message(
                role=role,
                content=content
            )
        except Exception as e:
            logger.warning(f"Failed to convert message to Strands format: {e}")
            return None
    
    def _convert_from_strands_message(self, message: Message) -> dict | None:
        """Convert Strands Message to DynamoDB storage format."""
        try:
            # Extract text content from Strands message
            content_text = ""
            if hasattr(message, 'content') and message.content:
                if isinstance(message.content, list) and len(message.content) > 0:
                    # Handle different content types
                    content_item = message.content[0]
                    if isinstance(content_item, dict):
                        content_text = content_item.get('text', '')
                    else:
                        content_text = str(content_item)
                elif isinstance(message.content, str):
                    content_text = message.content
            
            return {
                'role': message.role,
                'content': content_text,
                'timestamp': datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Failed to convert Strands message to storage format: {e}")
            return None


def create_dynamodb_session_manager(
    session_id: str,
    table_name: str,
    endpoint_url: str | None = None,
    actor_id: str | None = None,
) -> DynamoDBStrandsSessionManager:
    """Factory to create DynamoDB session manager for Strands.
    
    Args:
        session_id: Unique session identifier
        table_name: DynamoDB table name
        endpoint_url: DynamoDB endpoint (None for AWS, or local URL for testing)
        actor_id: Optional actor/user identifier
        
    Returns:
        DynamoDBStrandsSessionManager configured and ready to use
        
    Example:
        >>> session_manager = create_dynamodb_session_manager(
        ...     session_id="session-123",
        ...     table_name="chat-sessions",
        ...     actor_id="user@example.com"
        ... )
    """
    repository = DynamoDBSessionRepository(
        table_name=table_name,
        endpoint_url=endpoint_url
    )
    
    return DynamoDBStrandsSessionManager(
        session_id=session_id,
        dynamodb_repository=repository,
        actor_id=actor_id
    )