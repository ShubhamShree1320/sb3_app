from uuid import UUID


class NotFoundError(Exception):
    pass


class SessionNotFoundError(NotFoundError):
    def __init__(self, session_id: UUID) -> None:
        self.id = session_id
        message = f"Session {session_id} not found"
        super().__init__(message)


class AuthorizationError(NotFoundError):
    def __init__(self, user: str) -> None:
        message = f"User {user} cannot access this resource."
        super().__init__(message)


class FeedbackNotFoundError(NotFoundError):
    def __init__(self, feedback_id: UUID) -> None:
        self.id = feedback_id
        message = f"Feedback {feedback_id} not found"
        super().__init__(message)


class UserNotFoundError(NotFoundError):
    def __init__(self, user: str) -> None:
        message = f"User {user} not found"
        super().__init__(message)


class CollectionNotFoundError(NotFoundError):
    def __init__(self, collection_name: str) -> None:
        self.name = collection_name
        message = f"Collection {collection_name} not found"
        super().__init__(message)
