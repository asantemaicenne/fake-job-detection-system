from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from bson import ObjectId
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.api.middleware.auth import create_access_token
from src.data.repositories.database import DatabaseManager


class MockCursor:
    """Simulates an asynchronous Motor cursor with chaining support."""

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self._data = data

    def sort(self, *args, **kwargs) -> "MockCursor":
        return self

    def skip(self, n: int) -> "MockCursor":
        self._data = self._data[n:]
        return self

    def limit(self, n: int) -> "MockCursor":
        self._data = self._data[:n]
        return self

    async def to_list(self, length: int) -> List[Dict[str, Any]]:
        return self._data[:length]


class MockCollection:
    """In-memory collection mock handling basic Motor document operations."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.store: List[Dict[str, Any]] = []

    async def insert_one(self, doc: Dict[str, Any]) -> Any:
        stored_doc = doc.copy()
        if "_id" not in stored_doc or stored_doc["_id"] is None:
            stored_doc["_id"] = str(ObjectId())
        self.store.append(stored_doc)
        res = MagicMock()
        res.inserted_id = stored_doc["_id"]
        return res

    async def count_documents(self, query: Dict[str, Any]) -> int:
        count = 0
        for doc in self.store:
            matches = True
            for k, v in query.items():
                if k == "_id" and isinstance(v, dict) and "$ne" in v:
                    if doc.get("_id") == v["$ne"]:
                        matches = False
                elif k == "posted_at" and isinstance(v, dict) and "$gte" in v:
                    pass
                elif k in doc and doc[k] != v:
                    matches = False
            if matches:
                count += 1
        return count

    def find(self, query: Optional[Dict[str, Any]] = None) -> MockCursor:
        query = query or {}
        results = []
        for doc in self.store:
            matches = True
            for k, v in query.items():
                if doc.get(k) != v:
                    matches = False
                    break
            if matches:
                results.append(doc)
        return MockCursor(results)


class MockDatabase:
    """In-memory mock representing the application MongoDB database."""

    def __init__(self) -> None:
        self.collections: Dict[str, MockCollection] = {}

    def __getitem__(self, name: str) -> MockCollection:
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]


@pytest.fixture(autouse=True)
def patch_database(monkeypatch: pytest.MonkeyPatch) -> MockDatabase:
    """Replaces DatabaseManager singleton with an in-memory mock for isolated
    testing.
    """
    mock_db = MockDatabase()
    monkeypatch.setattr(DatabaseManager, "client", MagicMock())
    monkeypatch.setattr(DatabaseManager, "db", mock_db)
    monkeypatch.setattr(DatabaseManager, "connect", AsyncMock())
    monkeypatch.setattr(DatabaseManager, "disconnect", AsyncMock())
    monkeypatch.setattr(DatabaseManager, "get_database", lambda: mock_db)
    return mock_db


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Provides an asynchronous HTTP test client bound directly to FastAPI."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as test_client:
        yield test_client


@pytest.fixture
def admin_headers() -> Dict[str, str]:
    """Generates authorization headers with administrator claims."""
    token = create_access_token(subject="admin_user", role="admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def analyst_headers() -> Dict[str, str]:
    """Generates authorization headers with analyst claims."""
    token = create_access_token(subject="analyst_user", role="analyst")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api_user_headers() -> Dict[str, str]:
    """Generates authorization headers with standard api_user claims."""
    token = create_access_token(subject="client_app", role="api_user")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def unauthorized_role_headers() -> Dict[str, str]:
    """Generates authorization headers with an unprivileged role."""
    token = create_access_token(subject="restricted_user", role="guest")
    return {"Authorization": f"Bearer {token}"}
