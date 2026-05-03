import pytest

from methylcurate.api.session import SessionStore, RunSession


class TestSessionStore:
    def test_create_returns_session(self):
        store = SessionStore()
        session = store.create("run-1")
        assert isinstance(session, RunSession)
        assert session.run_id == "run-1"

    def test_get_returns_existing_session(self):
        store = SessionStore()
        store.create("run-1")
        session = store.get("run-1")
        assert session.run_id == "run-1"

    def test_get_raises_keyerror_for_missing(self):
        store = SessionStore()
        with pytest.raises(KeyError):
            store.get("nonexistent")

    def test_delete_removes_session(self):
        store = SessionStore()
        store.create("run-1")
        store.delete("run-1")
        assert not store.exists("run-1")

    def test_delete_raises_keyerror_for_missing(self):
        store = SessionStore()
        with pytest.raises(KeyError):
            store.delete("nonexistent")

    def test_list_runs(self):
        store = SessionStore()
        store.create("run-1")
        store.create("run-2")
        runs = store.list_runs()
        assert set(runs) == {"run-1", "run-2"}

    def test_exists(self):
        store = SessionStore()
        assert not store.exists("run-1")
        store.create("run-1")
        assert store.exists("run-1")
