from typing import Dict, Any


class DocumentStore:
    def __init__(self):
        self._items: Dict[str, Dict[str, Any]] = {}

    def put(self, document_id: str, value: Dict[str, Any]) -> None:
        self._items[document_id] = value

    def get(self, document_id: str) -> Dict[str, Any] | None:
        return self._items.get(document_id)

    def update(self, document_id: str, value: Dict[str, Any]) -> None:
        self._items[document_id] = value


store = DocumentStore()
