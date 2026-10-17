"""用于有声书导入任务状态的 MongoDB 持久化。"""
from datetime import datetime, timezone
from typing import Any

from app.shared.clients.mongo_history_utils import get_history_mongo_tool


class ImportTaskRepository:
    @property
    def collection(self):
        collection = get_history_mongo_tool().db["import_task"]
        collection.create_index("task_id", unique=True)
        return collection

    def save(
        self,
        task_id: str,
        *,
        status: str,
        source_file: str = "",
        metadata: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        now = datetime.now(timezone.utc)
        self.collection.update_one(
            {"task_id": task_id},
            {
                "$set": {
                    "status": status,
                    "source_file": source_file,
                    "metadata": metadata or {},
                    "error": error,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self.collection.find_one({"task_id": task_id}, {"_id": 0})


import_task_repository = ImportTaskRepository()
