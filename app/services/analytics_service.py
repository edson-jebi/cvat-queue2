"""Analytics service for tracking annotation progress over time."""

from __future__ import annotations

import json
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.models import AnnotationSnapshot, LabelSnapshot
from app.services.cvat_client import CVATClient


class AnalyticsService:
    """Service for tracking and analyzing annotation progress."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def capture_snapshot(
        self,
        client: CVATClient,
        task_id: int,
        task_name: str
    ) -> AnnotationSnapshot:
        """
        Capture current annotation count for a task.
        Creates a new snapshot in the database.
        """
        count = await client.get_task_annotations_count(task_id)

        snapshot = AnnotationSnapshot(
            cvat_task_id=task_id,
            task_name=task_name,
            annotation_count=count,
            snapshot_time=datetime.utcnow()
        )
        self.db.add(snapshot)
        await self.db.commit()
        await self.db.refresh(snapshot)
        return snapshot

    async def get_task_history(self, task_id: int) -> list[AnnotationSnapshot]:
        """Get all annotation snapshots for a task, ordered by time."""
        result = await self.db.execute(
            select(AnnotationSnapshot)
            .where(AnnotationSnapshot.cvat_task_id == task_id)
            .order_by(AnnotationSnapshot.snapshot_time)
        )
        return result.scalars().all()

    async def get_latest_snapshot(self, task_id: int) -> AnnotationSnapshot | None:
        """Get the most recent snapshot for a task."""
        result = await self.db.execute(
            select(AnnotationSnapshot)
            .where(AnnotationSnapshot.cvat_task_id == task_id)
            .order_by(desc(AnnotationSnapshot.snapshot_time))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def capture_label_snapshot(
        self,
        client: CVATClient,
        task_id: int,
        task_name: str
    ) -> LabelSnapshot:
        """
        Capture current label statistics for a task.
        Creates a new label snapshot in the database.
        """
        label_stats = await client.get_task_labels_statistics(task_id)

        snapshot = LabelSnapshot(
            cvat_task_id=task_id,
            task_name=task_name,
            label_stats=json.dumps(label_stats),
            snapshot_time=datetime.utcnow()
        )
        self.db.add(snapshot)
        await self.db.commit()
        await self.db.refresh(snapshot)
        return snapshot

    async def get_label_history(self, task_id: int) -> list[LabelSnapshot]:
        """Get all label snapshots for a task, ordered by time."""
        result = await self.db.execute(
            select(LabelSnapshot)
            .where(LabelSnapshot.cvat_task_id == task_id)
            .order_by(LabelSnapshot.snapshot_time)
        )
        return result.scalars().all()
