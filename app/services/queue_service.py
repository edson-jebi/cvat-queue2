"""Queue management service for handling completed CVAT jobs."""

from __future__ import annotations

from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import QueuedJob, QueueStatus, RejectedJobTracker, Notification
from app.services.cvat_client import CVATClient


class QueueService:
    """Service for managing the job validation queue."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_pending_jobs(self) -> list[QueuedJob]:
        """Get all jobs pending validation."""
        result = await self.db.execute(
            select(QueuedJob).where(QueuedJob.status == QueueStatus.PENDING)
            .order_by(QueuedJob.completed_at)
        )
        return result.scalars().all()

    async def is_job_queued(self, cvat_job_id: int, cvat_host: str) -> bool:
        """Check if a job is already in the queue for a specific CVAT instance."""
        result = await self.db.execute(
            select(QueuedJob).where(
                QueuedJob.cvat_job_id == cvat_job_id,
                QueuedJob.cvat_host == cvat_host
            )
        )
        return result.scalar_one_or_none() is not None

    async def add_to_queue(
        self,
        cvat_job_id: int,
        cvat_task_id: int,
        cvat_host: str,
        task_name: str,
        completed_by_id: int,
        completed_by_username: str
    ) -> QueuedJob:
        """Add a completed job to the validation queue."""
        job = QueuedJob(
            cvat_job_id=cvat_job_id,
            cvat_task_id=cvat_task_id,
            cvat_host=cvat_host,
            task_name=task_name,
            status=QueueStatus.PENDING,
            completed_by=completed_by_id,
            completed_by_username=completed_by_username,
            completed_at=datetime.utcnow()
        )
        self.db.add(job)
        await self.db.commit()
        return job

    async def sync_completed_jobs(
        self,
        client: CVATClient,
        task_id: int,
        task_name: str,
        user_id: int,
        cvat_host: str
    ) -> int:
        """
        Sync all completed annotation jobs from a CVAT task to the queue.
        Only enqueues jobs where stage=annotation AND state=completed.
        Auto-assigns to first reviewer if job was previously rejected.
        Returns the number of new jobs added.
        """
        jobs = await client.get_jobs(task_id)
        added = 0

        for job in jobs:
            # Only enqueue when annotation stage is completed
            if job.stage == "annotation" and job.state == "completed":
                if not await self.is_job_queued(job.id, cvat_host):
                    # Check if this job was previously rejected (for this CVAT instance)
                    tracker_result = await self.db.execute(
                        select(RejectedJobTracker).where(
                            RejectedJobTracker.cvat_job_id == job.id,
                            RejectedJobTracker.cvat_host == cvat_host
                        ).order_by(RejectedJobTracker.rejected_at.desc())
                    )
                    tracker = tracker_result.scalar_one_or_none()

                    # Create the queued job
                    queued_job = QueuedJob(
                        cvat_job_id=job.id,
                        cvat_task_id=task_id,
                        cvat_host=cvat_host,
                        task_name=task_name,
                        status=QueueStatus.PENDING,
                        completed_by=user_id,
                        completed_by_username=job.assignee or "Unknown",
                        completed_at=datetime.utcnow()
                    )

                    # If previously rejected, copy rejection info and auto-assign
                    if tracker:
                        queued_job.rejection_count = tracker.rejection_count
                        queued_job.first_reviewer_id = tracker.first_reviewer_id
                        queued_job.last_rejection_notes = tracker.last_rejection_notes

                        # Auto-assign to first reviewer (only admin can change this)
                        if tracker.first_reviewer_id:
                            queued_job.assigned_to = tracker.first_reviewer_id
                            queued_job.assigned_at = datetime.utcnow()

                            # Notify the reviewer
                            notification = Notification(
                                user_id=tracker.first_reviewer_id,
                                message=f"Job #{job.id} has returned for review (rejection #{tracker.rejection_count})",
                                link="/queue"
                            )
                            self.db.add(notification)

                        # Delete the tracker as we've processed it
                        await self.db.delete(tracker)

                    self.db.add(queued_job)
                    await self.db.commit()
                    added += 1

        return added

    async def take_for_review(self, queue_id: int, admin_id: int) -> bool:
        """Mark a job as being reviewed by an admin."""
        result = await self.db.execute(
            select(QueuedJob).where(
                QueuedJob.id == queue_id,
                QueuedJob.status == QueueStatus.PENDING
            )
        )
        job = result.scalar_one_or_none()

        if not job:
            return False

        job.status = QueueStatus.IN_REVIEW
        job.validated_by = admin_id
        await self.db.commit()
        return True

    async def validate_job(
        self,
        queue_id: int,
        approved: bool,
        notes: str = ""
    ) -> bool:
        """Validate or reject a job."""
        result = await self.db.execute(
            select(QueuedJob).where(QueuedJob.id == queue_id)
        )
        job = result.scalar_one_or_none()

        if not job:
            return False

        job.status = QueueStatus.VALIDATED if approved else QueueStatus.REJECTED
        job.validated_at = datetime.utcnow()
        job.validation_notes = notes
        await self.db.commit()
        return True

    async def get_stats(self) -> dict:
        """Get queue statistics."""
        pending = await self.db.execute(
            select(QueuedJob).where(QueuedJob.status == QueueStatus.PENDING)
        )
        in_review = await self.db.execute(
            select(QueuedJob).where(QueuedJob.status == QueueStatus.IN_REVIEW)
        )
        validated = await self.db.execute(
            select(QueuedJob).where(QueuedJob.status == QueueStatus.VALIDATED)
        )
        rejected = await self.db.execute(
            select(QueuedJob).where(QueuedJob.status == QueueStatus.REJECTED)
        )

        return {
            "pending": len(pending.scalars().all()),
            "in_review": len(in_review.scalars().all()),
            "validated": len(validated.scalars().all()),
            "rejected": len(rejected.scalars().all())
        }
