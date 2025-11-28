"""Database models for the job queue system."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Enum
from sqlalchemy.orm import relationship, declarative_base
import enum

Base = declarative_base()


class QueueStatus(enum.Enum):
    """Status of a queued job."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    VALIDATED = "validated"
    REJECTED = "rejected"


class User(Base):
    """Local user linked to CVAT credentials."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    cvat_token = Column(String(500), nullable=True)
    cvat_host = Column(String(500), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    queued_jobs = relationship("QueuedJob", back_populates="completed_by_user", foreign_keys="QueuedJob.completed_by")
    validated_jobs = relationship("QueuedJob", back_populates="validated_by_user", foreign_keys="QueuedJob.validated_by")


class QueuedJob(Base):
    """A CVAT job that has been marked complete and is waiting validation."""
    __tablename__ = "queued_jobs"

    id = Column(Integer, primary_key=True)
    cvat_job_id = Column(Integer, nullable=False)
    cvat_task_id = Column(Integer, nullable=False)
    cvat_host = Column(String(500), nullable=False)  # Track which CVAT instance
    task_name = Column(String(500), nullable=True)
    status = Column(Enum(QueueStatus), default=QueueStatus.PENDING)

    # Who completed the job in CVAT
    completed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    completed_by_username = Column(String(100), nullable=True)
    completed_at = Column(DateTime, default=datetime.utcnow)

    # Assignment info (admin assigns job to a reviewer)
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)
    assigned_at = Column(DateTime, nullable=True)

    # First reviewer tracking (for auto-reassignment after rejection)
    first_reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Rejection tracking
    rejection_count = Column(Integer, default=0)
    last_rejection_notes = Column(String(1000), nullable=True)

    # Validation info
    validated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    validated_at = Column(DateTime, nullable=True)
    validation_notes = Column(String(1000), nullable=True)

    completed_by_user = relationship("User", back_populates="queued_jobs", foreign_keys=[completed_by])
    assigned_to_user = relationship("User", foreign_keys=[assigned_to])
    first_reviewer = relationship("User", foreign_keys=[first_reviewer_id])
    validated_by_user = relationship("User", back_populates="validated_jobs", foreign_keys=[validated_by])


class TrackedTask(Base):
    """Tasks being monitored for completed jobs."""
    __tablename__ = "tracked_tasks"

    id = Column(Integer, primary_key=True)
    cvat_task_id = Column(Integer, unique=True, nullable=False)
    task_name = Column(String(500), nullable=True)
    added_by = Column(Integer, ForeignKey("users.id"))
    added_at = Column(DateTime, default=datetime.utcnow)
    last_checked = Column(DateTime, nullable=True)


class AnnotationSnapshot(Base):
    """Snapshot of annotation counts for a task at a specific time."""
    __tablename__ = "annotation_snapshots"

    id = Column(Integer, primary_key=True)
    cvat_task_id = Column(Integer, nullable=False, index=True)
    task_name = Column(String(500), nullable=True)
    annotation_count = Column(Integer, nullable=False, default=0)
    snapshot_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class Notification(Base):
    """User notifications for job assignments and updates."""
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    message = Column(String(500), nullable=False)
    link = Column(String(200), nullable=True)  # Optional link to navigate to
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


class RejectedJobTracker(Base):
    """Tracks rejected jobs for auto-reassignment when they return to the queue."""
    __tablename__ = "rejected_job_tracker"

    id = Column(Integer, primary_key=True)
    cvat_job_id = Column(Integer, nullable=False, index=True)
    cvat_task_id = Column(Integer, nullable=False)
    cvat_host = Column(String(500), nullable=False)  # Track which CVAT instance
    task_name = Column(String(500), nullable=True)

    # The first reviewer who should be auto-assigned when job returns
    first_reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Rejection history
    rejection_count = Column(Integer, default=1)
    last_rejection_notes = Column(String(1000), nullable=True)
    rejected_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    rejected_at = Column(DateTime, default=datetime.utcnow)

    first_reviewer = relationship("User", foreign_keys=[first_reviewer_id])
    rejected_by_user = relationship("User", foreign_keys=[rejected_by])
