"""Main application routes."""

import os
import shutil
import math
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request, Form, Header
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.database import get_db, DATA_DIR

# Database file path (use same DATA_DIR as database.py)
DB_PATH = os.path.join(DATA_DIR, "queue.db")
BACKUP_DIR = os.environ.get("BACKUP_DIR", "./backups")
os.makedirs(BACKUP_DIR, exist_ok=True)
from app.models import User, QueuedJob, QueueStatus, TrackedTask, AnnotationSnapshot, Notification, RejectedJobTracker
from app.api.auth import get_current_user, require_user, require_admin
from app.services.cvat_client import CVATClient
from app.services.queue_service import QueueService
from app.services.analytics_service import AnalyticsService

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, db: AsyncSession = Depends(get_db)):
    """Login page or redirect to dashboard."""
    user = await get_current_user(request, db)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: User = Depends(require_user), db: AsyncSession = Depends(get_db)):
    """Main dashboard showing tasks and queue."""
    # Get CVAT tasks and projects
    client = CVATClient(user.cvat_host, user.cvat_token)
    tasks = await client.get_tasks(include_jobs_progress=True)
    projects = await client.get_projects()
    await client.close()

    # Create project lookup
    project_lookup = {p.id: p.name for p in projects}

    # Get pending queue count
    result = await db.execute(
        select(QueuedJob).where(QueuedJob.status == QueueStatus.PENDING)
    )
    pending_jobs = result.scalars().all()

    # Get jobs assigned to this user
    result = await db.execute(
        select(QueuedJob).where(
            QueuedJob.assigned_to == user.id,
            QueuedJob.status.in_([QueueStatus.PENDING, QueueStatus.IN_REVIEW])
        )
    )
    my_assigned_jobs = result.scalars().all()

    # Get unread notifications count
    result = await db.execute(
        select(Notification).where(
            Notification.user_id == user.id,
            Notification.is_read == False
        )
    )
    unread_count = len(result.scalars().all())

    # Get task activity stats from queue (top tasks by activity)
    from collections import defaultdict
    result = await db.execute(
        select(QueuedJob).where(QueuedJob.cvat_host == user.cvat_host)
    )
    all_queue_jobs = result.scalars().all()

    # Aggregate stats per task
    task_activity = defaultdict(lambda: {
        "task_id": 0,
        "task_name": "",
        "total_jobs": 0,
        "pending": 0,
        "in_review": 0,
        "validated": 0,
        "rejected": 0,
        "total_rejections": 0
    })

    for job in all_queue_jobs:
        task_key = job.cvat_task_id
        task_activity[task_key]["task_id"] = job.cvat_task_id
        task_activity[task_key]["task_name"] = job.task_name or f"Task {job.cvat_task_id}"
        task_activity[task_key]["total_jobs"] += 1

        if job.rejection_count:
            task_activity[task_key]["total_rejections"] += job.rejection_count

        if job.status == QueueStatus.PENDING:
            task_activity[task_key]["pending"] += 1
        elif job.status == QueueStatus.IN_REVIEW:
            task_activity[task_key]["in_review"] += 1
        elif job.status == QueueStatus.VALIDATED:
            task_activity[task_key]["validated"] += 1
        elif job.status == QueueStatus.REJECTED:
            task_activity[task_key]["rejected"] += 1

    # Create lookups for task info from CVAT tasks
    task_frames = {task.id: task.size for task in tasks}
    task_project_ids = {task.id: task.project_id for task in tasks}
    task_created_dates = {task.id: task.created_date for task in tasks}

    # Include ALL CVAT tasks (even those without queue activity)
    for cvat_task in tasks:
        if cvat_task.id not in task_activity:
            task_activity[cvat_task.id] = {
                "task_id": cvat_task.id,
                "task_name": cvat_task.name,
                "total_jobs": 0,
                "pending": 0,
                "in_review": 0,
                "validated": 0,
                "rejected": 0,
                "total_rejections": 0
            }
        # Add created_date to all tasks
        task_activity[cvat_task.id]["created_date"] = cvat_task.created_date

    # Convert to list and sort by created date (newest first)
    # Send all tasks - filtering is done on the frontend
    top_tasks_list = sorted(
        task_activity.values(),
        key=lambda x: x.get("created_date") or "",
        reverse=True
    )

    # Add frame count and project info to each task
    for task in top_tasks_list:
        task["frames"] = task_frames.get(task["task_id"], 0)
        project_id = task_project_ids.get(task["task_id"])
        task["project_id"] = project_id
        task["project_name"] = project_lookup.get(project_id, "No Project") if project_id else "No Project"

    # Build project summary for Sankey visualization
    project_tasks = {}
    for task in top_tasks_list:
        proj_id = task["project_id"]
        proj_name = task["project_name"]
        if proj_name not in project_tasks:
            project_tasks[proj_name] = {
                "project_id": proj_id,
                "project_name": proj_name,
                "task_ids": [],
                "total_jobs": 0,
                "total_frames": 0
            }
        project_tasks[proj_name]["task_ids"].append(task["task_id"])
        project_tasks[proj_name]["total_jobs"] += task["total_jobs"]
        project_tasks[proj_name]["total_frames"] += task["frames"]

    # Sort projects by total jobs
    projects_summary = sorted(project_tasks.values(), key=lambda x: x["total_jobs"], reverse=True)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "tasks": tasks,
        "pending_count": len(pending_jobs),
        "my_assigned_count": len(my_assigned_jobs),
        "unread_notifications": unread_count,
        "top_tasks": top_tasks_list,
        "projects_summary": projects_summary
    })


@router.get("/task/{task_id}", response_class=HTMLResponse)
async def task_detail(
    request: Request,
    task_id: int,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """View jobs for a specific task."""
    client = CVATClient(user.cvat_host, user.cvat_token)
    jobs = await client.get_jobs(task_id)
    tasks = await client.get_tasks()
    await client.close()

    task = next((t for t in tasks if t.id == task_id), None)

    # Check which jobs are already queued
    queued_ids = set()
    result = await db.execute(select(QueuedJob.cvat_job_id))
    for row in result:
        queued_ids.add(row[0])

    # Check if all jobs are in validation stage and completed state
    all_validation_completed = (
        len(jobs) > 0 and
        all(job.stage == "validation" and job.state == "completed" for job in jobs)
    )

    return templates.TemplateResponse("task_detail.html", {
        "request": request,
        "user": user,
        "task": task,
        "jobs": jobs,
        "queued_ids": queued_ids,
        "all_validation_completed": all_validation_completed,
        "cvat_host": user.cvat_host
    })


@router.post("/enqueue/{job_id}")
async def enqueue_job(
    job_id: int,
    task_id: int = Form(...),
    task_name: str = Form(""),
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """Add a completed job to the validation queue."""
    # Check if already queued for this CVAT instance
    result = await db.execute(
        select(QueuedJob).where(
            QueuedJob.cvat_job_id == job_id,
            QueuedJob.cvat_host == user.cvat_host
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Job already in queue")

    # Verify job is completed in CVAT
    client = CVATClient(user.cvat_host, user.cvat_token)
    job = await client.get_job(job_id)
    await client.close()

    if not job or job.state != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed")

    # Add to queue
    queued = QueuedJob(
        cvat_job_id=job_id,
        cvat_task_id=task_id,
        cvat_host=user.cvat_host,
        task_name=task_name,
        completed_by=user.id,
        completed_by_username=job.assignee or user.username
    )
    db.add(queued)
    await db.commit()

    return RedirectResponse(url=f"/task/{task_id}", status_code=303)


@router.get("/queue", response_class=HTMLResponse)
async def view_queue(
    request: Request,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """View the validation queue grouped by task."""
    # Filter by user's CVAT host - each instance has its own queue
    # Admin sees all jobs for their CVAT instance, regular users only see their assigned jobs
    if user.is_admin:
        result = await db.execute(
            select(QueuedJob).where(
                QueuedJob.cvat_host == user.cvat_host,
                QueuedJob.status.in_([QueueStatus.PENDING, QueueStatus.IN_REVIEW])
            ).order_by(QueuedJob.completed_at)
        )
    else:
        result = await db.execute(
            select(QueuedJob).where(
                QueuedJob.cvat_host == user.cvat_host,
                QueuedJob.assigned_to == user.id,
                QueuedJob.status.in_([QueueStatus.PENDING, QueueStatus.IN_REVIEW])
            ).order_by(QueuedJob.completed_at)
        )
    pending_jobs = result.scalars().all()

    # Group jobs by task
    jobs_by_task = {}
    for job in pending_jobs:
        task_key = job.task_name or f"Task {job.cvat_task_id}"
        if task_key not in jobs_by_task:
            jobs_by_task[task_key] = []
        jobs_by_task[task_key].append(job)

    # Get all users for assignment dropdown (only needed for admin)
    all_users = []
    if user.is_admin:
        result = await db.execute(select(User).order_by(User.username))
        all_users = result.scalars().all()

    # Stats
    total_jobs = len(pending_jobs)
    assigned_count = sum(1 for j in pending_jobs if j.assigned_to)
    pending_count = total_jobs - assigned_count

    # Unread notifications
    result = await db.execute(
        select(Notification).where(Notification.user_id == user.id, Notification.is_read == False)
    )
    unread_notifications = len(result.scalars().all())

    return templates.TemplateResponse("queue.html", {
        "request": request,
        "user": user,
        "jobs_by_task": jobs_by_task,
        "all_users": all_users,
        "total_jobs": total_jobs,
        "assigned_count": assigned_count,
        "pending_count": pending_count,
        "unread_notifications": unread_notifications
    })


@router.get("/labelers", response_class=HTMLResponse)
async def view_labelers(
    request: Request,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """View labeler statistics with task breakdown."""
    from collections import defaultdict
    from datetime import date

    # Get all users to identify admins
    result = await db.execute(select(User).order_by(User.username))
    all_users = result.scalars().all()
    admin_usernames = {u.username for u in all_users if u.is_admin}

    # Get all jobs for stats (both current queue and history)
    result = await db.execute(
        select(QueuedJob).where(
            QueuedJob.cvat_host == user.cvat_host,
            QueuedJob.completed_by_username.isnot(None)
        )
    )
    all_jobs = result.scalars().all()

    today = date.today()

    # Build labeler stats with task breakdown
    labeler_data = defaultdict(lambda: {
        "total": 0,
        "pending": 0,
        "in_review": 0,
        "validated": 0,
        "rejected_jobs": 0,
        "today_completed": 0,
        "today_validated": 0,
        "today_rejected": 0,
        "total_rejections": 0,
        "tasks": defaultdict(lambda: {"pending": 0, "in_review": 0, "validated": 0, "rejected": 0, "total": 0})
    })

    for job in all_jobs:
        username = job.completed_by_username
        if username and username not in admin_usernames:
            data = labeler_data[username]
            task_key = job.task_name or f"Task {job.cvat_task_id}"

            data["total"] += 1
            data["tasks"][task_key]["total"] += 1

            # Add rejection_count from job (cumulative rejections for this job)
            if job.rejection_count:
                data["total_rejections"] += job.rejection_count

            if job.completed_at and job.completed_at.date() == today:
                data["today_completed"] += 1

            if job.status == QueueStatus.PENDING:
                data["pending"] += 1
                data["tasks"][task_key]["pending"] += 1
            elif job.status == QueueStatus.IN_REVIEW:
                data["in_review"] += 1
                data["tasks"][task_key]["in_review"] += 1
            elif job.status == QueueStatus.VALIDATED:
                data["validated"] += 1
                data["tasks"][task_key]["validated"] += 1
                if job.validated_at and job.validated_at.date() == today:
                    data["today_validated"] += 1
            elif job.status == QueueStatus.REJECTED:
                data["rejected_jobs"] += 1
                data["tasks"][task_key]["rejected"] += 1
                if job.validated_at and job.validated_at.date() == today:
                    data["today_rejected"] += 1

    # Convert to list with task breakdown and calculate approval rate
    labeler_stats_list = []
    for username, data in labeler_data.items():
        waiting_review = data["pending"] + data["in_review"]

        # Convert tasks to sorted list (by pending count desc)
        tasks_list = [
            {"name": task_name, **task_stats}
            for task_name, task_stats in sorted(
                data["tasks"].items(),
                key=lambda x: (x[1]["pending"] + x[1]["in_review"], x[1]["total"]),
                reverse=True
            )
        ]
        # Only include tasks with pending jobs for the expandable section
        pending_tasks = [t for t in tasks_list if t["pending"] > 0 or t["in_review"] > 0]

        # Calculate approval rate (validated / (validated + total_rejections))
        total_decisions = data["validated"] + data["total_rejections"]
        approval_rate = (data["validated"] / total_decisions * 100) if total_decisions > 0 else 100.0

        labeler_stats_list.append({
            "username": username,
            "waiting_review": waiting_review,
            "has_many_pending": waiting_review >= 3,
            "total": data["total"],
            "pending": data["pending"],
            "in_review": data["in_review"],
            "validated": data["validated"],
            "rejected_jobs": data["rejected_jobs"],
            "total_rejections": data["total_rejections"],
            "approval_rate": approval_rate,
            "today_completed": data["today_completed"],
            "today_validated": data["today_validated"],
            "today_rejected": data["today_rejected"],
            "pending_tasks": pending_tasks,
            "all_tasks": tasks_list
        })

    # Sort by waiting_review descending
    labeler_stats_list.sort(key=lambda x: (x["waiting_review"], x["total"]), reverse=True)

    # Calculate summary stats
    total_validated = sum(l["validated"] for l in labeler_stats_list)
    total_rejections = sum(l["total_rejections"] for l in labeler_stats_list)
    avg_approval_rate = (total_validated / (total_validated + total_rejections) * 100) if (total_validated + total_rejections) > 0 else 100.0

    # Unread notifications
    result = await db.execute(
        select(Notification).where(Notification.user_id == user.id, Notification.is_read == False)
    )
    unread_notifications = len(result.scalars().all())

    return templates.TemplateResponse("labelers.html", {
        "request": request,
        "user": user,
        "labeler_stats": labeler_stats_list,
        "total_labelers": len(labeler_stats_list),
        "total_pending": sum(l["waiting_review"] for l in labeler_stats_list),
        "total_validated": total_validated,
        "total_rejections": total_rejections,
        "avg_approval_rate": avg_approval_rate,
        "unread_notifications": unread_notifications
    })


@router.post("/queue/{queue_id}/take")
async def take_job(
    queue_id: int,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
    x_requested_with: str = Header(default=None)
):
    """User takes an assigned job for review - updates CVAT and returns URL."""
    result = await db.execute(select(QueuedJob).where(QueuedJob.id == queue_id))
    job = result.scalar_one_or_none()

    if not job:
        if x_requested_with == "fetch":
            return JSONResponse({"error": "Job not found"}, status_code=404)
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if user is assigned to this job or is admin
    if job.assigned_to != user.id and not user.is_admin:
        if x_requested_with == "fetch":
            return JSONResponse({"error": "You are not assigned to this job"}, status_code=403)
        raise HTTPException(status_code=403, detail="You are not assigned to this job")

    if job.status != QueueStatus.PENDING:
        if x_requested_with == "fetch":
            return JSONResponse({"error": "Job not available"}, status_code=400)
        raise HTTPException(status_code=400, detail="Job not available")

    # Update CVAT job to validation stage with new state
    client = CVATClient(user.cvat_host, user.cvat_token)
    update_success = await client.update_job_state(job.cvat_job_id, stage="validation", state="new")
    await client.close()

    if not update_success:
        if x_requested_with == "fetch":
            return JSONResponse({"error": "Failed to update job state in CVAT"}, status_code=500)
        raise HTTPException(status_code=500, detail="Failed to update job state in CVAT")

    job.status = QueueStatus.IN_REVIEW
    job.validated_by = user.id
    await db.commit()

    # Build CVAT job URL (strip trailing slash from host)
    cvat_host = user.cvat_host.rstrip('/')
    cvat_job_url = f"{cvat_host}/tasks/{job.cvat_task_id}/jobs/{job.cvat_job_id}"

    # Return JSON for fetch requests, redirect for form posts
    if x_requested_with == "fetch":
        return JSONResponse({"success": True, "cvat_url": cvat_job_url})

    return RedirectResponse(url="/queue", status_code=303)


@router.post("/queue/{queue_id}/assign")
async def assign_job_from_queue(
    queue_id: int,
    reviewer_id: int = Form(...),
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin assigns a job to a reviewer from the queue page."""
    result = await db.execute(select(QueuedJob).where(QueuedJob.id == queue_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Verify reviewer exists
    result = await db.execute(select(User).where(User.id == reviewer_id))
    reviewer = result.scalar_one_or_none()

    if not reviewer:
        raise HTTPException(status_code=404, detail="Reviewer not found")

    # Update assignment
    job.assigned_to = reviewer_id
    job.assigned_at = datetime.utcnow()

    # Set first_reviewer_id if this is the first assignment
    if not job.first_reviewer_id:
        job.first_reviewer_id = reviewer_id

    # Create notification for the assigned user
    notification = Notification(
        user_id=reviewer_id,
        message=f"You have been assigned to review Job #{job.cvat_job_id} from {job.task_name or 'Task ' + str(job.cvat_task_id)}",
        link="/queue"
    )
    db.add(notification)
    await db.commit()

    return RedirectResponse(url="/queue", status_code=303)


@router.post("/queue/{queue_id}/finish-validation")
async def finish_validation(
    queue_id: int,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
    x_requested_with: str = Header(None)
):
    """
    Finish validation by resetting job to annotation stage.
    Used when a labeler has too many rejections and cannot fix issues.
    This removes the job from queue and resets it in CVAT to stage:annotation, state:new.
    """
    result = await db.execute(select(QueuedJob).where(QueuedJob.id == queue_id))
    job = result.scalar_one_or_none()

    if not job:
        if x_requested_with == "fetch":
            return JSONResponse({"error": "Job not found"}, status_code=404)
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if user is assigned to this job or is admin
    if job.assigned_to != user.id and not user.is_admin:
        if x_requested_with == "fetch":
            return JSONResponse({"error": "You are not authorized to finish this validation"}, status_code=403)
        raise HTTPException(status_code=403, detail="You are not authorized to finish this validation")

    # Only allow finish validation for jobs with 2+ rejections
    if not job.rejection_count or job.rejection_count < 2:
        if x_requested_with == "fetch":
            return JSONResponse({"error": "Finish validation is only available for jobs with 2+ rejections"}, status_code=400)
        raise HTTPException(status_code=400, detail="Finish validation is only available for jobs with 2+ rejections")

    # Reset job in CVAT to annotation stage with new state
    client = CVATClient(user.cvat_host, user.cvat_token)
    reset_success = await client.update_job_state(job.cvat_job_id, stage="annotation", state="new")
    await client.close()

    if not reset_success:
        if x_requested_with == "fetch":
            return JSONResponse({"error": "Failed to reset job in CVAT"}, status_code=500)
        raise HTTPException(status_code=500, detail="Failed to reset job in CVAT")

    # Delete the job from queue (it can be reassigned to a different labeler)
    await db.delete(job)
    await db.commit()

    if x_requested_with == "fetch":
        return JSONResponse({"success": True, "message": "Job reset to annotation stage"})

    return RedirectResponse(url="/queue", status_code=303)


@router.post("/queue/assign-task")
async def assign_task_jobs(
    task_name: str = Form(...),
    reviewer_id: int = Form(...),
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin assigns all jobs from a task to a reviewer."""
    # Verify reviewer exists
    result = await db.execute(select(User).where(User.id == reviewer_id))
    reviewer = result.scalar_one_or_none()

    if not reviewer:
        raise HTTPException(status_code=404, detail="Reviewer not found")

    # Get all pending jobs for this task
    result = await db.execute(
        select(QueuedJob).where(
            QueuedJob.task_name == task_name,
            QueuedJob.status.in_([QueueStatus.PENDING, QueueStatus.IN_REVIEW])
        )
    )
    jobs = result.scalars().all()

    # Assign all jobs
    job_count = 0
    for job in jobs:
        job.assigned_to = reviewer_id
        job.assigned_at = datetime.utcnow()
        # Set first_reviewer_id if this is the first assignment
        if not job.first_reviewer_id:
            job.first_reviewer_id = reviewer_id
        job_count += 1

    # Create single notification for all jobs
    if job_count > 0:
        notification = Notification(
            user_id=reviewer_id,
            message=f"You have been assigned {job_count} job(s) to review from {task_name}",
            link="/queue"
        )
        db.add(notification)

    await db.commit()

    return RedirectResponse(url="/queue", status_code=303)


@router.post("/queue/{queue_id}/validate")
async def validate_job(
    queue_id: int,
    action: str = Form(...),
    notes: str = Form(""),
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """Validate or reject a job - accessible by assigned reviewer or admin."""
    result = await db.execute(select(QueuedJob).where(QueuedJob.id == queue_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if user is assigned to this job or is admin
    if job.assigned_to != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="You are not authorized to validate this job")

    if action == "validate":
        # Update CVAT job to validation stage with completed state
        client = CVATClient(user.cvat_host, user.cvat_token)
        update_success = await client.update_job_state(job.cvat_job_id, stage="validation", state="completed")
        await client.close()

        if not update_success:
            raise HTTPException(status_code=500, detail="Failed to update job state in CVAT")

        job.status = QueueStatus.VALIDATED
        job.validated_at = datetime.utcnow()
        job.validated_by = user.id
        job.validation_notes = notes
    else:
        # REJECTION: Reset job in CVAT and remove from queue
        # First, reset the job in CVAT to annotation stage with new state
        client = CVATClient(user.cvat_host, user.cvat_token)
        reset_success = await client.reset_job_for_rework(job.cvat_job_id)
        await client.close()

        if not reset_success:
            raise HTTPException(status_code=500, detail="Failed to reset job in CVAT")

        # Store rejection info before deleting from queue
        # We need to track this for when the job comes back
        rejection_count = (job.rejection_count or 0) + 1
        first_reviewer = job.first_reviewer_id or job.assigned_to

        # Create a record to track the rejected job for auto-reassignment
        tracker = RejectedJobTracker(
            cvat_job_id=job.cvat_job_id,
            cvat_task_id=job.cvat_task_id,
            cvat_host=job.cvat_host,
            task_name=job.task_name,
            first_reviewer_id=first_reviewer,
            rejection_count=rejection_count,
            last_rejection_notes=notes,
            rejected_by=user.id,
            rejected_at=datetime.utcnow()
        )
        db.add(tracker)

        # Notify the annotator that their job was rejected
        if job.completed_by:
            notification = Notification(
                user_id=job.completed_by,
                message=f"Job #{job.cvat_job_id} was rejected: {notes or 'No notes provided'}. Please rework and resubmit.",
                link=f"/task/{job.cvat_task_id}"
            )
            db.add(notification)

        # Delete the job from queue (it will be re-added when completed again)
        await db.delete(job)
        await db.commit()

        return RedirectResponse(url="/queue", status_code=303)

    await db.commit()

    return RedirectResponse(url="/queue", status_code=303)


@router.get("/history", response_class=HTMLResponse)
async def view_history(
    request: Request,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """View validation history."""
    result = await db.execute(
        select(QueuedJob).where(
            QueuedJob.cvat_host == user.cvat_host,
            QueuedJob.status.in_([QueueStatus.VALIDATED, QueueStatus.REJECTED])
        ).order_by(QueuedJob.validated_at.desc()).limit(50)
    )
    jobs = result.scalars().all()

    return templates.TemplateResponse("history.html", {
        "request": request,
        "user": user,
        "jobs": jobs
    })


@router.post("/sync-completed")
async def sync_completed_jobs(
    task_id: int = Form(...),
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """Sync all completed jobs from a task to the queue."""
    service = QueueService(db)
    client = CVATClient(user.cvat_host, user.cvat_token)

    tasks = await client.get_tasks()
    task = next((t for t in tasks if t.id == task_id), None)
    task_name = task.name if task else ""

    count = await service.sync_completed_jobs(client, task_id, task_name, user.id, user.cvat_host)
    await client.close()

    return RedirectResponse(url=f"/task/{task_id}?synced={count}", status_code=303)


@router.post("/admin/toggle/{user_id}")
async def toggle_admin(
    user_id: int,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Toggle admin status for a user."""
    if admin.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot modify own admin status")

    result = await db.execute(select(User).where(User.id == user_id))
    target = result.scalar_one_or_none()

    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    target.is_admin = not target.is_admin
    await db.commit()

    return RedirectResponse(url="/admin/users", status_code=303)


@router.get("/admin/users", response_class=HTMLResponse)
async def admin_users(
    request: Request,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin view of all users with reviewer statistics."""
    from collections import defaultdict
    from datetime import date

    result = await db.execute(select(User).order_by(User.created_at))
    users = result.scalars().all()

    today = date.today()

    # Build reviewer stats - who validated/rejected jobs
    result = await db.execute(
        select(QueuedJob).where(
            QueuedJob.cvat_host == user.cvat_host,
            QueuedJob.validated_by.isnot(None)
        )
    )
    reviewed_jobs = result.scalars().all()

    # Create user lookup for reviewer names
    user_lookup = {u.id: u for u in users}

    # Aggregate stats per reviewer
    reviewer_data = defaultdict(lambda: {
        "total_reviews": 0,
        "validated": 0,
        "rejected": 0,
        "today_reviews": 0,
        "today_validated": 0,
        "today_rejected": 0,
        "is_admin": False
    })

    for job in reviewed_jobs:
        reviewer_id = job.validated_by
        if reviewer_id and reviewer_id in user_lookup:
            reviewer = user_lookup[reviewer_id]
            data = reviewer_data[reviewer.username]
            data["is_admin"] = reviewer.is_admin
            data["total_reviews"] += 1

            if job.status == QueueStatus.VALIDATED:
                data["validated"] += 1
                if job.validated_at and job.validated_at.date() == today:
                    data["today_validated"] += 1
                    data["today_reviews"] += 1
            elif job.status == QueueStatus.REJECTED:
                data["rejected"] += 1
                if job.validated_at and job.validated_at.date() == today:
                    data["today_rejected"] += 1
                    data["today_reviews"] += 1

    # Convert to list
    reviewer_stats_list = []
    for username, data in reviewer_data.items():
        approval_rate = (data["validated"] / data["total_reviews"] * 100) if data["total_reviews"] > 0 else 0
        reviewer_stats_list.append({
            "username": username,
            "is_admin": data["is_admin"],
            "total_reviews": data["total_reviews"],
            "validated": data["validated"],
            "rejected": data["rejected"],
            "approval_rate": approval_rate,
            "today_reviews": data["today_reviews"],
            "today_validated": data["today_validated"],
            "today_rejected": data["today_rejected"]
        })

    # Sort by total reviews descending
    reviewer_stats_list.sort(key=lambda x: x["total_reviews"], reverse=True)

    return templates.TemplateResponse("admin_users.html", {
        "request": request,
        "user": user,
        "users": users,
        "reviewer_stats": reviewer_stats_list
    })


@router.get("/admin/queue", response_class=HTMLResponse)
async def admin_queue(
    request: Request,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin queue management - assign jobs to reviewers."""
    # Get pending jobs that need assignment
    result = await db.execute(
        select(QueuedJob).where(
            QueuedJob.status.in_([QueueStatus.PENDING, QueueStatus.IN_REVIEW])
        ).order_by(QueuedJob.completed_at)
    )
    jobs = result.scalars().all()

    # Get all users who can be assigned (reviewers)
    result = await db.execute(select(User).order_by(User.username))
    all_users = result.scalars().all()

    # Get unique task names for filtering
    task_names = sorted(set(job.task_name for job in jobs if job.task_name))

    return templates.TemplateResponse("admin_queue.html", {
        "request": request,
        "user": user,
        "jobs": jobs,
        "all_users": all_users,
        "task_names": task_names
    })


@router.post("/admin/queue/{queue_id}/assign")
async def assign_job(
    queue_id: int,
    reviewer_id: int = Form(...),
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin assigns a job to a specific reviewer."""
    result = await db.execute(select(QueuedJob).where(QueuedJob.id == queue_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Verify reviewer exists
    result = await db.execute(select(User).where(User.id == reviewer_id))
    reviewer = result.scalar_one_or_none()

    if not reviewer:
        raise HTTPException(status_code=404, detail="Reviewer not found")

    job.assigned_to = reviewer_id
    job.assigned_at = datetime.utcnow()
    await db.commit()

    return RedirectResponse(url="/admin/queue", status_code=303)


@router.post("/admin/queue/assign-bulk")
async def assign_jobs_bulk(
    request: Request,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Bulk assign multiple jobs to a reviewer."""
    form = await request.form()
    reviewer_id = int(form.get("reviewer_id"))
    job_ids = form.getlist("job_ids")

    if not job_ids:
        return RedirectResponse(url="/admin/queue", status_code=303)

    # Verify reviewer exists
    result = await db.execute(select(User).where(User.id == reviewer_id))
    reviewer = result.scalar_one_or_none()

    if not reviewer:
        raise HTTPException(status_code=404, detail="Reviewer not found")

    # Update all selected jobs
    for job_id in job_ids:
        result = await db.execute(select(QueuedJob).where(QueuedJob.id == int(job_id)))
        job = result.scalar_one_or_none()
        if job:
            job.assigned_to = reviewer_id
            job.assigned_at = datetime.utcnow()

    await db.commit()

    return RedirectResponse(url="/admin/queue", status_code=303)


@router.post("/admin/queue/{queue_id}/unassign")
async def unassign_job(
    queue_id: int,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin removes assignment from a job."""
    result = await db.execute(select(QueuedJob).where(QueuedJob.id == queue_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.assigned_to = None
    job.assigned_at = None
    await db.commit()

    return RedirectResponse(url="/admin/queue", status_code=303)


@router.get("/task/{task_id}/analytics", response_class=HTMLResponse)
async def task_analytics(
    request: Request,
    task_id: int,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """View annotation analytics for a task - only loads timeline by default."""
    # Get task info from CVAT
    client = CVATClient(user.cvat_host, user.cvat_token)
    tasks = await client.get_tasks()
    task = next((t for t in tasks if t.id == task_id), None)

    # Get analytics history (timeline)
    analytics = AnalyticsService(db)
    history = await analytics.get_task_history(task_id)

    await client.close()

    return templates.TemplateResponse("task_analytics.html", {
        "request": request,
        "user": user,
        "task": task,
        "task_id": task_id,
        "history": history
    })


@router.get("/task/{task_id}/analytics/jobs")
async def get_jobs_with_annotations(
    task_id: int,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """API endpoint to fetch jobs with annotation counts for a task."""
    client = CVATClient(user.cvat_host, user.cvat_token)
    jobs_with_annotations = await client.get_jobs_with_annotations(task_id)
    await client.close()

    # Format data for response (already dict from cvat_client)
    jobs_data = [
        {
            "id": job["id"],
            "assignee": job["assignee"] or "-",
            "stage": job["stage"],
            "state": job["state"],
            "frame_count": job["frame_count"],
            "annotation_count": job["annotation_count"],
            "label_count": job.get("label_count", 0)
        }
        for job in jobs_with_annotations
    ]

    return JSONResponse({
        "jobs": jobs_data,
        "total_jobs": len(jobs_data),
        "total_annotations": sum(j["annotation_count"] for j in jobs_data),
        "total_frames": sum(j["frame_count"] for j in jobs_data),
        "completed_jobs": sum(1 for j in jobs_data if j["state"] == "completed"),
        "in_progress_jobs": sum(1 for j in jobs_data if j["state"] == "in progress")
    })


@router.post("/task/{task_id}/analytics/refresh")
async def refresh_analytics(
    task_id: int,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """Capture a new annotation and label snapshot for a task."""
    # Get task info
    client = CVATClient(user.cvat_host, user.cvat_token)
    tasks = await client.get_tasks()
    task = next((t for t in tasks if t.id == task_id), None)

    if not task:
        await client.close()
        raise HTTPException(status_code=404, detail="Task not found")

    # Capture snapshots (both annotation and label)
    analytics = AnalyticsService(db)
    await analytics.capture_snapshot(client, task_id, task.name)
    await analytics.capture_label_snapshot(client, task_id, task.name)

    await client.close()

    return RedirectResponse(url=f"/task/{task_id}/analytics", status_code=303)


@router.get("/job/{job_id}/analytics/labels")
async def get_job_labels_statistics(
    job_id: int,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """API endpoint to fetch detailed label statistics for a specific job."""
    try:
        client = CVATClient(user.cvat_host, user.cvat_token)
        label_stats = await client.get_job_labels_statistics(job_id)
        await client.close()

        sorted_labels = sorted(label_stats.items(), key=lambda x: x[1], reverse=True)

        response_data = {
            "job_id": job_id,
            "labels": [{"name": name, "count": count} for name, count in sorted_labels],
            "total_labels": len(sorted_labels),
            "total_annotations": sum(label_stats.values())
        }

        return JSONResponse(response_data)
    except Exception as e:
        print(f"Error in get_job_labels_statistics endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": f"Failed to fetch job label statistics: {str(e)}"},
            status_code=500
        )


@router.get("/task/{task_id}/analytics/labels/history")
async def get_labels_history(
    task_id: int,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """API endpoint to fetch label statistics history for interactive charts."""
    import json
    from collections import defaultdict

    try:
        analytics = AnalyticsService(db)
        history = await analytics.get_label_history(task_id)

        if not history:
            return JSONResponse({
                "has_data": False,
                "message": "No historical data available. Click 'Refresh Data' to start tracking."
            })

        # Parse snapshots and organize data for time series
        timeline_data = defaultdict(list)
        timestamps = []

        for snapshot in history:
            label_stats = json.loads(snapshot.label_stats)
            timestamp = snapshot.snapshot_time.strftime('%Y-%m-%d %H:%M')
            timestamps.append(timestamp)

            for label_name, count in label_stats.items():
                timeline_data[label_name].append(count)

        # Get latest snapshot for current stats
        latest = history[-1]
        latest_stats = json.loads(latest.label_stats)
        sorted_labels = sorted(latest_stats.items(), key=lambda x: x[1], reverse=True)

        response_data = {
            "has_data": True,
            "timestamps": timestamps,
            "timeline_data": dict(timeline_data),
            "current_labels": [{"name": name, "count": count} for name, count in sorted_labels],
            "total_labels": len(sorted_labels),
            "total_annotations": sum(latest_stats.values()),
            "snapshots_count": len(history)
        }

        return JSONResponse(response_data)
    except Exception as e:
        print(f"Error in get_labels_history endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": f"Failed to fetch label history: {str(e)}"},
            status_code=500
        )


@router.get("/task/{task_id}/analytics/rejections")
async def get_task_rejection_stats(
    task_id: int,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """API endpoint to fetch rejection statistics per assignee for a task.

    Combines data from:
    1. RejectedJobTracker - historical rejection records
    2. QueuedJob - current jobs in queue with rejection_count > 0
    """
    from collections import defaultdict

    try:
        # Get CVAT client to fetch job assignees
        client = CVATClient(user.cvat_host, user.cvat_token)
        jobs = await client.get_jobs(task_id)
        await client.close()

        # Build job_id -> assignee mapping
        job_assignees = {job.id: job.assignee or "Unassigned" for job in jobs}

        # Track rejections by assignee and job
        # Structure: {assignee: {job_id: rejection_count}}
        assignee_job_rejections = defaultdict(lambda: defaultdict(int))

        # 1. Get rejection records from RejectedJobTracker (historical)
        result = await db.execute(
            select(RejectedJobTracker)
            .where(RejectedJobTracker.cvat_task_id == task_id)
        )
        trackers = result.scalars().all()

        for tracker in trackers:
            assignee = job_assignees.get(tracker.cvat_job_id, "Unknown")
            # Use the tracker's rejection_count as it represents the cumulative count at that point
            assignee_job_rejections[assignee][tracker.cvat_job_id] = max(
                assignee_job_rejections[assignee][tracker.cvat_job_id],
                tracker.rejection_count
            )

        # 2. Get current jobs in queue with rejection_count > 0
        result = await db.execute(
            select(QueuedJob)
            .where(QueuedJob.cvat_task_id == task_id)
            .where(QueuedJob.rejection_count > 0)
        )
        queued_jobs = result.scalars().all()

        for qjob in queued_jobs:
            assignee = job_assignees.get(qjob.cvat_job_id, "Unknown")
            # Take the max between tracker and queue (they might have different counts)
            assignee_job_rejections[assignee][qjob.cvat_job_id] = max(
                assignee_job_rejections[assignee][qjob.cvat_job_id],
                qjob.rejection_count
            )

        # Aggregate stats by assignee
        stats_list = []
        for assignee, job_rejections in assignee_job_rejections.items():
            total = sum(job_rejections.values())
            unique_jobs = len(job_rejections)
            stats_list.append({
                "assignee": assignee,
                "total_rejections": total,
                "unique_jobs_rejected": unique_jobs
            })

        # Sort by total rejections descending
        stats_list.sort(key=lambda x: x["total_rejections"], reverse=True)

        total_rejections = sum(s["total_rejections"] for s in stats_list)

        return JSONResponse({
            "task_id": task_id,
            "total_rejections": total_rejections,
            "by_assignee": stats_list
        })

    except Exception as e:
        print(f"Error in get_task_rejection_stats endpoint: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": f"Failed to fetch rejection statistics: {str(e)}"},
            status_code=500
        )


@router.get("/notifications", response_class=HTMLResponse)
async def view_notifications(
    request: Request,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """View user notifications."""
    result = await db.execute(
        select(Notification).where(Notification.user_id == user.id)
        .order_by(Notification.created_at.desc()).limit(50)
    )
    notifications = result.scalars().all()

    return templates.TemplateResponse("notifications.html", {
        "request": request,
        "user": user,
        "notifications": notifications
    })


@router.post("/notifications/{notif_id}/read")
async def mark_notification_read(
    notif_id: int,
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """Mark a notification as read."""
    result = await db.execute(
        select(Notification).where(
            Notification.id == notif_id,
            Notification.user_id == user.id
        )
    )
    notification = result.scalar_one_or_none()

    if notification:
        notification.is_read = True
        await db.commit()

    return RedirectResponse(url="/notifications", status_code=303)


@router.post("/notifications/read-all")
async def mark_all_notifications_read(
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """Mark all notifications as read."""
    await db.execute(
        update(Notification).where(
            Notification.user_id == user.id,
            Notification.is_read == False
        ).values(is_read=True)
    )
    await db.commit()

    return RedirectResponse(url="/notifications", status_code=303)


async def get_unread_notification_count(user_id: int, db: AsyncSession) -> int:
    """Helper to get unread notification count for a user."""
    result = await db.execute(
        select(Notification).where(
            Notification.user_id == user_id,
            Notification.is_read == False
        )
    )
    return len(result.scalars().all())


# =====================
# Database Backup Routes (Admin Only)
# =====================

@router.get("/admin/backup", response_class=HTMLResponse)
async def backup_page(
    request: Request,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Admin page to manage database backups."""
    # Ensure backup directory exists
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Get list of existing backups
    backups = []
    if os.path.exists(BACKUP_DIR):
        for filename in sorted(os.listdir(BACKUP_DIR), reverse=True):
            if filename.endswith('.db'):
                filepath = os.path.join(BACKUP_DIR, filename)
                stat = os.stat(filepath)
                backups.append({
                    'filename': filename,
                    'size': round(stat.st_size / 1024, 2),  # KB
                    'created': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })

    # Get current DB size
    db_size = 0
    if os.path.exists(DB_PATH):
        db_size = round(os.stat(DB_PATH).st_size / 1024, 2)  # KB

    return templates.TemplateResponse("admin_backup.html", {
        "request": request,
        "user": user,
        "backups": backups,
        "db_size": db_size
    })


@router.post("/admin/backup/create")
async def create_backup(
    user: User = Depends(require_admin)
):
    """Create a new database backup."""
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Database file not found")

    # Ensure backup directory exists
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Create backup with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"queue_backup_{timestamp}.db"
    backup_path = os.path.join(BACKUP_DIR, backup_filename)

    try:
        shutil.copy2(DB_PATH, backup_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create backup: {str(e)}")

    return RedirectResponse(url="/admin/backup", status_code=303)


@router.get("/admin/backup/download/{filename}")
async def download_backup(
    filename: str,
    user: User = Depends(require_admin)
):
    """Download a specific backup file."""
    # Security: prevent path traversal
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    backup_path = os.path.join(BACKUP_DIR, filename)

    if not os.path.exists(backup_path):
        raise HTTPException(status_code=404, detail="Backup not found")

    return FileResponse(
        path=backup_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@router.post("/admin/backup/delete/{filename}")
async def delete_backup(
    filename: str,
    user: User = Depends(require_admin)
):
    """Delete a specific backup file."""
    # Security: prevent path traversal
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    backup_path = os.path.join(BACKUP_DIR, filename)

    if not os.path.exists(backup_path):
        raise HTTPException(status_code=404, detail="Backup not found")

    try:
        os.remove(backup_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete backup: {str(e)}")

    return RedirectResponse(url="/admin/backup", status_code=303)


@router.get("/api/pending-sync-check")
async def check_pending_sync(
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Check for completed annotation jobs that are not yet in the queue.
    Returns count of jobs per task that need syncing (for admin users only).
    Checks the last 10 tasks for performance.
    """
    if not user.is_admin:
        return JSONResponse({"pending_tasks": [], "total_pending": 0})

    client = CVATClient(user.cvat_host, user.cvat_token)

    try:
        # Get all tasks (limit to recent ones for performance)
        tasks = await client.get_tasks()
        tasks = tasks[:10]  # Check only the last 10 tasks

        # Get all queued job IDs for this CVAT host
        result = await db.execute(
            select(QueuedJob.cvat_job_id).where(QueuedJob.cvat_host == user.cvat_host)
        )
        queued_job_ids = set(row[0] for row in result.all())

        pending_tasks = []
        total_pending = 0

        for task in tasks:
            jobs = await client.get_jobs(task.id)

            # Find jobs that are completed annotation but not queued
            pending_jobs = [
                j for j in jobs
                if j.stage == "annotation" and j.state == "completed" and j.id not in queued_job_ids
            ]

            if pending_jobs:
                pending_tasks.append({
                    "task_id": task.id,
                    "task_name": task.name,
                    "pending_count": len(pending_jobs)
                })
                total_pending += len(pending_jobs)

        await client.close()

        return JSONResponse({
            "pending_tasks": pending_tasks,
            "total_pending": total_pending
        })

    except Exception as e:
        await client.close()
        return JSONResponse({"error": str(e), "pending_tasks": [], "total_pending": 0})


@router.get("/api/project/{project_name}/analytics")
async def get_project_analytics(
    project_name: str,
    task_ids: str,  # Comma-separated task IDs
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get job analytics with label statistics for all tasks in a project.
    Creates/refreshes snapshots for each task and returns jobs grouped by top labels.
    """
    from collections import defaultdict
    import json

    try:
        task_id_list = [int(tid.strip()) for tid in task_ids.split(',') if tid.strip()]

        if not task_id_list:
            return JSONResponse({"error": "No task IDs provided"}, status_code=400)

        client = CVATClient(user.cvat_host, user.cvat_token)
        analytics = AnalyticsService(db)

        # Get task info
        tasks = await client.get_tasks()
        task_lookup = {t.id: t for t in tasks}

        project_data = {
            "project_name": project_name,
            "tasks": [],
            "label_summary": defaultdict(int),
            "jobs_by_label": defaultdict(list),
            "total_frames": 0,
            "validated_frames": 0
        }

        for task_id in task_id_list:
            task = task_lookup.get(task_id)
            if not task:
                continue

            # Capture fresh label snapshot
            await analytics.capture_label_snapshot(client, task_id, task.name)

            # Get jobs with their annotations
            jobs = await client.get_jobs(task_id)
            task_jobs_data = []

            for job in jobs:
                # Get label statistics for each job
                label_stats = await client.get_job_labels_statistics(job.id)

                # Sort labels by count descending
                sorted_labels = sorted(label_stats.items(), key=lambda x: x[1], reverse=True)
                top_labels = sorted_labels[:5]  # Top 5 labels

                job_data = {
                    "job_id": job.id,
                    "task_id": task_id,
                    "task_name": task.name,
                    "assignee": job.assignee or "Unassigned",
                    "stage": job.stage,
                    "state": job.state,
                    "frame_count": job.frame_count,
                    "total_annotations": sum(label_stats.values()),
                    "top_labels": [{"name": name, "count": count} for name, count in top_labels],
                    "all_labels": label_stats
                }
                task_jobs_data.append(job_data)

                # Track frame counts
                project_data["total_frames"] += job.frame_count
                # Validated jobs are in "acceptance" stage with "completed" state
                if job.stage == "acceptance" and job.state == "completed":
                    project_data["validated_frames"] += job.frame_count

                # Aggregate label counts
                for label_name, count in label_stats.items():
                    project_data["label_summary"][label_name] += count
                    # Group jobs by their dominant label
                    if sorted_labels:
                        dominant_label = sorted_labels[0][0]
                        if job.id not in [j["job_id"] for j in project_data["jobs_by_label"][dominant_label]]:
                            project_data["jobs_by_label"][dominant_label].append(job_data)

            project_data["tasks"].append({
                "task_id": task_id,
                "task_name": task.name,
                "jobs_count": len(task_jobs_data),
                "jobs": task_jobs_data
            })

        await client.close()

        # Sort label summary by count
        sorted_summary = sorted(project_data["label_summary"].items(), key=lambda x: x[1], reverse=True)

        # Build tree structure: labels -> jobs
        label_tree = []
        for label_name, total_count in sorted_summary[:10]:  # Top 10 labels
            jobs_with_label = project_data["jobs_by_label"].get(label_name, [])
            # Sort jobs by this label's count
            jobs_sorted = sorted(
                jobs_with_label,
                key=lambda j: j["all_labels"].get(label_name, 0),
                reverse=True
            )
            label_tree.append({
                "label_name": label_name,
                "total_count": total_count,
                "jobs_count": len(jobs_sorted),
                "jobs": jobs_sorted[:10]  # Top 10 jobs for this label
            })

        return JSONResponse({
            "project_name": project_name,
            "tasks_count": len(project_data["tasks"]),
            "total_jobs": sum(t["jobs_count"] for t in project_data["tasks"]),
            "total_annotations": sum(dict(sorted_summary).values()),
            "total_frames": project_data["total_frames"],
            "validated_frames": project_data["validated_frames"],
            "label_tree": label_tree,
            "tasks": project_data["tasks"]
        })

    except Exception as e:
        print(f"Error in get_project_analytics: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


def calculate_iou(box1: list, box2: list) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2] (top-left, bottom-right corners).
    For CVAT rectangles, points are [x1, y1, x2, y2].
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1[0], box1[1], box1[2], box1[3]
    x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[2], box2[3]

    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    # Check if there's an intersection
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def get_bbox_from_points(points: list, shape_type: str) -> list:
    """
    Convert CVAT shape points to bounding box [x1, y1, x2, y2].
    For rectangles: points are already [x1, y1, x2, y2]
    For polygons/polylines: calculate bounding box from all points
    """
    if not points:
        return None

    if shape_type == "rectangle":
        # Rectangle points are [x1, y1, x2, y2]
        if len(points) >= 4:
            return [points[0], points[1], points[2], points[3]]
    else:
        # For polygons, polylines, etc.: points are [x1, y1, x2, y2, x3, y3, ...]
        if len(points) >= 4:
            x_coords = [points[i] for i in range(0, len(points), 2)]
            y_coords = [points[i] for i in range(1, len(points), 2)]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    return None


def get_bbox_centroid(bbox: list) -> tuple:
    """
    Calculate the centroid (center point) of a bounding box.
    bbox is [x1, y1, x2, y2]
    Returns (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)


def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def percentile(data: list, p: float) -> float:
    """Calculate percentile of a sorted list."""
    if not data:
        return 0
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (data[c] - data[f]) * (k - f) if c != f else data[f]


def outliers_by_dbscan(points: list, eps: float = 50, min_samples: int = 3) -> list:
    """
    DBSCAN clustering - pure Python implementation.
    Points with label -1 are outliers (not in any cluster).

    Args:
        points: List of (x, y) tuples
        eps: Maximum distance between points in the same cluster
        min_samples: Minimum points to form a cluster

    Returns:
        List of boolean values, True if point is an outlier
    """
    n = len(points)
    if n < min_samples:
        return [False] * n

    labels = [-1] * n  # -1 means unvisited/noise
    cluster_id = 0

    def region_query(point_idx):
        """Find all points within eps distance."""
        neighbors = []
        for i in range(n):
            if euclidean_distance(points[point_idx], points[i]) <= eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(point_idx, neighbors, cluster_id):
        """Expand cluster from a core point."""
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:  # Was noise, now border point
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == -2:  # Unvisited
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_samples:
                    neighbors.extend(new_neighbors)
            i += 1

    # Mark all as unvisited (-2), -1 will mean noise after processing
    labels = [-2] * n

    for i in range(n):
        if labels[i] != -2:  # Already processed
            continue

        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            labels[i] = -1  # Mark as noise
        else:
            expand_cluster(i, neighbors, cluster_id)
            cluster_id += 1

    # Return True for outliers (label == -1)
    return [label == -1 for label in labels]


def outliers_by_class_aware(boxes: list, points: list, min_class_size: int = 2, k: float = 2.5) -> list:
    """
    Class-aware outlier detection - detect outliers within each label class separately.
    Points are only compared to others of the same class.

    Args:
        boxes: List of dicts with 'id', 'bbox', 'label_id' keys
        points: List of (x, y) centroid tuples
        min_class_size: Minimum points in a class to check for outliers
        k: IQR multiplier for threshold

    Returns:
        List of boolean values, True if point is an outlier
    """
    n = len(points)
    if n < 2:
        return [False] * n

    is_outlier = [False] * n

    # Group indices by label_id (class)
    classes = {}
    for i, box in enumerate(boxes):
        label_id = box.get("label_id", 0)
        if label_id not in classes:
            classes[label_id] = []
        classes[label_id].append(i)

    # Calculate global centroid for edge cases
    global_centroid = (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n
    )

    for label_id, indices in classes.items():
        class_points = [points[i] for i in indices]
        class_size = len(class_points)

        # Skip classes with too few points
        if class_size < min_class_size:
            continue

        # For classes with only 2 points, check if they're far apart
        if class_size == 2:
            dist = euclidean_distance(class_points[0], class_points[1])
            # Normalize by image diagonal (assume ~1000px as reference)
            normalized_dist = dist / 1000
            if normalized_dist > 0.3:
                # Flag the one farther from global centroid
                dists_to_global = [euclidean_distance(p, global_centroid) for p in class_points]
                outlier_local_idx = 0 if dists_to_global[0] > dists_to_global[1] else 1
                is_outlier[indices[outlier_local_idx]] = True
            continue

        # For larger classes, use IQR within class
        class_centroid = (
            sum(p[0] for p in class_points) / class_size,
            sum(p[1] for p in class_points) / class_size
        )

        distances = [euclidean_distance(p, class_centroid) for p in class_points]
        sorted_distances = sorted(distances)

        q1 = percentile(sorted_distances, 25)
        q3 = percentile(sorted_distances, 75)
        iqr = q3 - q1

        # More lenient threshold for small classes
        adaptive_k = k if class_size >= 10 else k + 0.5

        # Minimum IQR floor
        max_dist = max(distances) if distances else 0
        min_iqr = 0.05 * max_dist if max_dist > 0 else 0.01
        iqr = max(iqr, min_iqr)

        threshold = q3 + adaptive_k * iqr

        for local_idx, dist in enumerate(distances):
            if dist > threshold:
                is_outlier[indices[local_idx]] = True

    return is_outlier


def detect_centroid_outliers(boxes: list, min_votes: int = 2) -> list:
    """
    Consensus voting outlier detection combining DBSCAN and Class-aware methods.
    Only flags as outlier if both methods agree (when min_votes=2).

    Args:
        boxes: List of dicts with 'id', 'bbox', 'label_id' keys
        min_votes: Minimum number of methods that must agree (1 or 2)
                   - 1: Either method flags it (union)
                   - 2: Both methods must agree (intersection)

    Returns:
        List of box IDs that are outliers
    """
    # Need more boxes for meaningful consensus detection
    if len(boxes) < 6:
        return []

    # Calculate centroids for all boxes
    points = []
    for box in boxes:
        centroid = get_bbox_centroid(box["bbox"])
        points.append(centroid)

    # Method 1: DBSCAN (density-based clustering)
    # eps is adaptive based on point spread - more generous to reduce false positives
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        # Use 30% of spread (more generous) with higher minimum
        eps = max(spread * 0.30, 80)
    else:
        eps = 100

    # Require more points to form a cluster (reduces noise flagging)
    min_samples = max(3, len(points) // 4)  # At least 25% of points to form cluster
    dbscan_outliers = outliers_by_dbscan(points, eps=eps, min_samples=min_samples)

    # Method 2: Class-aware outlier detection with stricter threshold (k=3.0)
    class_outliers = outliers_by_class_aware(boxes, points, min_class_size=3, k=3.0)

    # Consensus voting - require BOTH methods to agree
    outlier_ids = []
    for i, box in enumerate(boxes):
        votes = 0
        if dbscan_outliers[i]:
            votes += 1
        if class_outliers[i]:
            votes += 1

        if votes >= min_votes:
            outlier_ids.append(box["id"])

    return outlier_ids


@router.get("/api/job/{job_id}/fast-validation")
async def fast_validation_check(
    job_id: int,
    threshold: float = 0.5,  # Default 50% overlap threshold
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Fast validation check for a single job.
    Runs IoU overlap and centroid outlier detection.
    Returns JSON with issues found or passed status.
    """
    try:
        client = CVATClient(user.cvat_host, user.cvat_token)

        # Get annotations with bounding boxes
        annotations = await client.get_job_annotations_with_boxes(job_id)
        await client.close()

        if "error" in annotations:
            return JSONResponse({
                "status": "error",
                "error": annotations.get("error", "Unknown error")
            }, status_code=500)

        frames_with_overlaps = []
        frames_with_outliers = []

        # Check each frame for overlapping boxes and outliers
        for frame_num, shapes in annotations.get("frames", {}).items():
            # Only check rectangle-type shapes (bounding boxes)
            boxes = []
            for shape in shapes:
                bbox = get_bbox_from_points(shape.get("points", []), shape.get("type", ""))
                if bbox:
                    boxes.append({
                        "id": shape.get("id"),
                        "bbox": bbox,
                        "label_id": shape.get("label_id")
                    })

            # Check all pairs of boxes for overlap (IoU check)
            overlaps = []
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = calculate_iou(boxes[i]["bbox"], boxes[j]["bbox"])
                    if iou > threshold:
                        overlaps.append({
                            "box1_id": boxes[i]["id"],
                            "box2_id": boxes[j]["id"],
                            "iou": round(iou * 100, 1)
                        })

            if overlaps:
                frames_with_overlaps.append({
                    "frame": int(frame_num),
                    "overlaps": overlaps,
                    "overlap_count": len(overlaps)
                })

            # Check for centroid outliers (consensus: DBSCAN + class-aware)
            outlier_ids = detect_centroid_outliers(boxes, min_votes=2)
            if outlier_ids:
                frames_with_outliers.append({
                    "frame": int(frame_num),
                    "outlier_box_ids": outlier_ids,
                    "outlier_count": len(outlier_ids),
                    "total_boxes": len(boxes)
                })

        # Determine if job has issues
        has_overlaps = len(frames_with_overlaps) > 0
        has_outliers = len(frames_with_outliers) > 0
        has_issues = has_overlaps or has_outliers

        if has_issues:
            return JSONResponse({
                "status": "issues_found",
                "job_id": job_id,
                "total_overlap_frames": len(frames_with_overlaps),
                "total_overlaps": sum(f["overlap_count"] for f in frames_with_overlaps),
                "frames_with_overlaps": sorted(frames_with_overlaps, key=lambda x: x["frame"]),
                "total_outlier_frames": len(frames_with_outliers),
                "total_outliers": sum(f["outlier_count"] for f in frames_with_outliers),
                "frames_with_outliers": sorted(frames_with_outliers, key=lambda x: x["frame"])
            })
        else:
            return JSONResponse({
                "status": "passed",
                "job_id": job_id,
                "message": "No issues detected"
            })

    except Exception as e:
        print(f"Error in fast_validation_check: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=500)


@router.get("/api/task/{task_id}/pre-acceptance-check")
async def pre_acceptance_check(
    task_id: int,
    threshold: float = 0.5,  # Default 50% overlap threshold
    user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Pre-acceptance algorithm: Check for bounding box overlaps across all jobs in a task.
    Returns Server-Sent Events (SSE) for real-time progress updates.

    Event types:
    - init: Initial info with total jobs count
    - progress: Progress update for each job analyzed
    - result: Final result for a job (with or without issues)
    - complete: Final summary when all jobs are processed
    - error: Error message if something goes wrong
    """
    from starlette.responses import StreamingResponse
    import json
    import asyncio

    async def generate_events():
        try:
            client = CVATClient(user.cvat_host, user.cvat_token)

            # Get all jobs for the task
            jobs = await client.get_jobs(task_id)

            # Filter to only validation/completed jobs
            validation_jobs = [j for j in jobs if j.stage == "validation" and j.state == "completed"]

            # Send init event
            init_data = {
                "type": "init",
                "task_id": task_id,
                "total_jobs": len(jobs),
                "validation_jobs": len(validation_jobs),
                "threshold": threshold * 100
            }
            yield f"data: {json.dumps(init_data)}\n\n"

            if not validation_jobs:
                complete_data = {
                    "type": "complete",
                    "all_passed": True,
                    "message": "No jobs in validation/completed state to check",
                    "jobs_with_issues": []
                }
                yield f"data: {json.dumps(complete_data)}\n\n"
                await client.close()
                return

            jobs_with_issues = []

            for idx, job in enumerate(validation_jobs):
                # Send progress event
                progress_data = {
                    "type": "progress",
                    "current": idx + 1,
                    "total": len(validation_jobs),
                    "job_id": job.id,
                    "assignee": job.assignee or "-",
                    "status": "analyzing"
                }
                yield f"data: {json.dumps(progress_data)}\n\n"

                # Get annotations with bounding boxes
                annotations = await client.get_job_annotations_with_boxes(job.id)

                if "error" in annotations:
                    result_data = {
                        "type": "result",
                        "job_id": job.id,
                        "assignee": job.assignee or "-",
                        "status": "error",
                        "error": annotations.get("error", "Unknown error")
                    }
                    yield f"data: {json.dumps(result_data)}\n\n"
                    continue

                frames_with_overlaps = []
                frames_with_outliers = []

                # Check each frame for overlapping boxes and outliers
                for frame_num, shapes in annotations.get("frames", {}).items():
                    # Only check rectangle-type shapes (bounding boxes)
                    boxes = []
                    for shape in shapes:
                        bbox = get_bbox_from_points(shape.get("points", []), shape.get("type", ""))
                        if bbox:
                            boxes.append({
                                "id": shape.get("id"),
                                "bbox": bbox,
                                "label_id": shape.get("label_id")
                            })

                    # Check all pairs of boxes for overlap (IoU check)
                    overlaps = []
                    for i in range(len(boxes)):
                        for j in range(i + 1, len(boxes)):
                            iou = calculate_iou(boxes[i]["bbox"], boxes[j]["bbox"])
                            if iou > threshold:
                                overlaps.append({
                                    "box1_id": boxes[i]["id"],
                                    "box2_id": boxes[j]["id"],
                                    "iou": round(iou * 100, 1)
                                })

                    if overlaps:
                        frames_with_overlaps.append({
                            "frame": int(frame_num),
                            "overlaps": overlaps,
                            "overlap_count": len(overlaps)
                        })

                    # Check for centroid outliers (consensus: DBSCAN + class-aware)
                    outlier_ids = detect_centroid_outliers(boxes, min_votes=2)
                    if outlier_ids:
                        frames_with_outliers.append({
                            "frame": int(frame_num),
                            "outlier_box_ids": outlier_ids,
                            "outlier_count": len(outlier_ids),
                            "total_boxes": len(boxes)
                        })

                # Determine if job has issues
                has_overlaps = len(frames_with_overlaps) > 0
                has_outliers = len(frames_with_outliers) > 0
                has_issues = has_overlaps or has_outliers

                # Send result event for this job
                if has_issues:
                    job_result = {
                        "job_id": job.id,
                        "assignee": job.assignee or "-",
                        "frames_with_overlaps": sorted(frames_with_overlaps, key=lambda x: x["frame"]),
                        "total_overlap_frames": len(frames_with_overlaps),
                        "total_overlaps": sum(f["overlap_count"] for f in frames_with_overlaps),
                        "frames_with_outliers": sorted(frames_with_outliers, key=lambda x: x["frame"]),
                        "total_outlier_frames": len(frames_with_outliers),
                        "total_outliers": sum(f["outlier_count"] for f in frames_with_outliers)
                    }
                    jobs_with_issues.append(job_result)
                    result_data = {
                        "type": "result",
                        "job_id": job.id,
                        "assignee": job.assignee or "-",
                        "status": "issues_found",
                        "total_overlap_frames": len(frames_with_overlaps),
                        "total_overlaps": sum(f["overlap_count"] for f in frames_with_overlaps),
                        "frames_with_overlaps": sorted(frames_with_overlaps, key=lambda x: x["frame"]),
                        "total_outlier_frames": len(frames_with_outliers),
                        "total_outliers": sum(f["outlier_count"] for f in frames_with_outliers),
                        "frames_with_outliers": sorted(frames_with_outliers, key=lambda x: x["frame"])
                    }
                else:
                    result_data = {
                        "type": "result",
                        "job_id": job.id,
                        "assignee": job.assignee or "-",
                        "status": "passed"
                    }

                yield f"data: {json.dumps(result_data)}\n\n"

            await client.close()

            # Send complete event
            complete_data = {
                "type": "complete",
                "all_passed": len(jobs_with_issues) == 0,
                "total_jobs": len(jobs),
                "validation_jobs": len(validation_jobs),
                "jobs_with_issues_count": len(jobs_with_issues),
                "jobs_with_issues": jobs_with_issues
            }
            yield f"data: {json.dumps(complete_data)}\n\n"

        except Exception as e:
            print(f"Error in pre_acceptance_check: {e}")
            import traceback
            traceback.print_exc()
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
