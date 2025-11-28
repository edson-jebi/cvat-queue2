"""CVAT API Client - Handles all interactions with CVAT instance."""
from __future__ import annotations

import httpx
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Job:
    """Represents a CVAT job."""
    id: int
    task_id: int
    stage: str
    state: str
    assignee: Optional[str]
    assignee_id: Optional[int]
    frame_count: int


@dataclass
class Task:
    """Represents a CVAT task."""
    id: int
    name: str
    project_id: Optional[int]
    status: str
    size: int
    assignee: Optional[str]


class CVATClient:
    """Client for CVAT REST API."""

    def __init__(self, host: str, token: Optional[str] = None):
        self.host = host.rstrip("/")
        self.token = token
        self._client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Token {self.token}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                headers=self._get_headers(),
                timeout=30.0
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def login(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Authenticate with CVAT and get session token.
        Returns (success, token_or_error_message).
        """
        client = await self._get_client()
        try:
            response = await client.post(
                "/api/auth/login",
                json={"username": username, "password": password}
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("key")
                self._client = None  # Reset client to use new token
                return True, self.token
            return False, f"Login failed: {response.status_code}"
        except httpx.RequestError as e:
            return False, f"Connection error: {str(e)}"

    async def get_tasks(self) -> List[Task]:
        """Fetch all tasks accessible to the user."""
        client = await self._get_client()
        tasks = []
        page = 1

        while True:
            response = await client.get(f"/api/tasks", params={"page": page})
            if response.status_code != 200:
                break

            data = response.json()
            for t in data.get("results", []):
                assignee = t.get("assignee")
                tasks.append(Task(
                    id=t["id"],
                    name=t["name"],
                    project_id=t.get("project_id"),
                    status=t["status"],
                    size=t.get("size", 0),
                    assignee=assignee.get("username") if assignee else None
                ))

            if not data.get("next"):
                break
            page += 1

        return tasks

    async def get_jobs(self, task_id: int) -> List[Job]:
        """Fetch all jobs for a specific task."""
        client = await self._get_client()
        jobs = []
        page = 1

        while True:
            response = await client.get(f"/api/jobs", params={"task_id": task_id, "page": page})

            if response.status_code != 200:
                break

            data = response.json()
            for j in data.get("results", []):
                assignee = j.get("assignee")
                jobs.append(Job(
                    id=j["id"],
                    task_id=j["task_id"],
                    stage=j["stage"],
                    state=j["state"],
                    assignee=assignee.get("username") if assignee else None,
                    assignee_id=assignee.get("id") if assignee else None,
                    frame_count=j.get("stop_frame", 0) - j.get("start_frame", 0) + 1
                ))

            if not data.get("next"):
                break
            page += 1

        return jobs

    async def get_job(self, job_id: int) -> Optional[Job]:
        """Fetch a single job by ID."""
        client = await self._get_client()
        response = await client.get(f"/api/jobs/{job_id}")

        if response.status_code != 200:
            return None

        j = response.json()
        assignee = j.get("assignee")
        return Job(
            id=j["id"],
            task_id=j["task_id"],
            stage=j["stage"],
            state=j["state"],
            assignee=assignee.get("username") if assignee else None,
            assignee_id=assignee.get("id") if assignee else None,
            frame_count=j.get("stop_frame", 0) - j.get("start_frame", 0) + 1
        )

    async def validate_token(self) -> bool:
        """Check if current token is valid."""
        if not self.token:
            return False
        client = await self._get_client()
        response = await client.get("/api/users/self")
        return response.status_code == 200

    async def get_task_annotations_count(self, task_id: int) -> int:
        """
        Get the total number of annotations for a task.
        Returns the count of all shapes/objects in the task.
        """
        client = await self._get_client()
        try:
            # Get task annotations - CVAT API returns all annotations for a task
            response = await client.get(f"/api/tasks/{task_id}/annotations")

            if response.status_code != 200:
                return 0

            data = response.json()
            # Count all types of annotations: shapes, tracks, and tags
            count = 0
            count += len(data.get("shapes", []))
            count += len(data.get("tracks", []))
            count += len(data.get("tags", []))

            return count
        except Exception:
            return 0

    async def update_job_state(self, job_id: int, stage: str, state: str) -> bool:
        """
        Update a job's stage and state in CVAT.

        Args:
            job_id: The CVAT job ID
            stage: The stage to set (e.g., "annotation", "validation", "acceptance")
            state: The state to set (e.g., "new", "in progress", "completed", "rejected")

        Returns:
            True if successful, False otherwise
        """
        client = await self._get_client()
        try:
            response = await client.patch(
                f"/api/jobs/{job_id}",
                json={"stage": stage, "state": state}
            )
            return response.status_code == 200
        except Exception:
            return False

    async def reset_job_for_rework(self, job_id: int) -> bool:
        """
        Reset a job back to annotation stage with new state.
        Used when a job is rejected and needs rework.

        Args:
            job_id: The CVAT job ID

        Returns:
            True if successful, False otherwise
        """
        return await self.update_job_state(job_id, stage="annotation", state="new")

    async def get_job_annotations_count(self, job_id: int) -> int:
        """
        Get the total number of annotations for a specific job.
        Returns the count of all shapes/tracks/tags in the job.
        """
        client = await self._get_client()
        try:
            response = await client.get(f"/api/jobs/{job_id}/annotations")

            if response.status_code != 200:
                return 0

            data = response.json()
            count = 0
            count += len(data.get("shapes", []))
            count += len(data.get("tracks", []))
            count += len(data.get("tags", []))

            return count
        except Exception:
            return 0

    async def get_jobs_with_annotations(self, task_id: int) -> List[dict]:
        """
        Get all jobs for a task with their annotation counts.
        Returns a list of dicts with job info and annotation count.
        """
        jobs = await self.get_jobs(task_id)
        result = []

        for job in jobs:
            ann_count = await self.get_job_annotations_count(job.id)
            result.append({
                "id": job.id,
                "stage": job.stage,
                "state": job.state,
                "assignee": job.assignee,
                "frame_count": job.frame_count,
                "annotation_count": ann_count
            })

        return result
