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
    jobs_count: int = 0
    completed_jobs_count: int = 0

    @property
    def progress_percent(self) -> int:
        """Calculate completion percentage based on completed jobs."""
        if self.jobs_count == 0:
            return 0
        return int((self.completed_jobs_count / self.jobs_count) * 100)


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

    async def get_tasks(self, include_jobs_progress: bool = False) -> List[Task]:
        """Fetch all tasks accessible to the user.

        Args:
            include_jobs_progress: If True, fetch job counts for progress calculation.
        """
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
                jobs_count = 0
                completed_jobs_count = 0

                # CVAT includes jobs summary in task response
                jobs_summary = t.get("jobs", {})
                if isinstance(jobs_summary, dict):
                    jobs_count = jobs_summary.get("count", 0)
                    completed_jobs_count = jobs_summary.get("completed", 0)

                tasks.append(Task(
                    id=t["id"],
                    name=t["name"],
                    project_id=t.get("project_id"),
                    status=t["status"],
                    size=t.get("size", 0),
                    assignee=assignee.get("username") if assignee else None,
                    jobs_count=jobs_count,
                    completed_jobs_count=completed_jobs_count
                ))

            if not data.get("next"):
                break
            page += 1

        # If jobs progress not in response and requested, fetch separately
        if include_jobs_progress:
            for task in tasks:
                if task.jobs_count == 0:
                    jobs = await self.get_jobs(task.id)
                    task.jobs_count = len(jobs)
                    task.completed_jobs_count = sum(1 for j in jobs if j.state == "completed")

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
        Get all jobs for a task with their annotation counts and label counts.
        Returns a list of dicts with job info, annotation count, and label count.
        """
        jobs = await self.get_jobs(task_id)
        result = []

        for job in jobs:
            ann_count = await self.get_job_annotations_count(job.id)
            label_stats = await self.get_job_labels_statistics(job.id)
            label_count = len([label for label, count in label_stats.items() if count > 0])

            result.append({
                "id": job.id,
                "stage": job.stage,
                "state": job.state,
                "assignee": job.assignee,
                "frame_count": job.frame_count,
                "annotation_count": ann_count,
                "label_count": label_count
            })

        return result

    async def _fetch_all_labels_paginated(self, client, labels_data) -> dict:
        """
        Fetch all labels with pagination support.
        Returns a dict mapping label_id to label_name.
        """
        label_map = {}

        # If labels is a URL reference (paginated), fetch all pages
        if isinstance(labels_data, dict) and "url" in labels_data:
            labels_url = labels_data["url"]
            if labels_url.startswith(self.host):
                labels_url = labels_url[len(self.host):]

            all_labels = []
            page = 1
            while True:
                # Handle URL query params correctly
                separator = "&" if "?" in labels_url else "?"
                page_url = f"{labels_url}{separator}page={page}&page_size=100"

                print(f"Fetching labels page {page}: {page_url}")
                labels_response = await client.get(page_url)

                if labels_response.status_code != 200:
                    print(f"Failed to fetch labels page {page}: {labels_response.status_code}")
                    break

                page_data = labels_response.json()

                if isinstance(page_data, dict) and "results" in page_data:
                    results = page_data["results"]
                    all_labels.extend(results)
                    print(f"Page {page}: fetched {len(results)} labels, total so far: {len(all_labels)}")

                    # Check if there's more pages
                    if not page_data.get("next"):
                        print(f"No more pages. Total labels fetched: {len(all_labels)}")
                        break
                    page += 1
                elif isinstance(page_data, list):
                    all_labels.extend(page_data)
                    print(f"Fetched {len(page_data)} labels (non-paginated response)")
                    break
                else:
                    print(f"Unexpected labels response format: {type(page_data)}")
                    break

            # Build label map from all fetched labels
            for label in all_labels:
                if isinstance(label, dict) and label.get("id"):
                    label_map[label["id"]] = label.get("name", f"Label {label['id']}")

        # If labels is already a list (not paginated)
        elif isinstance(labels_data, list):
            for label in labels_data:
                if isinstance(label, dict) and label.get("id"):
                    label_map[label["id"]] = label.get("name", f"Label {label['id']}")
            print(f"Labels provided as list: {len(label_map)} labels")

        print(f"Final label_map has {len(label_map)} entries")
        return label_map

    async def get_job_labels_statistics(self, job_id: int) -> dict:
        """
        Get label statistics for a specific job with pagination support.
        Returns a dict of {label_name: count}.
        """
        client = await self._get_client()
        try:
            print(f"=== Getting labels statistics for job {job_id} ===")

            # Get job info to find task_id
            job_response = await client.get(f"/api/jobs/{job_id}")
            if job_response.status_code != 200:
                print(f"Failed to get job {job_id}: {job_response.status_code}")
                return {}

            job_data = job_response.json()
            task_id = job_data.get("task_id")
            print(f"Job {job_id} belongs to task {task_id}")

            # Get task to get labels
            task_response = await client.get(f"/api/tasks/{task_id}")
            if task_response.status_code != 200:
                print(f"Failed to get task {task_id}: {task_response.status_code}")
                return {}

            task_data = task_response.json()
            labels = task_data.get("labels", [])

            # Fetch all labels with pagination
            label_map = await self._fetch_all_labels_paginated(client, labels)
            print(f"Built label_map with {len(label_map)} labels")

            # Get job annotations
            response = await client.get(f"/api/jobs/{job_id}/annotations")
            if response.status_code != 200:
                print(f"Failed to get annotations for job {job_id}: {response.status_code}")
                return {}

            data = response.json()
            label_counts = {}

            shapes_count = len(data.get("shapes", []))
            tracks_count = len(data.get("tracks", []))
            tags_count = len(data.get("tags", []))
            print(f"Job {job_id} has {shapes_count} shapes, {tracks_count} tracks, {tags_count} tags")

            # Count shapes
            for shape in data.get("shapes", []):
                label_id = shape.get("label_id") or shape.get("label")
                if label_id and label_id in label_map:
                    label_name = label_map[label_id]
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1
                else:
                    unmapped_label = f"Label {label_id} (*)" if label_id else "Unknown (*)"
                    label_counts[unmapped_label] = label_counts.get(unmapped_label, 0) + 1

            # Count tracks
            for track in data.get("tracks", []):
                label_id = track.get("label_id") or track.get("label")
                if label_id and label_id in label_map:
                    label_name = label_map[label_id]
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1
                else:
                    unmapped_label = f"Label {label_id} (*)" if label_id else "Unknown (*)"
                    label_counts[unmapped_label] = label_counts.get(unmapped_label, 0) + 1

            # Count tags
            for tag in data.get("tags", []):
                label_id = tag.get("label_id") or tag.get("label")
                if label_id and label_id in label_map:
                    label_name = label_map[label_id]
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1
                else:
                    unmapped_label = f"Label {label_id} (*)" if label_id else "Unknown (*)"
                    label_counts[unmapped_label] = label_counts.get(unmapped_label, 0) + 1

            total_annotations = sum(label_counts.values())
            unmapped_count = sum(v for k, v in label_counts.items() if "(*)" in k)
            print(f"Job {job_id}: {total_annotations} total annotations, {unmapped_count} unmapped")
            print(f"=== End job {job_id} ===")

            return label_counts
        except Exception as e:
            print(f"Error getting job labels statistics: {e}")
            import traceback
            traceback.print_exc()
            return {}

    async def get_task_labels_statistics(self, task_id: int) -> dict:
        """
        Get label statistics for a task with pagination support.
        Returns a dict of {label_name: count}.
        """
        client = await self._get_client()
        try:
            print(f"=== Getting labels statistics for task {task_id} ===")

            # Get task to get labels
            task_response = await client.get(f"/api/tasks/{task_id}")
            if task_response.status_code != 200:
                print(f"Failed to get task {task_id}: {task_response.status_code}")
                return {}

            task_data = task_response.json()
            labels = task_data.get("labels", [])

            # Fetch all labels with pagination
            label_map = await self._fetch_all_labels_paginated(client, labels)
            print(f"Built label_map with {len(label_map)} labels for task {task_id}")

            # Get task annotations
            response = await client.get(f"/api/tasks/{task_id}/annotations")
            if response.status_code != 200:
                print(f"Failed to get annotations for task {task_id}: {response.status_code}")
                return {}

            data = response.json()
            label_counts = {}

            shapes_count = len(data.get("shapes", []))
            tracks_count = len(data.get("tracks", []))
            tags_count = len(data.get("tags", []))
            print(f"Task {task_id} has {shapes_count} shapes, {tracks_count} tracks, {tags_count} tags")

            # Count shapes
            for shape in data.get("shapes", []):
                label_id = shape.get("label_id") or shape.get("label")
                if label_id and label_id in label_map:
                    label_name = label_map[label_id]
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1
                else:
                    unmapped_label = f"Label {label_id} (*)" if label_id else "Unknown (*)"
                    label_counts[unmapped_label] = label_counts.get(unmapped_label, 0) + 1

            # Count tracks
            for track in data.get("tracks", []):
                label_id = track.get("label_id") or track.get("label")
                if label_id and label_id in label_map:
                    label_name = label_map[label_id]
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1
                else:
                    unmapped_label = f"Label {label_id} (*)" if label_id else "Unknown (*)"
                    label_counts[unmapped_label] = label_counts.get(unmapped_label, 0) + 1

            # Count tags
            for tag in data.get("tags", []):
                label_id = tag.get("label_id") or tag.get("label")
                if label_id and label_id in label_map:
                    label_name = label_map[label_id]
                    label_counts[label_name] = label_counts.get(label_name, 0) + 1
                else:
                    unmapped_label = f"Label {label_id} (*)" if label_id else "Unknown (*)"
                    label_counts[unmapped_label] = label_counts.get(unmapped_label, 0) + 1

            total_annotations = sum(label_counts.values())
            unmapped_count = sum(v for k, v in label_counts.items() if "(*)" in k)
            print(f"Task {task_id}: {total_annotations} total annotations, {unmapped_count} unmapped")
            print(f"=== End task {task_id} ===")

            return label_counts
        except Exception as e:
            print(f"Error getting task labels statistics: {e}")
            import traceback
            traceback.print_exc()
            return {}
