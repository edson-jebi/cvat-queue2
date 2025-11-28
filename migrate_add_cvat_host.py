"""Migration script to add cvat_host column to queued_jobs and rejected_job_tracker tables."""

import sqlite3

def migrate():
    conn = sqlite3.connect('queue.db')
    cursor = conn.cursor()

    # Check if cvat_host column exists in queued_jobs
    cursor.execute("PRAGMA table_info(queued_jobs)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'cvat_host' not in columns:
        print("Adding cvat_host column to queued_jobs...")
        # Add column with default empty string first
        cursor.execute("ALTER TABLE queued_jobs ADD COLUMN cvat_host VARCHAR(500) DEFAULT ''")
        # Update existing rows to have a default host (you may need to change this)
        cursor.execute("UPDATE queued_jobs SET cvat_host = 'http://localhost:8080' WHERE cvat_host = ''")
        print("Done!")
    else:
        print("cvat_host column already exists in queued_jobs")

    # Check if cvat_host column exists in rejected_job_tracker
    cursor.execute("PRAGMA table_info(rejected_job_tracker)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'cvat_host' not in columns:
        print("Adding cvat_host column to rejected_job_tracker...")
        cursor.execute("ALTER TABLE rejected_job_tracker ADD COLUMN cvat_host VARCHAR(500) DEFAULT ''")
        cursor.execute("UPDATE rejected_job_tracker SET cvat_host = 'http://localhost:8080' WHERE cvat_host = ''")
        print("Done!")
    else:
        print("cvat_host column already exists in rejected_job_tracker")

    conn.commit()
    conn.close()
    print("Migration complete!")

if __name__ == "__main__":
    migrate()
