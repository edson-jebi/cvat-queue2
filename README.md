# CVAT Queue Manager

A web-based queue management system for CVAT (Computer Vision Annotation Tool) job validation workflows. Allows teams to manage annotation review processes with job assignment, validation tracking, and analytics.

## Features

- **Job Queue Management**: Automatically queue completed annotation jobs for review
- **User Assignment**: Admin can assign jobs to specific reviewers
- **Validation Workflow**: Approve or reject jobs with notes
- **Rejection Tracking**: Track rejection counts and auto-reassign to original reviewer
- **Multi-Instance Support**: Separate queues for different CVAT instances
- **Analytics Dashboard**: Track annotation progress with timeline charts
- **Notifications**: In-app notifications for job assignments and rejections
- **Database Backup**: Admin backup/restore functionality for data migration
- **Role-Based Access**: Admin and regular user roles

## Screenshots

The application provides:
- Dashboard with task overview
- Queue view with job assignment
- Analytics with annotation timeline charts
- Admin panels for user and backup management

## Requirements

- Python 3.11+
- CVAT instance with API access
- Docker (optional, for containerized deployment)

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cvat-queue.git
cd cvat-queue

# Run in development mode (with hot reload)
./run.sh dev
```

### Production

```bash
# Run in production mode (with Gunicorn)
./run.sh prod
```

### Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or pull from Docker Hub
docker pull YOUR_USERNAME/cvat-queue:latest
docker run -d \
  -p 8000:8000 \
  -v ./data:/app/data \
  -v ./backups:/app/backups \
  --name cvat-queue \
  YOUR_USERNAME/cvat-queue:latest
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Directory for database storage | `./data` |
| `BACKUP_DIR` | Directory for backups | `./backups` |
| `WORKERS` | Number of Gunicorn workers | `CPU * 2 + 1` |
| `LOG_LEVEL` | Logging level (debug/info/warning/error) | `info` |
| `BIND` | Host and port to bind | `0.0.0.0:8000` |

## Usage

### First Time Setup

1. Start the application
2. Navigate to `http://localhost:8000`
3. Register a new account with your CVAT credentials:
   - **CVAT Host**: Your CVAT instance URL (e.g., `http://192.168.1.100:8080`)
   - **CVAT Token**: API token from CVAT (Profile → Settings → API Token)

### User Roles

- **Admin**: Can assign jobs, manage users, create backups
- **User**: Can view assigned jobs and validate them

### Workflow

1. **Sync Jobs**: On task detail page, click "Sync Completed" to queue completed jobs
2. **Assign Jobs**: Admin assigns jobs to reviewers from Queue or Assign page
3. **Start Review**: Reviewer clicks "Start Review" to open job in CVAT
4. **Validate**: After review, approve or reject the job with notes
5. **Rejection Flow**: Rejected jobs return to annotator; when re-completed, they're auto-assigned to the same reviewer

## Project Structure

```
cvat-queue/
├── app/
│   ├── api/
│   │   ├── auth.py          # Authentication routes
│   │   └── routes.py        # Main application routes
│   ├── services/
│   │   ├── cvat_client.py   # CVAT API client
│   │   ├── queue_service.py # Queue management
│   │   └── analytics_service.py
│   ├── templates/           # Jinja2 HTML templates
│   ├── static/              # Static files (CSS, JS)
│   ├── database.py          # Database configuration
│   └── models.py            # SQLAlchemy models
├── data/                    # Database storage
├── backups/                 # Database backups
├── main.py                  # Application entry point
├── gunicorn.conf.py         # Production server config
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose config
└── run.sh                   # Startup script
```

## API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login
- `GET /auth/logout` - Logout

### Main Routes
- `GET /dashboard` - Main dashboard
- `GET /task/{task_id}` - Task detail with jobs
- `GET /queue` - Validation queue
- `GET /history` - Validation history
- `GET /notifications` - User notifications

### Analytics
- `GET /task/{task_id}/analytics` - Task analytics
- `GET /task/{task_id}/analytics/jobs` - Jobs with annotation counts (API)
- `POST /task/{task_id}/analytics/refresh` - Capture new snapshot

### Admin
- `GET /admin/users` - User management
- `GET /admin/queue` - Bulk job assignment
- `GET /admin/backup` - Database backup management

## Database

SQLite database stored in `./data/queue.db`. Tables:

- `users` - User accounts and CVAT credentials
- `queued_jobs` - Jobs in validation queue
- `rejected_job_tracker` - Tracks rejections for auto-reassignment
- `tracked_tasks` - Tasks being monitored
- `annotation_snapshots` - Historical annotation counts
- `notifications` - User notifications

### Backup & Restore

1. Go to Admin → Backup
2. Click "Create Backup" to save current database
3. Download backup file for safekeeping

To restore:
1. Stop the application
2. Replace `data/queue.db` with your backup file
3. Restart the application

## Development

### Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests (if available)
pytest
```

### Adding New Features

1. Models go in `app/models.py`
2. Routes go in `app/api/routes.py`
3. Templates go in `app/templates/`
4. Services go in `app/services/`

## Docker Hub

### Build and Push

```bash
# Build image
docker build -t cvat-queue:latest .

# Tag for Docker Hub
docker tag cvat-queue:latest YOUR_USERNAME/cvat-queue:latest

# Push to Docker Hub
docker login
docker push YOUR_USERNAME/cvat-queue:latest
```

### Or use the build script

```bash
./docker-build.sh YOUR_DOCKERHUB_USERNAME latest
```

## Troubleshooting

### CVAT Connection Issues
- Verify CVAT host URL is correct (include port if not 80/443)
- Ensure API token is valid and not expired
- Check network connectivity between Queue Manager and CVAT

### Job URL Not Opening Correctly
- Make sure CVAT host doesn't have trailing slash
- Verify user has permission to access the job in CVAT

### Database Issues
- Check `data/` directory has write permissions
- For Docker, ensure volume is mounted correctly

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
