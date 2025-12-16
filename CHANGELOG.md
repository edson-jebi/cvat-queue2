# Changelog

All notable changes to the CVAT Queue Manager will be documented in this file.

## [Unreleased]

### Added
- **Task Progress Bar**: Dashboard now displays a visual progress bar showing task completion percentage based on completed jobs
- **Pending Sync Alert**: Real-time dynamic alert for admin users showing completed jobs ready for queue synchronization
  - Animated loading indicator while checking for pending jobs
  - Clickable task links to sync individual tasks
  - "Sync All" button for batch synchronization
  - Toast notifications for sync success/failure
- **Rejection Statistics in Analytics**: Task analytics export now includes per-labeler rejection counts and rejection rates
- **Finish Validation Action**: New action for jobs with 2+ rejections that resets the job to annotation stage
  - Removes job from validation queue
  - Resets CVAT job to stage: annotation, state: new
  - Available for both admin and assigned reviewers

### Changed
- Renamed analytics table to "Performance by Labeler" with new columns: Rejections, Rejection Rate
- CSV export now includes rejection data per labeler
- Action buttons in queue now use flex-wrap for better responsive layout

### UI/UX Improvements
- Added `.apple-btn-warning` button style (orange gradient) for warning actions
- Added `.apple-alert-warning` alert style for pending sync notifications
- Added animated loading dots for sync check indicator
- Progress bars color-coded by completion percentage (red < 25%, yellow < 50%, blue < 75%, green >= 75%)
