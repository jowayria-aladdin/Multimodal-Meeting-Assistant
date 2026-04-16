# Frontend Team Integration Guide

This document explains how the frontend should talk to the AI-NoteTaker backend for meeting upload and processing.

## What the frontend should send

When creating a meeting from audio, the frontend must send:

- `title` - required, user-defined meeting title
- `lang` - required language code: `en`, `ar`, or `cs`
- `mainVideoUrl` - required Cloudinary playback URL for the main meeting video
- `signVideo` - required sign-language `.webm` file
- `wavFiles` - required, one or more `.wav` files

The frontend should not send `company_id` in the upload request. The backend derives the company from the authenticated user and the selected tenant context.

## Upload endpoint

### `POST /api/meetings/upload`

Content type: `multipart/form-data`

Required headers:

- `Authorization: Bearer <token>`
- `X-Company-Id: <companyId>`

Form fields:

- `title` - text
- `lang` - text (`en`, `ar`, `cs`)
- `mainVideoUrl` - text (Cloudinary URL)
- `mainVideoPublicId` - optional text
- `signVideo` - file (`webm`)
- `wavFiles` - file array

Example form data:

- `title = Sprint Review Audio`
- `lang = en`
- `mainVideoUrl = https://res.cloudinary.com/<cloud>/video/upload/.../main_video.mp4`
- `mainVideoPublicId = main_video_42` (optional)
- `signVideo = <sign-video.webm>`
- `wavFiles = <chunk1.wav>`
- `wavFiles = <chunk2.wav>`

## Main video handling

1. Upload the main meeting video to Cloudinary first.
2. Use the returned `secure_url` as `mainVideoUrl` when calling backend upload.
3. The backend stores this URL and does not send the main video file to FastAPI.
4. The UI should stream the saved Cloudinary URL directly.

## Expected frontend behavior

1. Show upload progress while the files are being sent to the backend.
2. After upload succeeds, treat the returned meeting as the source of truth.
3. Use the meeting status endpoints or event stream to show processing progress.
4. Render transcript, summary, and tasks when processing completes.

## Meeting status endpoints

### `GET /api/meetings/:id/status`

Use this for polling.

Returns:

- `meetingId`
- `status`
- `progress`
- `stage`
- `error`
- `processingStartedAt`
- `processingCompletedAt`

### `GET /api/meetings/:id/events`

Use this for live progress updates with Server-Sent Events.

Events sent by the backend:

- `meeting.uploaded`
- `meeting.snapshot`
- `meeting.queued`
- `meeting.progress`
- `meeting.completed`
- `meeting.failed`
- `meeting.cancelled`

## Status lifecycle

The frontend should expect these statuses:

- `UPLOADED`
- `QUEUED`
- `PROCESSING`
- `COMPLETED`
- `FAILED`
- `CANCELLED`

## Recommended UI flow

1. User chooses a custom title.
2. User uploads one `.webm` file and one or more `.wav` files.
3. Frontend POSTs the multipart form to `/api/meetings/upload`.
4. Backend returns a meeting object with initial processing state.
5. Frontend subscribes to `/api/meetings/:id/events` or polls `/api/meetings/:id/status`.
6. Frontend updates the UI until the meeting is completed or failed.

## Notes

- Use the meeting title as a fully customizable user input.
- Do not trust frontend values for company access; backend tenant auth decides that.
- Do not delete uploaded files in the frontend. The backend handles temporary file cleanup after processing.