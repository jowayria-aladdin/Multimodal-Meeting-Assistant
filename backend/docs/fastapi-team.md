# AI Integration Team Guide

This document explains what the AI-NoteTaker backend expects from the FastAPI service.

## Role of FastAPI

The backend sends uploaded meeting audio to FastAPI for processing. FastAPI is expected to produce:

- transcript
- summary
- tasks
- optional progress updates

The backend is the source of truth for meeting storage, tenant ownership, and final persistence.

## Backend-to-FastAPI request

### `POST /process-audio`

This is the FastAPI endpoint the backend calls using `FASTAPI_PROCESS_URL`.

The backend sends:

- `meetingId`
- `companyId`
- `title`
- `lang` (`en`, `ar`, `cs`)
- `signVideo` (`webm`)
- `wavFile`

Headers sent by the backend:

- `x-internal-secret: <shared secret>`

## Expected request format

The backend currently sends `multipart/form-data` with files included.

Expected fields:

- `meetingId` - text or number
- `companyId` - text or number
- `title` - text
- `lang` - required text: `en`, `ar`, or `cs`
- `signVideo` - file (`webm`)
- `wavFile` - single file (`wav`)

The main meeting video is uploaded to Cloudinary by the client and stored by backend as URL metadata. FastAPI does not receive or process the main video file. The backend forwards only the sign-language video and the single wav file.

## What FastAPI should return

FastAPI should return a successful response quickly after accepting the job. It can either:

- process synchronously and return the final result, or
- acknowledge the request and send progress/final output through the backend callback endpoint

The recommended approach is to process asynchronously and use the backend callback endpoint for updates.

## Callback to the backend

### `POST /api/internal/meetings/:id/callback`

The backend exposes this internal endpoint for FastAPI to send progress and final results.

Required header:

- `X-Internal-Secret: <shared secret>`

## Callback payload

### Progress update

```json
{
  "status": "PROCESSING",
  "progress": 45,
  "stage": "transcribing",
  "message": "Transcribing audio"
}
```

### Completion update

```json
{
  "status": "COMPLETED",
  "progress": 100,
  "stage": "completed",
  "message": "Processing finished",
  "result": {
    "transcript": "...",
    "summary": "...",
    "tasks": [
      {
        "task_text": "Follow up with stakeholders",
        "due_date": null,
        "status": "TODO"
      }
    ]
  }
}
```

### Failure update

```json
{
  "status": "FAILED",
  "progress": 100,
  "stage": "failed",
  "message": "Model error",
  "error": "Stack trace or readable error message"
}
```

## Expected statuses

Use one of these values in the callback:

- `UPLOADED`
- `QUEUED`
- `PROCESSING`
- `COMPLETED`
- `FAILED`
- `CANCELLED`

## Task format

Each extracted task should use a structure similar to this:

```json
{
  "task_text": "Prepare the summary deck",
  "due_date": "2026-04-20T12:00:00.000Z",
  "status": "TODO"
}
```

The backend will store the tasks in its database when it receives a `COMPLETED` callback.

## File cleanup expectation

The backend deletes temporary uploaded audio files after final completion or failure. FastAPI does not need to manage backend temp files.

## Important expectations

- FastAPI should validate the internal secret.
- FastAPI should not invent company ownership; the backend owns tenant logic.
- FastAPI should return predictable JSON.
- FastAPI should keep progress updates small and consistent.
- FastAPI should make sure transcript, summary, and tasks are separated clearly in the final response.

## Recommended processing stages

Suggested stage names:

- `validate_input`
- `transcribe`
- `diarize`
- `summarize`
- `extract_tasks`
- `persist_results`
- `cleanup`

## Summary

The backend expects FastAPI to process meeting audio and provide structured output. The backend will handle meeting persistence, tenant ownership, status updates, and cleanup of temporary files.