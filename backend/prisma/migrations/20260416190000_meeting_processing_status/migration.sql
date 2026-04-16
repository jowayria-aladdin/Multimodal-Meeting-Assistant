-- CreateEnum
CREATE TYPE "MeetingProcessingStatus" AS ENUM ('UPLOADED', 'QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELLED');

-- AlterTable
ALTER TABLE "Meeting"
ADD COLUMN "processing_status" "MeetingProcessingStatus" NOT NULL DEFAULT 'UPLOADED',
ADD COLUMN "progress_percent" INTEGER NOT NULL DEFAULT 0,
ADD COLUMN "status_message" TEXT,
ADD COLUMN "processing_started_at" TIMESTAMP(3),
ADD COLUMN "processing_completed_at" TIMESTAMP(3),
ADD COLUMN "error_message" TEXT,
ADD COLUMN "source_webm_path" TEXT,
ADD COLUMN "source_wav_paths" JSONB;