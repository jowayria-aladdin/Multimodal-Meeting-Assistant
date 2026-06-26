-- CreateTable
CREATE TABLE "MeetingChunk" (
    "id" SERIAL NOT NULL,
    "meeting_id" INTEGER NOT NULL,
    "company_id" INTEGER NOT NULL,
    "chunk_type" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "embedding" DOUBLE PRECISION[],
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "MeetingChunk_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "MeetingChunk_meeting_id_idx" ON "MeetingChunk"("meeting_id");

-- CreateIndex
CREATE INDEX "MeetingChunk_company_id_idx" ON "MeetingChunk"("company_id");

-- AddForeignKey
ALTER TABLE "MeetingChunk" ADD CONSTRAINT "MeetingChunk_meeting_id_fkey" FOREIGN KEY ("meeting_id") REFERENCES "Meeting"("id") ON DELETE CASCADE ON UPDATE CASCADE;
