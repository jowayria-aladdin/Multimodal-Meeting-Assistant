-- AddSingleWavPath
ALTER TABLE "Meeting"
ADD COLUMN "source_wav_path" TEXT;

UPDATE "Meeting"
SET "source_wav_path" = CASE
  WHEN "source_wav_paths" IS NULL THEN NULL
  ELSE trim(both '"' from "source_wav_paths"::text)
END;

ALTER TABLE "Meeting"
DROP COLUMN "source_wav_paths";