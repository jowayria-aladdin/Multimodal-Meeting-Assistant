/*
  Warnings:

  - The `transcript` column on the `Meeting` table would be dropped and recreated. This will lead to data loss if there is data in the column.

*/
-- AlterTable
ALTER TABLE "Meeting" DROP COLUMN "transcript",
ADD COLUMN     "transcript" JSONB;
