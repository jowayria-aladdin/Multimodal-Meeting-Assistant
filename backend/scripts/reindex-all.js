import { prisma } from "../src/config/prisma.js";
import { indexMeeting } from "../src/services/rag.service.js";

const meetings = await prisma.meeting.findMany({
  where: { processing_status: "COMPLETED" },
  select: { id: true, company_id: true, title: true }
});

if (meetings.length === 0) {
  console.log("No completed meetings found.");
  await prisma.$disconnect();
  process.exit(0);
}

console.log(`Found ${meetings.length} completed meeting(s). Starting indexing...\n`);

let success = 0;
let failed = 0;

for (const meeting of meetings) {
  try {
    const result = await indexMeeting(meeting.id, meeting.company_id);
    console.log(`✓ [${meeting.id}] "${meeting.title}" — ${result.chunksCreated} chunks`);
    success++;
  } catch (err) {
    console.error(`✗ [${meeting.id}] "${meeting.title}" — ${err.message}`);
    failed++;
  }
}

console.log(`\nDone. ${success} indexed, ${failed} failed.`);
await prisma.$disconnect();
