import "dotenv/config";
import bcrypt from "bcrypt";
import { PrismaClient, TaskStatus } from "@prisma/client";

const prisma = new PrismaClient();

async function main() {
  const adminPasswordHash = await bcrypt.hash("Admin@123", 12);
  const userPasswordHash = await bcrypt.hash("User@123", 12);

  // Clear child tables first to respect FK constraints.
  await prisma.taskAssignees.deleteMany();
  await prisma.task.deleteMany();
  await prisma.meetingParticipants.deleteMany();
  await prisma.meeting.deleteMany();
  await prisma.companyMembership.deleteMany();
  await prisma.company.deleteMany();
  await prisma.user.deleteMany();

  const [admin, member] = await Promise.all([
    prisma.user.create({
      data: {
        username: "admin_user",
        email: "admin@example.com",
        password_hash: adminPasswordHash
      }
    }),
    prisma.user.create({
      data: {
        username: "member_user",
        email: "member@example.com",
        password_hash: userPasswordHash
      }
    })
  ]);

  const company = await prisma.company.create({
    data: {
      name: "AI NoteTaker Inc",
      logo: "https://example.com/logo.png"
    }
  });

  await prisma.companyMembership.createMany({
    data: [
      { user_id: admin.id, company_id: company.id, role: "owner" },
      { user_id: member.id, company_id: company.id, role: "member" }
    ]
  });

  const meeting = await prisma.meeting.create({
    data: {
      company_id: company.id,
      title: "Sprint Planning",
      transcript: "https://example.com/transcripts/sprint-planning.txt",
      summary: "Planned sprint goals, assigned owners, and clarified deadlines."
    }
  });

  await prisma.meetingParticipants.createMany({
    data: [
      { meeting_id: meeting.id, user_id: admin.id },
      { meeting_id: meeting.id, user_id: member.id }
    ]
  });

  const [taskOne, taskTwo] = await Promise.all([
    prisma.task.create({
      data: {
        meeting_id: meeting.id,
        task_text: "Prepare project architecture draft",
        due_date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
        status: TaskStatus.IN_PROGRESS
      }
    }),
    prisma.task.create({
      data: {
        meeting_id: meeting.id,
        task_text: "Set up CI pipeline",
        due_date: new Date(Date.now() + 10 * 24 * 60 * 60 * 1000),
        status: TaskStatus.TODO
      }
    })
  ]);

  await prisma.taskAssignees.createMany({
    data: [
      { task_id: taskOne.id, user_id: admin.id },
      { task_id: taskTwo.id, user_id: member.id }
    ]
  });

  console.log("Seeding completed successfully.");
  console.log("Users:", {
    admin: { id: admin.id, email: admin.email, password: "Admin@123" },
    member: { id: member.id, email: member.email, password: "User@123" }
  });
  console.log("Company ID:", company.id);
  console.log("Meeting ID:", meeting.id);
  console.log("Task IDs:", [taskOne.id, taskTwo.id]);
}

main()
  .catch((error) => {
    console.error("Seeding failed:", error);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
