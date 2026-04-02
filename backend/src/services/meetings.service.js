import { prisma } from "../config/prisma.js";
import { httpError } from "../utils/httpError.js";

const participantSelection = {
  meeting_id: true,
  user_id: true
};

export const createMeeting = async (companyId, { title, transcript, summary }) => {
  if (!title) {
    throw httpError(400, "title is required");
  }

  const company = await prisma.company.findUnique({ where: { id: companyId } });
  if (!company) {
    throw httpError(404, "Company not found");
  }

  return prisma.meeting.create({
    data: {
      company_id: companyId,
      title,
      transcript,
      summary
    }
  });
};

export const listMeetings = async (companyId) => prisma.meeting.findMany({
  where: { company_id: companyId },
  orderBy: { created_at: "desc" },
  include: {
    meetingParticipants: {
      select: participantSelection
    }
  }
});

export const getMeetingById = async (id, companyId) => {
  const meeting = await prisma.meeting.findFirst({
    where: {
      id,
      company_id: companyId
    },
    include: {
      meetingParticipants: {
        select: participantSelection
      },
      tasks: true
    }
  });

  if (!meeting) {
    throw httpError(404, "Meeting not found");
  }

  return meeting;
};

export const updateMeeting = async (id, companyId, payload) => {
  await getMeetingById(id, companyId);

  return prisma.meeting.update({
    where: { id },
    data: {
      title: payload.title,
      transcript: payload.transcript,
      summary: payload.summary
    }
  });
};

export const deleteMeeting = async (id, companyId) => {
  await getMeetingById(id, companyId);
  await prisma.meeting.delete({ where: { id } });
};

export const addMeetingParticipant = async (meetingId, userId, companyId) => {
  const [meeting, user] = await Promise.all([
    prisma.meeting.findFirst({ where: { id: meetingId, company_id: companyId } }),
    prisma.user.findFirst({
      where: {
        id: userId,
        companyMemberships: {
          some: { company_id: companyId }
        }
      }
    })
  ]);

  if (!meeting) {
    throw httpError(404, "Meeting not found");
  }

  if (!user) {
    throw httpError(404, "User not found");
  }

  return prisma.meetingParticipants.upsert({
    where: {
      meeting_id_user_id: {
        meeting_id: meetingId,
        user_id: userId
      }
    },
    update: {},
    create: {
      meeting_id: meetingId,
      user_id: userId
    }
  });
};

export const removeMeetingParticipant = async (meetingId, userId, companyId) => {
  await getMeetingById(meetingId, companyId);

  const existing = await prisma.meetingParticipants.findUnique({
    where: {
      meeting_id_user_id: {
        meeting_id: meetingId,
        user_id: userId
      }
    }
  });

  if (!existing) {
    throw httpError(404, "Participant not found in meeting");
  }

  await prisma.meetingParticipants.delete({
    where: {
      meeting_id_user_id: {
        meeting_id: meetingId,
        user_id: userId
      }
    }
  });
};
