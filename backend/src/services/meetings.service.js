import { prisma } from "../config/prisma.js";
import { env } from "../config/env.js";
import { httpError } from "../utils/httpError.js";
import { promises as fs } from "fs";
import path from "path";

const participantSelection = {
  meeting_id: true,
  user_id: true
};

const FINAL_STATUSES = new Set(["COMPLETED", "FAILED", "CANCELLED"]);
const meetingSubscribers = new Map();
const ALLOWED_UPLOAD_LANGUAGES = new Set(["en", "ar", "cs"]);
const CLOUDINARY_HOST = "res.cloudinary.com";

const normalizeUploadedFiles = (files) => {
  const wavFile = files?.wavFile?.[0] || null;

  if (!wavFile) {
    throw httpError(400, "wavFile is required");
  }

  return {
    signVideoPath: files?.signVideo?.[0]?.path || null,
    wavPath: wavFile.path
  };
};

const normalizeMainVideoMetadata = (payload) => {
  const mainVideoUrl = typeof payload.mainVideoUrl === "string" ? payload.mainVideoUrl.trim() : "";
  if (!mainVideoUrl) {
    throw httpError(400, "mainVideoUrl is required");
  }

  let parsedUrl;
  try {
    parsedUrl = new URL(mainVideoUrl);
  } catch {
    throw httpError(400, "mainVideoUrl must be a valid URL");
  }

  if (!parsedUrl.hostname.includes(CLOUDINARY_HOST)) {
    throw httpError(400, "mainVideoUrl must be a Cloudinary delivery URL");
  }

  const mainVideoPublicId = typeof payload.mainVideoPublicId === "string" && payload.mainVideoPublicId.trim()
    ? payload.mainVideoPublicId.trim()
    : null;

  return {
    mainVideoUrl,
    mainVideoPublicId
  };
};

const normalizeUploadLanguage = (lang) => {
  if (lang === undefined || lang === null || lang === "") {
    throw httpError(400, "lang is required and must be one of: en, ar, cs");
  }

  const normalizedLang = String(lang).trim().toLowerCase();
  if (!ALLOWED_UPLOAD_LANGUAGES.has(normalizedLang)) {
    throw httpError(400, "lang must be one of: en, ar, cs");
  }

  return normalizedLang;
};

const resolveUserByEmail = async (email) => {
  const normalizedEmail = typeof email === "string" ? email.trim() : "";

  if (!normalizedEmail) {
    throw httpError(400, "email is required");
  }

  const user = await prisma.user.findUnique({
    where: { email: normalizedEmail }
  });

  if (!user) {
    throw httpError(404, "User not found");
  }

  return user;
};

const emitMeetingEvent = (meetingId, eventType, payload) => {
  const subscribers = meetingSubscribers.get(meetingId);
  if (!subscribers || !subscribers.size) {
    return;
  }

  const eventPayload = JSON.stringify(payload);

  for (const res of subscribers) {
    res.write(`event: ${eventType}\n`);
    res.write(`data: ${eventPayload}\n\n`);
  }
};

const toSsePayload = (meeting) => ({
  meetingId: meeting.id,
  status: meeting.processing_status,
  progress: meeting.progress_percent,
  stage: meeting.status_message || null,
  timestamp: new Date().toISOString(),
  error: meeting.error_message || null
});

const createEventType = (status) => {
  if (status === "COMPLETED") {
    return "meeting.completed";
  }

  if (status === "FAILED") {
    return "meeting.failed";
  }

  if (status === "CANCELLED") {
    return "meeting.cancelled";
  }

  if (status === "QUEUED") {
    return "meeting.queued";
  }

  return "meeting.processing";
};

const cleanupPaths = async (pathsToDelete) => {
  await Promise.all(pathsToDelete.map(async (targetPath) => {
    try {
      await fs.unlink(targetPath);
    } catch (error) {
      if (error.code !== "ENOENT") {
        console.error(`Failed to remove temp file ${targetPath}`, error);
      }
    }
  }));
};

const cleanupMeetingSourceFiles = async (meetingId) => {
  const meeting = await prisma.meeting.findUnique({ where: { id: meetingId } });
  if (!meeting) {
    return;
  }

  const pathsToDelete = [
    ...(meeting.source_webm_path ? [meeting.source_webm_path] : []),
    ...(meeting.source_wav_path ? [meeting.source_wav_path] : [])
  ];

  if (pathsToDelete.length) {
    await cleanupPaths(pathsToDelete);
  }

  await prisma.meeting.update({
    where: { id: meetingId },
    data: {
      source_webm_path: null,
      source_wav_path: null
    }
  });
};

const updateMeetingStatus = async (meetingId, data, eventType) => {
  const updated = await prisma.meeting.update({
    where: { id: meetingId },
    data
  });

  emitMeetingEvent(meetingId, eventType || createEventType(updated.processing_status), toSsePayload(updated));

  if (FINAL_STATUSES.has(updated.processing_status)) {
    await cleanupMeetingSourceFiles(meetingId);
  }

  return updated;
};

const buildFastApiRequestBody = async (meeting, lang) => {
  const formData = new FormData();

  formData.append("meetingId", String(meeting.id));
  formData.append("companyId", String(meeting.company_id));
  formData.append("title", meeting.title);

  formData.append("lang", lang);

  if (meeting.source_webm_path) {
    const signVideoBytes = await fs.readFile(meeting.source_webm_path);
    formData.append(
      "signVideo",
      new Blob([signVideoBytes], { type: "video/webm" }),
      path.basename(meeting.source_webm_path)
    );
  }

  if (meeting.source_wav_path) {
    const wavBytes = await fs.readFile(meeting.source_wav_path);
    formData.append(
      "wavFile",
      new Blob([wavBytes], { type: "audio/wav" }),
      path.basename(meeting.source_wav_path)
    );
  }

  return formData;
};

const sendMeetingToFastApi = async (meeting, lang) => {
  const formData = await buildFastApiRequestBody(meeting, lang);

  const response = await fetch(env.fastApiProcessUrl, {
    method: "POST",
    headers: {
      "x-internal-secret": env.internalCallbackSecret
    },
    body: formData
  });

  if (!response.ok) {
    const failureBody = await response.text();
    throw httpError(502, `FastAPI processing request failed: ${failureBody || response.statusText}`);
  }
};

const startFastApiProcessing = async (meetingId, lang) => {
  const meeting = await prisma.meeting.findUnique({ where: { id: meetingId } });
  if (!meeting) {
    throw httpError(404, "Meeting not found");
  }

  await updateMeetingStatus(meetingId, {
    processing_status: "QUEUED",
    progress_percent: 5,
    status_message: "Queued for AI processing",
    error_message: null,
    processing_started_at: new Date(),
    processing_completed_at: null
  });

  try {
    await sendMeetingToFastApi(meeting, lang);

    await updateMeetingStatus(meetingId, {
      processing_status: "PROCESSING",
      progress_percent: 10,
      status_message: "Audio sent to AI processor"
    });
  } catch (error) {
    await updateMeetingStatus(meetingId, {
      processing_status: "FAILED",
      progress_percent: 100,
      status_message: "Failed to start AI processing",
      error_message: error.message,
      processing_completed_at: new Date()
    });
  }
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

export const createMeetingWithAudio = async (companyId, payload, files) => {
  const { title } = payload;
  const lang = normalizeUploadLanguage(payload.lang);
  const { mainVideoUrl, mainVideoPublicId } = normalizeMainVideoMetadata(payload);

  if (!title) {
    throw httpError(400, "title is required");
  }

  const company = await prisma.company.findUnique({ where: { id: companyId } });
  if (!company) {
    throw httpError(404, "Company not found");
  }

  const { signVideoPath, wavPath } = normalizeUploadedFiles(files);

  const meeting = await prisma.meeting.create({
    data: {
      company_id: companyId,
      title,
      main_video_url: mainVideoUrl,
      main_video_public_id: mainVideoPublicId,
      processing_status: "UPLOADED",
      progress_percent: 0,
      status_message: "Audio uploaded",
      source_webm_path: signVideoPath,
      source_wav_path: wavPath
    }
  });

  emitMeetingEvent(meeting.id, "meeting.uploaded", toSsePayload(meeting));

  void startFastApiProcessing(meeting.id, lang);

  return meeting;
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

export const getMeetingStatusById = async (id, companyId) => {
  const meeting = await getMeetingById(id, companyId);

  return {
    meetingId: meeting.id,
    status: meeting.processing_status,
    progress: meeting.progress_percent,
    stage: meeting.status_message,
    error: meeting.error_message,
    processingStartedAt: meeting.processing_started_at,
    processingCompletedAt: meeting.processing_completed_at
  };
};

export const reprocessMeeting = async (id, companyId) => {
  const meeting = await getMeetingById(id, companyId);

  if (!meeting.source_webm_path || !meeting.source_wav_path) {
    throw httpError(400, "No source audio files found. Upload audio again before reprocessing.");
  }

  void startFastApiProcessing(meeting.id);

  return {
    meetingId: meeting.id,
    status: "QUEUED",
    message: "Meeting reprocessing started"
  };
};

export const subscribeMeetingEvents = async (id, companyId, res) => {
  const meeting = await getMeetingById(id, companyId);

  if (!meetingSubscribers.has(meeting.id)) {
    meetingSubscribers.set(meeting.id, new Set());
  }

  const subscribers = meetingSubscribers.get(meeting.id);
  subscribers.add(res);

  emitMeetingEvent(meeting.id, "meeting.snapshot", toSsePayload(meeting));

  return meeting.id;
};

export const unsubscribeMeetingEvents = (meetingId, res) => {
  const subscribers = meetingSubscribers.get(meetingId);
  if (!subscribers) {
    return;
  }

  subscribers.delete(res);

  if (!subscribers.size) {
    meetingSubscribers.delete(meetingId);
  }
};

export const handleMeetingProcessingCallback = async (meetingId, payload) => {
  const meeting = await prisma.meeting.findUnique({ where: { id: meetingId } });
  if (!meeting) {
    throw httpError(404, "Meeting not found");
  }

  const normalizedStatus = String(payload.status || "").toUpperCase();

  if (!normalizedStatus) {
    throw httpError(400, "status is required in callback payload");
  }

  if (normalizedStatus === "COMPLETED") {
    const transcript = payload.result?.transcript || payload.result?.segments || payload.transcript || null;
    const summary = payload.result?.summary || payload.summary || null;
    const callbackTasks = payload.result?.tasks || payload.tasks || [];

    await prisma.$transaction(async (tx) => {
      await tx.meeting.update({
        where: { id: meetingId },
        data: {
          transcript,
          summary,
          processing_status: "COMPLETED",
          progress_percent: 100,
          status_message: payload.message || "Processing completed",
          error_message: null,
          processing_completed_at: new Date()
        }
      });

      await tx.task.deleteMany({ where: { meeting_id: meetingId } });

      for (const taskItem of callbackTasks) {
        const taskText = taskItem.task_text || taskItem.title;
        if (!taskText) {
          continue;
        }

        await tx.task.create({
          data: {
            meeting_id: meetingId,
            task_text: taskText,
            due_date: taskItem.due_date ? new Date(taskItem.due_date) : null,
            status: taskItem.status || "TODO",
            assignee: taskItem.assignee || null
          }
        });
      }
    });

    const completedMeeting = await prisma.meeting.findUnique({ where: { id: meetingId } });
    emitMeetingEvent(meetingId, "meeting.completed", toSsePayload(completedMeeting));
    await cleanupMeetingSourceFiles(meetingId);

    return completedMeeting;
  }

  if (normalizedStatus === "FAILED") {
    return updateMeetingStatus(meetingId, {
      processing_status: "FAILED",
      progress_percent: 100,
      status_message: payload.message || "Processing failed",
      error_message: payload.error || payload.message || "Processing failed",
      processing_completed_at: new Date()
    }, "meeting.failed");
  }

  const progress = Number(payload.progress);

  return updateMeetingStatus(meetingId, {
    processing_status: normalizedStatus === "QUEUED" ? "QUEUED" : "PROCESSING",
    progress_percent: Number.isFinite(progress) ? Math.max(0, Math.min(99, Math.round(progress))) : meeting.progress_percent,
    status_message: payload.message || payload.stage || "Processing"
  }, normalizedStatus === "QUEUED" ? "meeting.queued" : "meeting.progress");
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

export const addMeetingParticipant = async (meetingId, email, companyId) => {
  const targetUser = await resolveUserByEmail(email);

  const [meeting, companyUser] = await Promise.all([
    prisma.meeting.findFirst({ where: { id: meetingId, company_id: companyId } }),
    prisma.user.findFirst({
      where: {
        id: targetUser.id,
        companyMemberships: {
          some: { company_id: companyId }
        }
      }
    })
  ]);

  if (!meeting) {
    throw httpError(404, "Meeting not found");
  }

  if (!companyUser) {
    throw httpError(404, "User not found");
  }

  return prisma.meetingParticipants.upsert({
    where: {
      meeting_id_user_id: {
        meeting_id: meetingId,
        user_id: targetUser.id
      }
    },
    update: {},
    create: {
      meeting_id: meetingId,
      user_id: targetUser.id
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
