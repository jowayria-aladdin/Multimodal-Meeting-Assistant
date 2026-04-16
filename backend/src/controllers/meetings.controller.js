import { asyncHandler } from "../utils/asyncHandler.js";
import * as meetingsService from "../services/meetings.service.js";

export const createMeeting = asyncHandler(async (req, res) => {
  const meeting = await meetingsService.createMeeting(req.tenant.companyId, req.body);
  res.status(201).json(meeting);
});

export const createMeetingFromUpload = asyncHandler(async (req, res) => {
  const meeting = await meetingsService.createMeetingWithAudio(req.tenant.companyId, req.body, req.files);
  res.status(201).json(meeting);
});

export const listMeetings = asyncHandler(async (req, res) => {
  const meetings = await meetingsService.listMeetings(req.tenant.companyId);
  res.status(200).json(meetings);
});

export const getMeetingById = asyncHandler(async (req, res) => {
  const meeting = await meetingsService.getMeetingById(Number(req.params.id), req.tenant.companyId);
  res.status(200).json(meeting);
});

export const getMeetingStatusById = asyncHandler(async (req, res) => {
  const meetingStatus = await meetingsService.getMeetingStatusById(
    Number(req.params.id),
    req.tenant.companyId
  );
  res.status(200).json(meetingStatus);
});

export const updateMeeting = asyncHandler(async (req, res) => {
  const meeting = await meetingsService.updateMeeting(
    Number(req.params.id),
    req.tenant.companyId,
    req.body
  );
  res.status(200).json(meeting);
});

export const deleteMeeting = asyncHandler(async (req, res) => {
  await meetingsService.deleteMeeting(Number(req.params.id), req.tenant.companyId);
  res.status(204).send();
});

export const addMeetingParticipant = asyncHandler(async (req, res) => {
  const participant = await meetingsService.addMeetingParticipant(
    Number(req.params.id),
    req.body.email,
    req.tenant.companyId
  );

  res.status(201).json(participant);
});

export const removeMeetingParticipant = asyncHandler(async (req, res) => {
  await meetingsService.removeMeetingParticipant(
    Number(req.params.id),
    Number(req.params.userId),
    req.tenant.companyId
  );
  res.status(204).send();
});

export const reprocessMeeting = asyncHandler(async (req, res) => {
  const result = await meetingsService.reprocessMeeting(
    Number(req.params.id),
    req.tenant.companyId
  );

  res.status(202).json(result);
});

export const streamMeetingEvents = asyncHandler(async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const meetingId = await meetingsService.subscribeMeetingEvents(
    Number(req.params.id),
    req.tenant.companyId,
    res
  );

  req.on("close", () => {
    meetingsService.unsubscribeMeetingEvents(meetingId, res);
    res.end();
  });
});

export const meetingProcessingCallback = asyncHandler(async (req, res) => {
  const meeting = await meetingsService.handleMeetingProcessingCallback(
    Number(req.params.id),
    req.body
  );

  res.status(200).json({
    meetingId: meeting.id,
    status: meeting.processing_status
  });
});
