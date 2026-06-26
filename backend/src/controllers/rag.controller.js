import { asyncHandler } from "../utils/asyncHandler.js";
import * as ragService from "../services/rag.service.js";

export const query = asyncHandler(async (req, res) => {
  const result = await ragService.queryRag(
    req.tenant.companyId,
    req.body.question,
    { meetingIds: req.body.meetingIds, topK: req.body.topK }
  );
  res.status(200).json(result);
});

export const indexMeeting = asyncHandler(async (req, res) => {
  const result = await ragService.indexMeeting(
    Number(req.params.id),
    req.tenant.companyId
  );
  res.status(200).json(result);
});

export const getStatus = asyncHandler(async (req, res) => {
  const status = await ragService.getIndexStatus(
    Number(req.params.id),
    req.tenant.companyId
  );
  res.status(200).json(status);
});

export const listIndexed = asyncHandler(async (req, res) => {
  const meetings = await ragService.listIndexedMeetings(req.tenant.companyId);
  res.status(200).json(meetings);
});
