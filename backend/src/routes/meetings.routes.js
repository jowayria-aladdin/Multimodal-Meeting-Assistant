import { Router } from "express";
import {
  addMeetingParticipant,
  createMeeting,
  createMeetingFromUpload,
  deleteMeeting,
  getMeetingById,
  getMeetingStatusById,
  listMeetings,
  reprocessMeeting,
  removeMeetingParticipant,
  streamMeetingEvents,
  updateMeeting
} from "../controllers/meetings.controller.js";
import {
  requireTenantAdminFromHeader,
  requireTenantMembershipFromHeader
} from "../middlewares/tenantMiddleware.js";
import { uploadMeetingAudio } from "../middlewares/meetingUploadMiddleware.js";

const router = Router();

router.post("/", requireTenantMembershipFromHeader, createMeeting);
router.post("/upload", requireTenantMembershipFromHeader, uploadMeetingAudio, createMeetingFromUpload);
router.get("/", requireTenantMembershipFromHeader, listMeetings);
router.get("/:id/status", requireTenantMembershipFromHeader, getMeetingStatusById);
router.get("/:id/events", requireTenantMembershipFromHeader, streamMeetingEvents);
router.post("/:id/reprocess", requireTenantMembershipFromHeader, reprocessMeeting);
router.get("/:id", requireTenantMembershipFromHeader, getMeetingById);
router.patch("/:id", requireTenantMembershipFromHeader, updateMeeting);
router.delete("/:id", requireTenantMembershipFromHeader, deleteMeeting);

router.post("/:id/participants", requireTenantAdminFromHeader, addMeetingParticipant);
router.delete("/:id/participants/:userId", requireTenantAdminFromHeader, removeMeetingParticipant);

export default router;
