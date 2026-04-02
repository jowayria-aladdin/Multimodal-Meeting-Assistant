import { Router } from "express";
import {
  addMeetingParticipant,
  createMeeting,
  deleteMeeting,
  getMeetingById,
  listMeetings,
  removeMeetingParticipant,
  updateMeeting
} from "../controllers/meetings.controller.js";
import {
  requireTenantAdminFromHeader,
  requireTenantMembershipFromHeader
} from "../middlewares/tenantMiddleware.js";

const router = Router();

router.post("/", requireTenantMembershipFromHeader, createMeeting);
router.get("/", requireTenantMembershipFromHeader, listMeetings);
router.get("/:id", requireTenantMembershipFromHeader, getMeetingById);
router.patch("/:id", requireTenantMembershipFromHeader, updateMeeting);
router.delete("/:id", requireTenantMembershipFromHeader, deleteMeeting);

router.post("/:id/participants", requireTenantAdminFromHeader, addMeetingParticipant);
router.delete("/:id/participants/:userId", requireTenantAdminFromHeader, removeMeetingParticipant);

export default router;
