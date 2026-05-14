import { Router } from "express";
import { meetingProcessingCallback } from "../controllers/meetings.controller.js";
import { requireInternalCallbackSecret } from "../middlewares/internalAuthMiddleware.js";

const router = Router();

router.post("/meetings/:id/callback", requireInternalCallbackSecret, meetingProcessingCallback);

export default router;
