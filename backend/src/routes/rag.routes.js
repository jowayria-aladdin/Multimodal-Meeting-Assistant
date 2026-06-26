import { Router } from "express";
import { query, indexMeeting, getStatus, listIndexed } from "../controllers/rag.controller.js";
import { requireTenantMembershipFromHeader } from "../middlewares/tenantMiddleware.js";

const router = Router();

router.use(requireTenantMembershipFromHeader);

router.post("/query", query);
router.get("/meetings", listIndexed);
router.post("/meetings/:id/index", indexMeeting);
router.get("/meetings/:id/status", getStatus);

export default router;
