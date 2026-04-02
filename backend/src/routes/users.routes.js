import { Router } from "express";
import { getUserById, listUsers } from "../controllers/users.controller.js";
import { requireTenantMembershipFromHeader } from "../middlewares/tenantMiddleware.js";

const router = Router();

router.get("/", requireTenantMembershipFromHeader, listUsers);
router.get("/:id", requireTenantMembershipFromHeader, getUserById);

export default router;
