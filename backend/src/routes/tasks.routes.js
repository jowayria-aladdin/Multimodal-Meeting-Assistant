import { Router } from "express";
import {
  addTaskAssignee,
  createTask,
  deleteTask,
  getTaskById,
  listTasks,
  removeTaskAssignee,
  updateTask
} from "../controllers/tasks.controller.js";
import {
  requireTenantAdminFromHeader,
  requireTenantMembershipFromHeader
} from "../middlewares/tenantMiddleware.js";

const router = Router();

router.post("/", requireTenantMembershipFromHeader, createTask);
router.get("/", requireTenantMembershipFromHeader, listTasks);
router.get("/:id", requireTenantMembershipFromHeader, getTaskById);
router.patch("/:id", requireTenantMembershipFromHeader, updateTask);
router.delete("/:id", requireTenantMembershipFromHeader, deleteTask);

router.post("/:id/assignees", requireTenantAdminFromHeader, addTaskAssignee);
router.delete("/:id/assignees/:userId", requireTenantAdminFromHeader, removeTaskAssignee);

export default router;
