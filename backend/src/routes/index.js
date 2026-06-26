import { Router } from "express";
import authRoutes from "./auth.routes.js";
import usersRoutes from "./users.routes.js";
import companiesRoutes from "./companies.routes.js";
import meetingsRoutes from "./meetings.routes.js";
import tasksRoutes from "./tasks.routes.js";
import internalRoutes from "./internal.routes.js";
import ragRoutes from "./rag.routes.js";
import { requireAuth } from "../middlewares/authMiddleware.js";

const router = Router();

router.use("/auth", authRoutes);
router.use("/internal", internalRoutes);
router.use("/users", requireAuth, usersRoutes);
router.use("/companies", requireAuth, companiesRoutes);
router.use("/meetings", requireAuth, meetingsRoutes);
router.use("/tasks", requireAuth, tasksRoutes);
router.use("/rag", requireAuth, ragRoutes);

export const apiRouter = router;
