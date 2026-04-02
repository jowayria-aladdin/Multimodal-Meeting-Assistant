import { Router } from "express";
import {
  addMembership,
  createCompany,
  deleteCompany,
  getCompanyById,
  listCompanies,
  removeMembership,
  updateCompany
} from "../controllers/companies.controller.js";
import {
  requireTenantAdminFromCompanyParam,
  requireTenantMembershipFromCompanyParam
} from "../middlewares/tenantMiddleware.js";

const router = Router();

router.post("/", createCompany);
router.get("/", listCompanies);
router.get("/:id", requireTenantMembershipFromCompanyParam, getCompanyById);
router.patch("/:id", requireTenantAdminFromCompanyParam, updateCompany);
router.delete("/:id", requireTenantAdminFromCompanyParam, deleteCompany);

router.post("/:id/memberships", requireTenantAdminFromCompanyParam, addMembership);
router.delete("/:id/memberships/:userId", requireTenantAdminFromCompanyParam, removeMembership);

export default router;
