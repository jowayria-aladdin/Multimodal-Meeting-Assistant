import { prisma } from "../config/prisma.js";

const ADMIN_ROLES = new Set(["owner", "admin"]);

const parseCompanyId = (value) => {
  const companyId = Number(value);
  return Number.isInteger(companyId) && companyId > 0 ? companyId : null;
};

const assertMembership = async (req, res, next, companyId) => {
  if (!companyId) {
    return res.status(400).json({
      message: "company_id is required. Use X-Company-Id header or company route parameter."
    });
  }

  const membership = await prisma.companyMembership.findUnique({
    where: {
      user_id_company_id: {
        user_id: Number(req.user.id),
        company_id: companyId
      }
    }
  });

  if (!membership) {
    return res.status(403).json({ message: "Access denied for this company" });
  }

  req.tenant = {
    companyId,
    role: membership.role
  };

  return next();
};

export const requireTenantMembershipFromHeader = async (req, res, next) => {
  try {
    const companyId = parseCompanyId(req.headers["x-company-id"]);
    return await assertMembership(req, res, next, companyId);
  } catch (error) {
    return next(error);
  }
};

export const requireTenantMembershipFromCompanyParam = async (req, res, next) => {
  try {
    const companyId = parseCompanyId(req.params.companyId || req.params.id);
    return await assertMembership(req, res, next, companyId);
  } catch (error) {
    return next(error);
  }
};

const requireAdminRole = (req, res, next) => {
  if (!req.tenant || !ADMIN_ROLES.has(req.tenant.role)) {
    return res.status(403).json({ message: "Admin role required for this company" });
  }

  return next();
};

export const requireTenantAdminFromHeader = [
  requireTenantMembershipFromHeader,
  requireAdminRole
];

export const requireTenantAdminFromCompanyParam = [
  requireTenantMembershipFromCompanyParam,
  requireAdminRole
];
