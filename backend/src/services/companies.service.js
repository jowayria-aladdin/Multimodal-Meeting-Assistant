import { prisma } from "../config/prisma.js";
import { httpError } from "../utils/httpError.js";

const membershipSelection = {
  user_id: true,
  company_id: true,
  role: true
};

const ALLOWED_MEMBERSHIP_ROLES = new Set(["admin", "member"]);

const resolveUserByEmail = async (email) => {
  const normalizedEmail = typeof email === "string" ? email.trim() : "";

  if (!normalizedEmail) {
    throw httpError(400, "email is required");
  }

  const user = await prisma.user.findUnique({
    where: { email: normalizedEmail }
  });

  if (!user) {
    throw httpError(404, "User not found");
  }

  return user;
};

export const createCompany = async ({ name }, creatorUserId) => {
  if (!name) {
    throw httpError(400, "name is required");
  }

  return prisma.company.create({
    data: {
      name,
      memberships: {
        create: {
          user_id: creatorUserId,
          role: "owner"
        }
      }
    },
    include: {
      memberships: {
        select: membershipSelection
      }
    }
  });
};

export const listCompanies = async (userId) => prisma.company.findMany({
  where: {
    memberships: {
      some: { user_id: userId }
    }
  },
  orderBy: { id: "asc" },
  include: {
    memberships: {
      select: membershipSelection
    }
  }
});

export const getCompanyById = async (id, userId) => {
  const company = await prisma.company.findFirst({
    where: {
      id,
      memberships: {
        some: { user_id: userId }
      }
    },
    include: {
      memberships: {
        select: membershipSelection
      }
    }
  });

  if (!company) {
    throw httpError(404, "Company not found");
  }

  return company;
};

export const updateCompany = async (id, { name }, userId) => {
  await getCompanyById(id, userId);

  return prisma.company.update({
    where: { id },
    data: {
      name
    }
  });
};

export const deleteCompany = async (id, userId) => {
  await getCompanyById(id, userId);
  await prisma.company.delete({ where: { id } });
};

export const addMembership = async (companyId, { email, role }) => {
  const normalizedRole = (role || "member").trim().toLowerCase();

  if (!normalizedRole) {
    throw httpError(400, "role is required");
  }

  if (!ALLOWED_MEMBERSHIP_ROLES.has(normalizedRole)) {
    throw httpError(400, "role must be one of: admin, member");
  }

  const user = await resolveUserByEmail(email);

  const company = await prisma.company.findUnique({ where: { id: companyId } });
  if (!company) {
    throw httpError(404, "Company not found");
  }

  return prisma.companyMembership.upsert({
    where: {
      user_id_company_id: {
        user_id: user.id,
        company_id: companyId
      }
    },
    update: { role: normalizedRole },
    create: {
      user_id: user.id,
      company_id: companyId,
      role: normalizedRole
    }
  });
};

export const updateMembershipRole = async (companyId, userId, { role }) => {
  const normalizedRole = typeof role === "string" ? role.trim().toLowerCase() : "";

  if (!normalizedRole) {
    throw httpError(400, "role is required");
  }

  if (!ALLOWED_MEMBERSHIP_ROLES.has(normalizedRole)) {
    throw httpError(400, "role must be one of: admin, member");
  }

  const existing = await prisma.companyMembership.findUnique({
    where: {
      user_id_company_id: {
        user_id: userId,
        company_id: companyId
      }
    }
  });

  if (!existing) {
    throw httpError(404, "Membership not found");
  }

  return prisma.companyMembership.update({
    where: {
      user_id_company_id: {
        user_id: userId,
        company_id: companyId
      }
    },
    data: {
      role: normalizedRole
    },
    select: membershipSelection
  });
};

export const removeMembership = async (companyId, userId) => {
  const existing = await prisma.companyMembership.findUnique({
    where: {
      user_id_company_id: {
        user_id: userId,
        company_id: companyId
      }
    }
  });

  if (!existing) {
    throw httpError(404, "Membership not found");
  }

  await prisma.companyMembership.delete({
    where: {
      user_id_company_id: {
        user_id: userId,
        company_id: companyId
      }
    }
  });
};
