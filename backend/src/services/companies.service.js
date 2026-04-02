import { prisma } from "../config/prisma.js";
import { httpError } from "../utils/httpError.js";

const membershipSelection = {
  user_id: true,
  company_id: true,
  role: true
};

export const createCompany = async ({ name, logo }, creatorUserId) => {
  if (!name) {
    throw httpError(400, "name is required");
  }

  return prisma.company.create({
    data: {
      name,
      logo,
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

export const updateCompany = async (id, { name, logo }, userId) => {
  await getCompanyById(id, userId);

  return prisma.company.update({
    where: { id },
    data: {
      name,
      logo
    }
  });
};

export const deleteCompany = async (id, userId) => {
  await getCompanyById(id, userId);
  await prisma.company.delete({ where: { id } });
};

export const addMembership = async (companyId, { user_id, role }) => {
  const normalizedRole = (role || "member").trim().toLowerCase();

  if (!user_id) {
    throw httpError(400, "user_id is required");
  }

  if (!normalizedRole) {
    throw httpError(400, "role is required");
  }

  await getCompanyById(companyId);

  const userExists = await prisma.user.findUnique({ where: { id: user_id } });
  if (!userExists) {
    throw httpError(404, "User not found");
  }

  return prisma.companyMembership.upsert({
    where: {
      user_id_company_id: {
        user_id,
        company_id: companyId
      }
    },
    update: { role: normalizedRole },
    create: {
      user_id,
      company_id: companyId,
      role: normalizedRole
    }
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
