import { prisma } from "../config/prisma.js";
import { httpError } from "../utils/httpError.js";

const sanitizeUser = (user) => ({
  id: user.id,
  username: user.username,
  email: user.email
});

export const listUsers = async (companyId) => {
  const users = await prisma.user.findMany({
    where: {
      companyMemberships: {
        some: {
          company_id: companyId
        }
      }
    },
    orderBy: { id: "asc" }
  });
  return users.map(sanitizeUser);
};

export const getUserById = async (id, companyId) => {
  const user = await prisma.user.findFirst({
    where: {
      id,
      companyMemberships: {
        some: {
          company_id: companyId
        }
      }
    }
  });
  if (!user) {
    throw httpError(404, "User not found");
  }
  return sanitizeUser(user);
};
