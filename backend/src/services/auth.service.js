import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { prisma } from "../config/prisma.js";
import { env } from "../config/env.js";
import { httpError } from "../utils/httpError.js";

const sanitizeUser = (user) => ({
  id: user.id,
  username: user.username,
  email: user.email
});

const issueToken = (user) => jwt.sign(
  { id: user.id, username: user.username, email: user.email },
  env.jwtSecret,
  { expiresIn: env.jwtExpiresIn }
);

export const register = async ({ username, email, password }) => {
  if (!username || !email || !password) {
    throw httpError(400, "username, email and password are required");
  }

  const existing = await prisma.user.findFirst({
    where: {
      OR: [{ username }, { email }]
    }
  });

  if (existing) {
    throw httpError(409, "username or email already exists");
  }

  const password_hash = await bcrypt.hash(password, 12);

  const user = await prisma.user.create({
    data: { username, email, password_hash }
  });

  return {
    user: sanitizeUser(user),
    token: issueToken(user)
  };
};

export const login = async ({ email, password }) => {
  if (!email || !password) {
    throw httpError(400, "email and password are required");
  }

  const user = await prisma.user.findUnique({ where: { email } });
  if (!user) {
    throw httpError(401, "Invalid credentials");
  }

  const match = await bcrypt.compare(password, user.password_hash);
  if (!match) {
    throw httpError(401, "Invalid credentials");
  }

  return {
    user: sanitizeUser(user),
    token: issueToken(user)
  };
};
