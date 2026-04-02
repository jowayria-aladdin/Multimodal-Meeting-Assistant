import { asyncHandler } from "../utils/asyncHandler.js";
import * as usersService from "../services/users.service.js";

export const listUsers = asyncHandler(async (req, res) => {
  const users = await usersService.listUsers(req.tenant.companyId);
  res.status(200).json(users);
});

export const getUserById = asyncHandler(async (req, res) => {
  const user = await usersService.getUserById(Number(req.params.id), req.tenant.companyId);
  res.status(200).json(user);
});
