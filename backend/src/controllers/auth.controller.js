import { asyncHandler } from "../utils/asyncHandler.js";
import * as authService from "../services/auth.service.js";

export const register = asyncHandler(async (req, res) => {
  const result = await authService.register(req.body);
  res.status(201).json(result);
});

export const login = asyncHandler(async (req, res) => {
  const result = await authService.login(req.body);
  res.status(200).json(result);
});

export const me = asyncHandler(async (req, res) => {
  const headerCompanyId = Number(req.headers["x-company-id"]);
  const activeCompanyId = Number.isInteger(headerCompanyId) && headerCompanyId > 0
    ? headerCompanyId
    : null;

  const result = await authService.me({
    userId: Number(req.user.id),
    activeCompanyId
  });

  res.status(200).json(result);
});
