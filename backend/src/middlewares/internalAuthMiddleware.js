import { env } from "../config/env.js";

export const requireInternalCallbackSecret = (req, res, next) => {
  const headerSecret = req.headers["x-internal-secret"];

  if (!headerSecret || headerSecret !== env.internalCallbackSecret) {
    return res.status(401).json({ message: "Invalid internal callback secret" });
  }

  return next();
};
