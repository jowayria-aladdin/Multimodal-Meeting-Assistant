import { beforeEach, describe, expect, it, vi } from "vitest";

const { jwtMock } = vi.hoisted(() => ({
  jwtMock: {
    verify: vi.fn()
  }
}));

vi.mock("jsonwebtoken", () => ({ default: jwtMock }));

import { requireAuth } from "../../src/middlewares/authMiddleware.js";

const createRes = () => {
  const res = {
    status: vi.fn(),
    json: vi.fn()
  };
  res.status.mockReturnValue(res);
  return res;
};

describe("auth middleware", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns 401 when no token provided", () => {
    const req = { headers: {}, query: {} };
    const res = createRes();
    const next = vi.fn();

    requireAuth(req, res, next);

    expect(res.status).toHaveBeenCalledWith(401);
    expect(next).not.toHaveBeenCalled();
  });

  it("accepts bearer token from header and attaches user", () => {
    jwtMock.verify.mockReturnValue({ id: 1, username: "u1" });

    const req = { headers: { authorization: "Bearer valid-token" }, query: {} };
    const res = createRes();
    const next = vi.fn();

    requireAuth(req, res, next);

    expect(jwtMock.verify).toHaveBeenCalledWith("valid-token", process.env.JWT_SECRET);
    expect(req.user).toEqual({ id: 1, username: "u1" });
    expect(next).toHaveBeenCalledOnce();
  });

  it("accepts token from query param and attaches user", () => {
    jwtMock.verify.mockReturnValue({ id: 2, username: "u2" });

    const req = { headers: {}, query: { token: "query-token" } };
    const res = createRes();
    const next = vi.fn();

    requireAuth(req, res, next);

    expect(jwtMock.verify).toHaveBeenCalledWith("query-token", process.env.JWT_SECRET);
    expect(req.user).toEqual({ id: 2, username: "u2" });
    expect(next).toHaveBeenCalledOnce();
  });

  it("returns 401 when token invalid", () => {
    jwtMock.verify.mockImplementation(() => { throw new Error("bad"); });

    const req = { headers: { authorization: "Bearer bad-token" }, query: {} };
    const res = createRes();
    const next = vi.fn();

    requireAuth(req, res, next);

    expect(res.status).toHaveBeenCalledWith(401);
    expect(next).not.toHaveBeenCalled();
  });
});
