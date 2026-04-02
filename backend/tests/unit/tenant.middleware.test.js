import { beforeEach, describe, expect, it, vi } from "vitest";

const { prismaMock } = vi.hoisted(() => ({
  prismaMock: {
    companyMembership: {
      findUnique: vi.fn()
    }
  }
}));

vi.mock("../../src/config/prisma.js", () => ({
  prisma: prismaMock
}));

import {
  requireTenantAdminFromHeader,
  requireTenantMembershipFromHeader
} from "../../src/middlewares/tenantMiddleware.js";

const createRes = () => {
  const res = {
    status: vi.fn(),
    json: vi.fn()
  };
  res.status.mockReturnValue(res);
  return res;
};

describe("tenant middleware unit tests", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns 400 when X-Company-Id is missing", async () => {
    const req = { headers: {}, user: { id: 1 } };
    const res = createRes();
    const next = vi.fn();

    await requireTenantMembershipFromHeader(req, res, next);

    expect(res.status).toHaveBeenCalledWith(400);
    expect(next).not.toHaveBeenCalled();
  });

  it("returns 403 when user is not a member", async () => {
    prismaMock.companyMembership.findUnique.mockResolvedValue(null);

    const req = { headers: { "x-company-id": "1" }, user: { id: 1 } };
    const res = createRes();
    const next = vi.fn();

    await requireTenantMembershipFromHeader(req, res, next);

    expect(res.status).toHaveBeenCalledWith(403);
    expect(next).not.toHaveBeenCalled();
  });

  it("attaches req.tenant and calls next when membership exists", async () => {
    prismaMock.companyMembership.findUnique.mockResolvedValue({ role: "owner" });

    const req = { headers: { "x-company-id": "3" }, user: { id: 42 } };
    const res = createRes();
    const next = vi.fn();

    await requireTenantMembershipFromHeader(req, res, next);

    expect(req.tenant).toEqual({ companyId: 3, role: "owner" });
    expect(next).toHaveBeenCalledOnce();
  });

  it("admin chain rejects non-admin role", async () => {
    const [membershipMiddleware, adminMiddleware] = requireTenantAdminFromHeader;

    prismaMock.companyMembership.findUnique.mockResolvedValue({ role: "member" });

    const req = { headers: { "x-company-id": "3" }, user: { id: 42 } };
    const res = createRes();
    const next = vi.fn();

    await membershipMiddleware(req, res, next);
    expect(next).toHaveBeenCalledOnce();

    adminMiddleware(req, res, next);
    expect(res.status).toHaveBeenCalledWith(403);
  });
});
