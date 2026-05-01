import { beforeEach, describe, expect, it, vi } from "vitest";

const { prismaMock } = vi.hoisted(() => ({
  prismaMock: {
    user: {
      findUnique: vi.fn()
    },
    company: {
      findUnique: vi.fn()
    },
    companyMembership: {
      upsert: vi.fn(),
      findUnique: vi.fn(),
      update: vi.fn(),
      delete: vi.fn()
    }
  }
}));

vi.mock("../../src/config/prisma.js", () => ({
  prisma: prismaMock
}));

import { addMembership, updateMembershipRole } from "../../src/services/companies.service.js";

describe("companies.service unit tests", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("addMembership rejects owner role", async () => {
    await expect(
      addMembership(1, { email: "user@example.com", role: "owner" })
    ).rejects.toMatchObject({
      status: 400,
      message: "role must be one of: admin, member"
    });

    expect(prismaMock.user.findUnique).not.toHaveBeenCalled();
  });

  it("updateMembershipRole rejects invalid role", async () => {
    await expect(
      updateMembershipRole(1, 2, { role: "owner" })
    ).rejects.toMatchObject({
      status: 400,
      message: "role must be one of: admin, member"
    });

    expect(prismaMock.companyMembership.findUnique).not.toHaveBeenCalled();
  });

  it("updateMembershipRole returns 404 when membership does not exist", async () => {
    prismaMock.companyMembership.findUnique.mockResolvedValue(null);

    await expect(
      updateMembershipRole(1, 2, { role: "admin" })
    ).rejects.toMatchObject({
      status: 404,
      message: "Membership not found"
    });
  });

  it("updateMembershipRole updates and returns membership", async () => {
    prismaMock.companyMembership.findUnique.mockResolvedValue({
      user_id: 2,
      company_id: 1,
      role: "member"
    });

    prismaMock.companyMembership.update.mockResolvedValue({
      user_id: 2,
      company_id: 1,
      role: "admin"
    });

    const result = await updateMembershipRole(1, 2, { role: " Admin " });

    expect(prismaMock.companyMembership.update).toHaveBeenCalledWith({
      where: {
        user_id_company_id: {
          user_id: 2,
          company_id: 1
        }
      },
      data: {
        role: "admin"
      },
      select: {
        user_id: true,
        company_id: true,
        role: true
      }
    });

    expect(result).toEqual({
      user_id: 2,
      company_id: 1,
      role: "admin"
    });
  });
});
