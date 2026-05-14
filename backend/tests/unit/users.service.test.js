import { beforeEach, describe, expect, it, vi } from "vitest";

const { prismaMock } = vi.hoisted(() => ({
  prismaMock: {
    user: {
      findMany: vi.fn(),
      findFirst: vi.fn()
    }
  }
}));

vi.mock("../../src/config/prisma.js", () => ({
  prisma: prismaMock
}));

import { listUsers, getUserById } from "../../src/services/users.service.js";

describe("users.service unit tests", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("listUsers returns users with their company role", async () => {
    prismaMock.user.findMany.mockResolvedValue([
      {
        id: 1,
        username: "admin_user",
        email: "admin@example.com",
        companyMemberships: [
          { company_id: 10, role: "admin" }
        ]
      },
      {
        id: 2,
        username: "member_user",
        email: "member@example.com",
        companyMemberships: [
          { company_id: 10, role: "member" }
        ]
      }
    ]);

    const result = await listUsers(10);

    expect(result).toEqual([
      {
        id: 1,
        username: "admin_user",
        email: "admin@example.com",
        role: "admin"
      },
      {
        id: 2,
        username: "member_user",
        email: "member@example.com",
        role: "member"
      }
    ]);

    expect(prismaMock.user.findMany).toHaveBeenCalledWith({
      where: {
        companyMemberships: {
          some: {
            company_id: 10
          }
        }
      },
      include: {
        companyMemberships: {
          where: {
            company_id: 10
          }
        }
      },
      orderBy: { id: "asc" }
    });
  });

  it("getUserById returns user with their company role", async () => {
    prismaMock.user.findFirst.mockResolvedValue({
      id: 5,
      username: "test_user",
      email: "test@example.com",
      companyMemberships: [
        { company_id: 10, role: "member" }
      ]
    });

    const result = await getUserById(5, 10);

    expect(result).toEqual({
      id: 5,
      username: "test_user",
      email: "test@example.com",
      role: "member"
    });
  });

  it("getUserById returns 404 when user not found in company", async () => {
    prismaMock.user.findFirst.mockResolvedValue(null);

    await expect(getUserById(999, 10)).rejects.toMatchObject({
      status: 404,
      message: "User not found"
    });
  });

  it("getUserById defaults to member role if membership not found", async () => {
    prismaMock.user.findFirst.mockResolvedValue({
      id: 7,
      username: "orphan_user",
      email: "orphan@example.com",
      companyMemberships: []
    });

    const result = await getUserById(7, 10);

    expect(result.role).toBe("member");
  });
});
