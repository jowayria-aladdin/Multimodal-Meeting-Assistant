import { beforeEach, describe, expect, it, vi } from "vitest";

const { prismaMock, bcryptMock, jwtMock } = vi.hoisted(() => ({
  prismaMock: {
    user: {
      findFirst: vi.fn(),
      create: vi.fn(),
      findUnique: vi.fn()
    }
  },
  bcryptMock: {
    hash: vi.fn(),
    compare: vi.fn()
  },
  jwtMock: {
    sign: vi.fn()
  }
}));

vi.mock("../../src/config/prisma.js", () => ({
  prisma: prismaMock
}));

vi.mock("bcrypt", () => ({
  default: bcryptMock
}));

vi.mock("jsonwebtoken", () => ({
  default: jwtMock
}));

import { login, register } from "../../src/services/auth.service.js";

describe("auth.service unit tests", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    jwtMock.sign.mockReturnValue("mock-jwt");
  });

  it("register throws 400 when required fields are missing", async () => {
    await expect(register({ username: "u" })).rejects.toMatchObject({
      status: 400,
      message: "username, email and password are required"
    });
  });

  it("register throws 409 when username or email exists", async () => {
    prismaMock.user.findFirst.mockResolvedValue({ id: 1 });

    await expect(
      register({ username: "existing", email: "existing@example.com", password: "Pass@123" })
    ).rejects.toMatchObject({
      status: 409,
      message: "username or email already exists"
    });
  });

  it("register creates user and returns sanitized user + token", async () => {
    prismaMock.user.findFirst.mockResolvedValue(null);
    bcryptMock.hash.mockResolvedValue("hashed-pass");
    prismaMock.user.create.mockResolvedValue({
      id: 10,
      username: "new_user",
      email: "new_user@example.com",
      password_hash: "hashed-pass"
    });

    const result = await register({
      username: "new_user",
      email: "new_user@example.com",
      password: "Pass@123"
    });

    expect(prismaMock.user.create).toHaveBeenCalledWith({
      data: {
        username: "new_user",
        email: "new_user@example.com",
        password_hash: "hashed-pass"
      }
    });
    expect(result).toEqual({
      user: {
        id: 10,
        username: "new_user",
        email: "new_user@example.com"
      },
      token: "mock-jwt"
    });
  });

  it("login throws 401 for invalid credentials", async () => {
    prismaMock.user.findUnique.mockResolvedValue(null);

    await expect(
      login({ email: "wrong@example.com", password: "wrong" })
    ).rejects.toMatchObject({
      status: 401,
      message: "Invalid credentials"
    });
  });

  it("login succeeds with valid credentials", async () => {
    prismaMock.user.findUnique.mockResolvedValue({
      id: 5,
      username: "member_user",
      email: "member@example.com",
      password_hash: "hashed"
    });
    bcryptMock.compare.mockResolvedValue(true);

    const result = await login({ email: "member@example.com", password: "User@123" });

    expect(result).toEqual({
      user: {
        id: 5,
        username: "member_user",
        email: "member@example.com"
      },
      token: "mock-jwt"
    });
  });
});
