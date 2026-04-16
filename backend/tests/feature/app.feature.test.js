import request from "supertest";
import jwt from "jsonwebtoken";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { authServiceMock } = vi.hoisted(() => ({
  authServiceMock: {
    register: vi.fn(),
    login: vi.fn(),
    me: vi.fn()
  }
}));

vi.mock("../../src/services/auth.service.js", () => authServiceMock);

import app from "../../src/app.js";

describe("app feature tests", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("GET /health returns status ok", async () => {
    const response = await request(app).get("/health");

    expect(response.statusCode).toBe(200);
    expect(response.body.status).toBe("ok");
  });

  it("GET /docs.json serves OpenAPI spec", async () => {
    const response = await request(app).get("/docs.json");

    expect(response.statusCode).toBe(200);
    expect(response.body.openapi).toBe("3.0.3");
    expect(response.body.info.title).toBe("AI-NoteTaker Backend API");
  });

  it("POST /api/auth/register routes request to auth service", async () => {
    authServiceMock.register.mockResolvedValue({
      user: { id: 1, username: "u1", email: "u1@example.com" },
      token: "jwt-token"
    });

    const payload = {
      username: "u1",
      email: "u1@example.com",
      password: "Pass@123"
    };

    const response = await request(app)
      .post("/api/auth/register")
      .send(payload);

    expect(response.statusCode).toBe(201);
    expect(authServiceMock.register).toHaveBeenCalledWith(payload);
    expect(response.body.token).toBe("jwt-token");
  });

  it("GET /api/users returns 401 without authorization", async () => {
    const response = await request(app).get("/api/users");

    expect(response.statusCode).toBe(401);
    expect(response.body.message).toMatch(/authorization/i);
  });

  it("GET /api/auth/me returns 401 without authorization", async () => {
    const response = await request(app).get("/api/auth/me");

    expect(response.statusCode).toBe(401);
    expect(response.body.message).toMatch(/authorization/i);
  });

  it("GET /api/auth/me routes request to auth service", async () => {
    authServiceMock.me.mockResolvedValue({
      user: { id: 1, username: "u1", email: "u1@example.com" },
      memberships: [
        { companyId: 10, companyName: "Acme", role: "owner" }
      ],
      activeCompanyId: 10,
      activeRole: "owner"
    });

    const token = jwt.sign({ id: 1 }, process.env.JWT_SECRET);

    const response = await request(app)
      .get("/api/auth/me")
      .set("Authorization", `Bearer ${token}`)
      .set("X-Company-Id", "10");

    expect(response.statusCode).toBe(200);
    expect(authServiceMock.me).toHaveBeenCalledWith({ userId: 1, activeCompanyId: 10 });
    expect(response.body.activeRole).toBe("owner");
  });
});
