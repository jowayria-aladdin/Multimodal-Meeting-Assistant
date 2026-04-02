import request from "supertest";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { authServiceMock } = vi.hoisted(() => ({
  authServiceMock: {
    register: vi.fn(),
    login: vi.fn()
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
});
