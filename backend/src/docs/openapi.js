export const openApiSpec = {
  openapi: "3.0.3",
  info: {
    title: "AI-NoteTaker Backend API",
    version: "1.0.0",
    description: "API documentation for auth, company tenancy, meetings, and tasks."
  },
  servers: [
    {
      url: "http://localhost:3000",
      description: "Local development"
    }
  ],
  tags: [
    { name: "Health" },
    { name: "Auth" },
    { name: "Companies" },
    { name: "Users" },
    { name: "Meetings" },
    { name: "Tasks" }
  ],
  components: {
    securitySchemes: {
      bearerAuth: {
        type: "http",
        scheme: "bearer",
        bearerFormat: "JWT"
      },
      tenantHeader: {
        type: "apiKey",
        in: "header",
        name: "X-Company-Id",
        description: "Required for tenant-scoped endpoints."
      }
    },
    schemas: {
      ErrorResponse: {
        type: "object",
        properties: {
          message: { type: "string", example: "Access denied for this company" }
        }
      },
      AuthRequestRegister: {
        type: "object",
        required: ["username", "email", "password"],
        properties: {
          username: { type: "string", example: "admin_user" },
          email: { type: "string", format: "email", example: "admin@example.com" },
          password: { type: "string", format: "password", example: "Admin@123" }
        }
      },
      AuthRequestLogin: {
        type: "object",
        required: ["email", "password"],
        properties: {
          email: { type: "string", format: "email", example: "admin@example.com" },
          password: { type: "string", format: "password", example: "Admin@123" }
        }
      },
      AuthResponse: {
        type: "object",
        properties: {
          user: {
            type: "object",
            properties: {
              id: { type: "integer", example: 1 },
              username: { type: "string", example: "admin_user" },
              email: { type: "string", format: "email", example: "admin@example.com" }
            }
          },
          token: { type: "string", example: "eyJhbGciOiJI..." }
        }
      },
      AuthMeResponse: {
        type: "object",
        properties: {
          user: {
            type: "object",
            properties: {
              id: { type: "integer", example: 1 },
              username: { type: "string", example: "admin_user" },
              email: { type: "string", format: "email", example: "admin@example.com" }
            }
          },
          memberships: {
            type: "array",
            items: {
              type: "object",
              properties: {
                companyId: { type: "integer", example: 1 },
                companyName: { type: "string", example: "AI NoteTaker Inc" },
                role: { type: "string", example: "owner" }
              }
            }
          },
          activeCompanyId: { type: "integer", nullable: true, example: 1 },
          activeRole: { type: "string", nullable: true, example: "owner" }
        }
      },
      Company: {
        type: "object",
        properties: {
          id: { type: "integer", example: 1 },
          name: { type: "string", example: "AI NoteTaker Inc" },
          logo: { type: "string", nullable: true, example: "https://example.com/logo.png" }
        }
      },
      CompanyMembership: {
        type: "object",
        properties: {
          user_id: { type: "integer", example: 2 },
          company_id: { type: "integer", example: 1 },
          role: { type: "string", example: "member" }
        }
      },
      User: {
        type: "object",
        properties: {
          id: { type: "integer", example: 1 },
          username: { type: "string", example: "admin_user" },
          email: { type: "string", format: "email", example: "admin@example.com" }
        }
      },
      Meeting: {
        type: "object",
        properties: {
          id: { type: "integer", example: 1 },
          company_id: { type: "integer", example: 1 },
          title: { type: "string", example: "Weekly Sync" },
          transcript: { type: "string", nullable: true, example: "https://example.com/transcript.txt" },
          summary: { type: "string", nullable: true, example: "Discussed blockers and next steps." },
          created_at: { type: "string", format: "date-time" }
        }
      },
      Task: {
        type: "object",
        properties: {
          id: { type: "integer", example: 1 },
          meeting_id: { type: "integer", example: 1 },
          task_text: { type: "string", example: "Prepare architecture draft" },
          due_date: { type: "string", format: "date-time", nullable: true },
          status: { type: "string", enum: ["TODO", "IN_PROGRESS", "DONE"], example: "TODO" }
        }
      }
    }
  },
  paths: {
    "/health": {
      get: {
        tags: ["Health"],
        summary: "Health check",
        responses: {
          "200": {
            description: "Service is healthy"
          }
        }
      }
    },
    "/api/auth/register": {
      post: {
        tags: ["Auth"],
        summary: "Register user",
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/AuthRequestRegister" }
            }
          }
        },
        responses: {
          "201": {
            description: "User created",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/AuthResponse" }
              }
            }
          }
        }
      }
    },
    "/api/auth/login": {
      post: {
        tags: ["Auth"],
        summary: "Login user",
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/AuthRequestLogin" }
            }
          }
        },
        responses: {
          "200": {
            description: "Login successful",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/AuthResponse" }
              }
            }
          }
        }
      }
    },
    "/api/auth/me": {
      get: {
        tags: ["Auth"],
        summary: "Get current user profile with company memberships",
        security: [{ bearerAuth: [] }],
        parameters: [
          {
            name: "X-Company-Id",
            in: "header",
            required: false,
            schema: { type: "integer" },
            description: "Optional active company selector for activeRole/activeCompanyId"
          }
        ],
        responses: {
          "200": {
            description: "Profile returned",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/AuthMeResponse" }
              }
            }
          },
          "401": { description: "Invalid or missing token" }
        }
      }
    },
    "/api/companies": {
      get: {
        tags: ["Companies"],
        summary: "List companies visible to current user",
        security: [{ bearerAuth: [] }],
        responses: {
          "200": {
            description: "Company list",
            content: {
              "application/json": {
                schema: {
                  type: "array",
                  items: { $ref: "#/components/schemas/Company" }
                }
              }
            }
          }
        }
      },
      post: {
        tags: ["Companies"],
        summary: "Create company (creator becomes owner)",
        security: [{ bearerAuth: [] }],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["name"],
                properties: {
                  name: { type: "string", example: "My Company" },
                  logo: { type: "string", example: "https://example.com/logo.png" }
                }
              }
            }
          }
        },
        responses: {
          "201": { description: "Company created" }
        }
      }
    },
    "/api/companies/{id}": {
      get: {
        tags: ["Companies"],
        summary: "Get company by ID",
        security: [{ bearerAuth: [] }],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "integer" }
          }
        ],
        responses: {
          "200": { description: "Company returned" },
          "403": { description: "Not member of company" }
        }
      },
      patch: {
        tags: ["Companies"],
        summary: "Update company (admin only)",
        security: [{ bearerAuth: [] }],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "integer" }
          }
        ],
        responses: {
          "200": { description: "Company updated" },
          "403": { description: "Admin role required" }
        }
      },
      delete: {
        tags: ["Companies"],
        summary: "Delete company (admin only)",
        security: [{ bearerAuth: [] }],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "integer" }
          }
        ],
        responses: {
          "204": { description: "Company deleted" },
          "403": { description: "Admin role required" }
        }
      }
    },
    "/api/companies/{id}/memberships": {
      post: {
        tags: ["Companies"],
        summary: "Add or update company member (admin only)",
        security: [{ bearerAuth: [] }],
        parameters: [
          {
            name: "id",
            in: "path",
            required: true,
            schema: { type: "integer" }
          }
        ],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["user_id", "role"],
                properties: {
                  user_id: { type: "integer", example: 2 },
                  role: { type: "string", example: "member" }
                }
              }
            }
          }
        },
        responses: {
          "201": {
            description: "Membership created or updated",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/CompanyMembership" }
              }
            }
          }
        }
      }
    },
    "/api/companies/{id}/memberships/{userId}": {
      delete: {
        tags: ["Companies"],
        summary: "Remove company member (admin only)",
        security: [{ bearerAuth: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } },
          { name: "userId", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "204": { description: "Membership deleted" }
        }
      }
    },
    "/api/users": {
      get: {
        tags: ["Users"],
        summary: "List users in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        responses: {
          "200": {
            description: "Tenant users",
            content: {
              "application/json": {
                schema: {
                  type: "array",
                  items: { $ref: "#/components/schemas/User" }
                }
              }
            }
          }
        }
      }
    },
    "/api/users/{id}": {
      get: {
        tags: ["Users"],
        summary: "Get user by ID in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "200": { description: "Tenant user returned" },
          "404": { description: "User not found in tenant" }
        }
      }
    },
    "/api/meetings": {
      get: {
        tags: ["Meetings"],
        summary: "List meetings in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        responses: {
          "200": {
            description: "Meeting list",
            content: {
              "application/json": {
                schema: {
                  type: "array",
                  items: { $ref: "#/components/schemas/Meeting" }
                }
              }
            }
          }
        }
      },
      post: {
        tags: ["Meetings"],
        summary: "Create meeting in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["title"],
                properties: {
                  title: { type: "string", example: "Sprint Review" },
                  transcript: { type: "string", example: "https://example.com/transcript.txt" },
                  summary: { type: "string", example: "Reviewed progress and blockers" }
                }
              }
            }
          }
        },
        responses: {
          "201": { description: "Meeting created" }
        }
      }
    },
    "/api/meetings/{id}": {
      get: {
        tags: ["Meetings"],
        summary: "Get meeting by ID in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "200": { description: "Meeting returned" },
          "404": { description: "Meeting not found" }
        }
      },
      patch: {
        tags: ["Meetings"],
        summary: "Update meeting in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "200": { description: "Meeting updated" }
        }
      },
      delete: {
        tags: ["Meetings"],
        summary: "Delete meeting in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "204": { description: "Meeting deleted" }
        }
      }
    },
    "/api/meetings/{id}/participants": {
      post: {
        tags: ["Meetings"],
        summary: "Add participant to meeting (tenant admin only)",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["user_id"],
                properties: {
                  user_id: { type: "integer", example: 2 }
                }
              }
            }
          }
        },
        responses: {
          "201": { description: "Participant added" },
          "403": { description: "Admin role required" }
        }
      }
    },
    "/api/meetings/{id}/participants/{userId}": {
      delete: {
        tags: ["Meetings"],
        summary: "Remove participant from meeting (tenant admin only)",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } },
          { name: "userId", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "204": { description: "Participant removed" }
        }
      }
    },
    "/api/tasks": {
      get: {
        tags: ["Tasks"],
        summary: "List tasks in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        responses: {
          "200": {
            description: "Task list",
            content: {
              "application/json": {
                schema: {
                  type: "array",
                  items: { $ref: "#/components/schemas/Task" }
                }
              }
            }
          }
        }
      },
      post: {
        tags: ["Tasks"],
        summary: "Create task in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["meeting_id", "task_text"],
                properties: {
                  meeting_id: { type: "integer", example: 1 },
                  task_text: { type: "string", example: "Follow up with stakeholders" },
                  due_date: { type: "string", format: "date-time" },
                  status: { type: "string", enum: ["TODO", "IN_PROGRESS", "DONE"] }
                }
              }
            }
          }
        },
        responses: {
          "201": { description: "Task created" }
        }
      }
    },
    "/api/tasks/{id}": {
      get: {
        tags: ["Tasks"],
        summary: "Get task by ID in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "200": { description: "Task returned" }
        }
      },
      patch: {
        tags: ["Tasks"],
        summary: "Update task in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "200": { description: "Task updated" }
        }
      },
      delete: {
        tags: ["Tasks"],
        summary: "Delete task in tenant company",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "204": { description: "Task deleted" }
        }
      }
    },
    "/api/tasks/{id}/assignees": {
      post: {
        tags: ["Tasks"],
        summary: "Add assignee to task (tenant admin only)",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["user_id"],
                properties: {
                  user_id: { type: "integer", example: 2 }
                }
              }
            }
          }
        },
        responses: {
          "201": { description: "Assignee added" },
          "403": { description: "Admin role required" }
        }
      }
    },
    "/api/tasks/{id}/assignees/{userId}": {
      delete: {
        tags: ["Tasks"],
        summary: "Remove assignee from task (tenant admin only)",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } },
          { name: "userId", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "204": { description: "Assignee removed" }
        }
      }
    }
  }
};
