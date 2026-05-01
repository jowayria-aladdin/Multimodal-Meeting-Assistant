export const openApiSpec = {
  openapi: "3.0.3",
  info: {
    title: "AI-NoteTaker Backend API",
    version: "1.1.1",
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
      },
      internalSecret: {
        type: "apiKey",
        in: "header",
        name: "X-Internal-Secret",
        description: "Required for FastAPI callback endpoints."
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
          name: { type: "string", example: "AI NoteTaker Inc" }
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
          email: { type: "string", format: "email", example: "admin@example.com" },
          role: { type: "string", example: "admin" }
        }
      },
      Meeting: {
        type: "object",
        properties: {
          id: { type: "integer", example: 1 },
          company_id: { type: "integer", example: 1 },
          title: { type: "string", example: "Weekly Sync" },
          main_video_url: {
            type: "string",
            format: "uri",
            nullable: true,
            example: "https://res.cloudinary.com/demo/video/upload/v1713000000/main_video.mp4"
          },
          main_video_public_id: {
            type: "string",
            nullable: true,
            example: "main_video_42"
          },
          transcript: { type: "string", nullable: true, example: "https://example.com/transcript.txt" },
          summary: { type: "string", nullable: true, example: "Discussed blockers and next steps." },
          processing_status: {
            type: "string",
            enum: ["UPLOADED", "QUEUED", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED"],
            example: "PROCESSING"
          },
          progress_percent: { type: "integer", example: 35 },
          status_message: { type: "string", nullable: true, example: "Transcribing audio" },
          processing_started_at: { type: "string", format: "date-time", nullable: true },
          processing_completed_at: { type: "string", format: "date-time", nullable: true },
          error_message: { type: "string", nullable: true },
          created_at: { type: "string", format: "date-time" }
        }
      },
      MeetingStatus: {
        type: "object",
        properties: {
          meetingId: { type: "integer", example: 1 },
          status: {
            type: "string",
            enum: ["UPLOADED", "QUEUED", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED"],
            example: "PROCESSING"
          },
          progress: { type: "integer", example: 60 },
          stage: { type: "string", nullable: true, example: "Extracting tasks" },
          error: { type: "string", nullable: true },
          processingStartedAt: { type: "string", format: "date-time", nullable: true },
          processingCompletedAt: { type: "string", format: "date-time", nullable: true }
        }
      },
      Task: {
        type: "object",
        properties: {
          id: { type: "integer", example: 1 },
          meeting_id: { type: "integer", example: 1 },
          task_text: { type: "string", example: "Prepare architecture draft" },
          due_date: { type: "string", format: "date-time", nullable: true },
          status: { type: "string", enum: ["TODO", "IN_PROGRESS", "DONE"], example: "TODO" },
          taskAssignees: {
            type: "array",
            items: {
              type: "object",
              properties: {
                task_id: { type: "integer", example: 1 },
                user_id: { type: "integer", example: 2 },
                user: {
                  type: "object",
                  properties: {
                    id: { type: "integer", example: 2 },
                    username: { type: "string", example: "mohamed" },
                    email: { type: "string", format: "email", example: "mohamed@example.com" }
                  }
                }
              }
            }
          }
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
                  name: { type: "string", example: "My Company" }
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
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  name: { type: "string", example: "Updated Company Name" }
                }
              }
            }
          }
        },
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
                required: ["email", "role"],
                properties: {
                  email: { type: "string", format: "email", example: "user@example.com" },
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
      patch: {
        tags: ["Companies"],
        summary: "Update company member role (admin only)",
        security: [{ bearerAuth: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } },
          { name: "userId", in: "path", required: true, schema: { type: "integer" } }
        ],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["role"],
                properties: {
                  role: {
                    type: "string",
                    enum: ["admin", "member"],
                    example: "admin"
                  }
                }
              }
            }
          }
        },
        responses: {
          "200": {
            description: "Membership role updated",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/CompanyMembership" }
              }
            }
          },
          "400": { description: "Invalid role" },
          "403": { description: "Admin role required" },
          "404": { description: "Membership not found" }
        }
      },
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
    "/api/meetings/upload": {
      post: {
        tags: ["Meetings"],
        summary: "Create meeting from sign-video + single wav file and Cloudinary main video URL",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        requestBody: {
          required: true,
          content: {
            "multipart/form-data": {
              schema: {
                type: "object",
                required: ["title", "lang", "mainVideoUrl", "signVideo", "wavFile"],
                properties: {
                  title: { type: "string", example: "Sprint Review Audio" },
                  lang: {
                    type: "string",
                    enum: ["en", "ar", "cs"],
                    description: "Required language code passed to FastAPI"
                  },
                  mainVideoUrl: {
                    type: "string",
                    format: "uri",
                    description: "Required Cloudinary playback URL for the main meeting video"
                  },
                  mainVideoPublicId: {
                    type: "string",
                    description: "Optional Cloudinary public id for lifecycle operations"
                  },
                  signVideo: {
                    type: "string",
                    format: "binary",
                    description: "Required sign-language video file (webm) sent to FastAPI"
                  },
                  wavFile: {
                    type: "string",
                    format: "binary",
                    description: "Required wav file sent to FastAPI"
                  }
                }
              }
            }
          }
        },
        responses: {
          "201": {
            description: "Meeting created and queued",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/Meeting" }
              }
            }
          }
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
          "200": {
            description: "Meeting returned",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/Meeting" },
                example: {
                  id: 42,
                  company_id: 3,
                  title: "Sprint Planning - Week 12",
                  main_video_url: "https://res.cloudinary.com/demo/video/upload/v1713000000/main_video.mp4",
                  main_video_public_id: "main_video_42",
                  transcript: "SPEAKER_00 [00:00:01.120 - 00:00:06.840]: Good morning everyone, let's start with blockers.\nSPEAKER_01 [00:00:07.050 - 00:00:14.200]: Backend API for meeting upload is ready, I still need to finalize callback validation.\nSPEAKER_00 [00:00:14.410 - 00:00:20.030]: Great, please create tasks for docs and frontend integration.",
                  summary: "Team reviewed blockers, confirmed backend upload pipeline readiness, and assigned follow-up actions for docs and frontend integration.",
                  processing_status: "COMPLETED",
                  progress_percent: 100,
                  status_message: "Processing completed",
                  processing_started_at: "2026-04-16T18:45:11.000Z",
                  processing_completed_at: "2026-04-16T18:47:03.000Z",
                  error_message: null,
                  created_at: "2026-04-16T18:44:58.000Z",
                  meetingParticipants: [
                    {
                      meeting_id: 42,
                      user_id: 7
                    },
                    {
                      meeting_id: 42,
                      user_id: 11
                    }
                  ],
                  tasks: [
                    {
                      id: 101,
                      meeting_id: 42,
                      task_text: "Publish backend API integration notes for frontend team",
                      due_date: "2026-04-18T12:00:00.000Z",
                      status: "IN_PROGRESS"
                    },
                    {
                      id: 102,
                      meeting_id: 42,
                      task_text: "Implement SSE progress updates with polling fallback",
                      due_date: null,
                      status: "TODO"
                    }
                  ]
                }
              }
            }
          },
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
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  title: { type: "string", example: "Updated meeting title" },
                  transcript: { type: "string", nullable: true, example: "Updated transcript text" },
                  summary: { type: "string", nullable: true, example: "Updated meeting summary" }
                }
              }
            }
          }
        },
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
    "/api/meetings/{id}/status": {
      get: {
        tags: ["Meetings"],
        summary: "Get processing status for a meeting",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "200": {
            description: "Meeting status returned",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/MeetingStatus" }
              }
            }
          }
        }
      }
    },
    "/api/meetings/{id}/events": {
      get: {
        tags: ["Meetings"],
        summary: "Stream meeting processing events (SSE)",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "200": { description: "SSE stream opened" }
        }
      }
    },
    "/api/meetings/{id}/reprocess": {
      post: {
        tags: ["Meetings"],
        summary: "Requeue processing for a meeting",
        security: [{ bearerAuth: [], tenantHeader: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        responses: {
          "202": { description: "Reprocess accepted" }
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
                required: ["email"],
                properties: {
                  email: { type: "string", format: "email", example: "user@example.com" }
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
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                properties: {
                  meeting_id: { type: "integer", example: 1 },
                  task_text: { type: "string", example: "Updated task text" },
                  due_date: { type: "string", format: "date-time", nullable: true },
                  status: { type: "string", enum: ["TODO", "IN_PROGRESS", "DONE"], example: "IN_PROGRESS" }
                }
              }
            }
          }
        },
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
                required: ["email"],
                properties: {
                  email: { type: "string", format: "email", example: "user@example.com" }
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
    },
    "/api/internal/meetings/{id}/callback": {
      post: {
        tags: ["Meetings"],
        summary: "Internal callback from FastAPI for meeting processing",
        security: [{ internalSecret: [] }],
        parameters: [
          { name: "id", in: "path", required: true, schema: { type: "integer" } }
        ],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: {
                type: "object",
                required: ["status"],
                properties: {
                  status: { type: "string", example: "PROCESSING" },
                  progress: { type: "integer", example: 50 },
                  stage: { type: "string", example: "Summarizing" },
                  message: { type: "string", example: "Running summarization model" },
                  error: { type: "string", nullable: true },
                  result: {
                    type: "object",
                    properties: {
                      transcript: { type: "string" },
                      summary: { type: "string" },
                      tasks: {
                        type: "array",
                        items: {
                          type: "object",
                          properties: {
                            task_text: { type: "string" },
                            title: { type: "string" },
                            due_date: { type: "string", format: "date-time", nullable: true },
                            status: { type: "string", enum: ["TODO", "IN_PROGRESS", "DONE"] }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        },
        responses: {
          "200": { description: "Callback processed" },
          "401": { description: "Invalid internal callback secret" }
        }
      }
    }
  }
};
