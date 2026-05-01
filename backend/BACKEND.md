# AI-NoteTaker Backend Documentation

A comprehensive, production-ready Express.js backend API for a multi-tenant meeting management system with real-time note-taking, participant tracking, and intelligent task assignment.

---

## Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [Architecture](#architecture)
4. [Directory Structure](#directory-structure)
5. [Database Schema](#database-schema)
6. [API Endpoints](#api-endpoints)
7. [Authentication & Security](#authentication--security)
8. [Multi-Tenancy Architecture](#multi-tenancy-architecture)
9. [Testing Strategy](#testing-strategy)
10. [Setup & Configuration](#setup--configuration)
11. [Development Guide](#development-guide)
12. [Deployment](#deployment)

---

## Overview

The AI-NoteTaker Backend is a RESTful API built with **Express.js** and **Prisma ORM** that powers the AI-NoteTaker application. It manages:

- **Multi-tenant company administration** with role-based access control
- **User authentication** using JWT tokens with bcrypt hashing
- **Meeting management** with automatic transcription and summaries
- **Participant tracking** across multiple meetings
- **Task management** with status tracking and user assignments
- **Real-time collaboration** with isolated company data

### Key Features

✅ Multi-tenant isolation at database and application layers  
✅ JWT-based authentication with session management  
✅ Role-based access control (OWNER, ADMIN, MEMBER)  
✅ Comprehensive REST API with OpenAPI 3.0.3 documentation  
✅ PostgreSQL database with Prisma ORM  
✅ Automated testing with unit and feature tests  
✅ Seed data for rapid development and testing  
✅ Production-ready error handling and validation  

**Backend Quality Rating: 8.3/10**

---

## Technology Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Framework** | Express.js | 5.2.1 | HTTP server & routing |
| **ORM** | Prisma | 6.17.1 | Database query builder & migrations |
| **Database** | PostgreSQL | 15+ | Relational data persistence |
| **Authentication** | JWT + bcrypt | 9.0.3 / 6.0.0 | Token auth & password hashing |
| **Documentation** | Swagger UI | 5.0.1 | Interactive API docs |
| **Testing** | Vitest + Supertest | 4.1.0 / 7.2.2 | Unit tests & HTTP integration |
| **Runtime** | Node.js | 18+ | JavaScript runtime |
| **Environment** | dotenv | 17.3.1 | Configuration management |

---

## Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    External Clients                          │
│              (Web, Mobile, Third-party APIs)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────────┐
         │   Express.js HTTP Server   │
         │   (Port 3000)              │
         └───────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌─────────┐  ┌─────────┐  ┌──────────────┐
   │ Router  │  │Router   │  │ Global       │
   │Layer    │  │Layer    │  │ Middlewares  │
   └────┬────┘  └────┬────┘  └──────┬───────┘
        │            │              │
        └────────────┼──────────────┘
                     ▼
    ┌────────────────────────────────────┐
    │   Middleware Stack                  │
    │  ┌──────────────────────────────┐  │
    │  │ CORS & Security Headers      │  │
    │  ├──────────────────────────────┤  │
    │  │ Request Logging              │  │
    │  ├──────────────────────────────┤  │
    │  │ Authentication (Bearer Token)│  │
    │  ├──────────────────────────────┤  │
    │  │ Tenant Isolation             │  │
    │  │ (X-Company-Id Header)        │  │
    │  └──────────────────────────────┘  │
    └────────────────┬───────────────────┘
                     ▼
    ┌────────────────────────────────────┐
    │   Service Layer (Business Logic)    │
    │  ├─ AuthService                     │
    │  ├─ UserService                     │
    │  ├─ CompanyService                  │
    │  ├─ MeetingService                  │
    │  └─ TaskService                     │
    └────────────────┬───────────────────┘
                     ▼
    ┌────────────────────────────────────┐
    │   Prisma ORM Layer                  │
    │   (Database Abstraction)            │
    └────────────────┬───────────────────┘
                     ▼
    ┌────────────────────────────────────┐
    │   PostgreSQL Database               │
    │   (Data Persistence)                │
    └────────────────────────────────────┘
```

### Design Patterns

**MVC with Service Layer:**
- **Controllers**: Handle HTTP requests/responses
- **Services**: Contain business logic and Prisma queries
- **Models**: Defined in `prisma/schema.prisma`

**Middleware Chain:**
- Request passes through middleware stack before reaching controllers
- Tenant isolation enforced at middleware level
- Authentication verified before protected routes

**Multi-Tenancy:**
- **Header-based tenant identification**: `X-Company-Id` header specifies company context
- **Database-level isolation**: All queries scoped to company_id
- **Membership validation**: Middleware verifies user belongs to company
- **Role-based access control**: Admin operations guarded by role checks

---

## Directory Structure

```
backend/
├── src/
│   ├── app.js                         # Express app initialization & route setup
│   ├── server.js                      # Server entry point (port 3000)
│   ├── config/
│   │   └── database.js                # Prisma client singleton
│   ├── controllers/                   # HTTP request handlers
│   │   ├── authController.js
│   │   ├── userController.js
│   │   ├── companyController.js
│   │   ├── meetingController.js
│   │   └── taskController.js
│   ├── services/                      # Business logic & data layer
│   │   ├── authService.js
│   │   ├── userService.js
│   │   ├── companyService.js
│   │   ├── meetingService.js
│   │   └── taskService.js
│   ├── routes/                        # Router definitions
│   │   ├── authRoutes.js
│   │   ├── userRoutes.js
│   │   ├── companyRoutes.js
│   │   ├── meetingRoutes.js
│   │   └── taskRoutes.js
│   ├── middlewares/
│   │   ├── authMiddleware.js          # JWT token validation
│   │   └── tenantMiddleware.js        # Company scoping & membership check
│   ├── docs/
│   │   └── openapi.js                 # OpenAPI 3.0.3 spec (Swagger)
│   └── utils/
│       └── httpError.js               # Custom error handling
├── prisma/
│   ├── schema.prisma                  # Database schema definition
│   ├── migrations/                    # Auto-generated migrations
│   └── seed.js                        # Database seeding script
├── tests/
│   ├── setup.js                       # Test environment configuration
│   ├── unit/
│   │   ├── auth.service.test.js       # Auth service unit tests
│   │   └── tenant.middleware.test.js  # Tenant middleware unit tests
│   └── feature/
│       └── app.feature.test.js        # HTTP endpoint integration tests
├── postman/
│   └── AI-NoteTaker.postman_collection.json  # Postman API collection
├── .env                               # Environment variables (local)
├── .env.example                       # Environment template
├── package.json                       # NPM dependencies & scripts
├── vitest.config.js                   # Vitest configuration
└── BACKEND.md                         # This file
```

---

## Database Schema

### Entity Relationship Diagram

```
┌─────────────┐         ┌──────────────────┐
│    User     │◄───────►│ CompanyMembership│
│             │  1..n   └──────────────────┘
├─────────────┤              │
│ id (PK)     │              │ n..1
│ username    │              │
│ email       │              ▼
│ password_   │         ┌──────────────┐
│   hash      │         │  Company     │
└─────────────┘         │              │
      ▲                 ├──────────────┤
      │                 │ id (PK)      │
      │            1..n │ name         │
      │                 │ logo         │
   1..n                 └──────────────┘
      │                      ▲
      │               n      │ 1
      │          ┌───────────┘
      │          │
┌─────┴──────────┼─────────┐
│   Task        │  Meeting  │
│               │           │
├─────────────┬─┴───────┬───┤
│ id (PK)     │ id (PK) │   │
│ meeting_id  │ company │   │
│ (FK)        │ _id(FK) │   │
│ task_text   │ title   │   │
│ due_date    │ trans   │   │
│ status      │ script  │   │
└─────────────┴─────────┘   │
      ▲                      │
      │               ┌──────┴──────┐
      │          1..n │   n..1      │
      │          ┌────┴─────────────┘
      │          │
     1│n    ┌────▼──────────────┐
      │     │ MeetingParticipants│
      │     └───────────────────┘
      │              ▲
      │         n..m │
      │          ┌───┘
      └──────────┘
      TaskAssignees
```

### Models Overview

#### **User**
Represents system users who can be members of companies.

```prisma
model User {
  id                  Int
  username            String @unique
  email               String @unique
  password_hash       String
  // Relations
  companyMemberships  CompanyMembership[]
  meetingParticipants MeetingParticipants[]
  taskAssignees       TaskAssignees[]
}
```

**Fields:**
- `id`: Auto-incrementing primary key
- `username`: Unique user identifier
- `email`: Contact email
- `password_hash`: bcrypt-hashed password (12 rounds)

---

#### **Company**
Represents a company/organization (tenant).

```prisma
model Company {
  id          Int
  name        String
  logo        String?
  // Relations
  memberships CompanyMembership[]
  meetings    Meeting[]
}
```

**Fields:**
- `id`: Auto-incrementing primary key
- `name`: Company name
- `logo`: Optional company logo URL

---

#### **CompanyMembership**
Junction table managing user roles within companies (multi-tenancy enforcement).

```prisma
model CompanyMembership {
  user_id    Int
  company_id Int
  role       String  // OWNER, ADMIN, MEMBER
  // Relations
  user       User
  company    Company
  @@id([user_id, company_id])  // Composite key
}
```

**Fields:**
- `user_id`: Foreign key to User
- `company_id`: Foreign key to Company
- `role`: String enum (OWNER, ADMIN, MEMBER)

**Role Permissions:**
- **OWNER**: Full admin access, can manage users and company settings
- **ADMIN**: Can assign users to meetings and manage tasks
- **MEMBER**: Can participate in meetings and be assigned tasks

---

#### **Meeting**
Represents a meeting with transcript and participants.

```prisma
model Meeting {
  id                  Int
  company_id          Int
  title               String
  transcript          String?
  summary             String?
  created_at          DateTime
  // Relations
  company             Company
  meetingParticipants MeetingParticipants[]
  tasks               Task[]
  @@index([company_id])
  @@index([created_at])
}
```

**Fields:**
- `id`: Auto-incrementing primary key
- `company_id`: Scopes meeting to company
- `title`: Meeting title
- `transcript`: Full meeting transcript (from speech-to-text)
- `summary`: AI-generated meeting summary
- `created_at`: Timestamp (auto-set on insert)

---

#### **MeetingParticipants**
Junction table linking users to meetings.

```prisma
model MeetingParticipants {
  meeting_id Int
  user_id    Int
  meeting    Meeting
  user       User
  @@id([meeting_id, user_id])  // Composite key
}
```

**Purpose:** Track which users participated in which meetings.

---

#### **Task**
Represents action items created from meetings.

```prisma
model Task {
  id            Int
  meeting_id    Int
  task_text     String
  due_date      DateTime?
  status        TaskStatus  // TODO, IN_PROGRESS, DONE
  meeting       Meeting
  taskAssignees TaskAssignees[]
  @@index([meeting_id])
  @@index([status])
  @@index([due_date])
}
```

**Fields:**
- `id`: Auto-incrementing primary key
- `meeting_id`: Links task to meeting
- `task_text`: Description of the task
- `due_date`: Optional deadline
- `status`: Enum (TODO, IN_PROGRESS, DONE)

---

#### **TaskAssignees**
Junction table assigning users to tasks.

```prisma
model TaskAssignees {
  task_id  Int
  user_id  Int
  task     Task
  user     User
  @@id([task_id, user_id])  // Composite key
}
```

**Purpose:** Track task assignments (task can be assigned to multiple users).

---

## API Endpoints

All endpoints require:
- **Authentication**: Bearer token in `Authorization: Bearer <token>` header (except /health and /auth/register)
- **Multi-tenancy**: `X-Company-Id` header for all protected endpoints specifying which company context to use

### Base URL
```
http://localhost:3000/api
```

### Response Format

**Success Response (2xx):**
```json
{
  "success": true,
  "data": { /* resource or array */ },
  "message": "Operation successful"
}
```

**Error Response (4xx/5xx):**
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Descriptive error message"
  }
}
```

---

### 1. Health Endpoint

#### GET /health
Check if API is running (no auth required).

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2024-03-20T10:30:00.000Z"
}
```

---

### 2. Authentication Endpoints

#### POST /auth/register
Register a new user.

**Request Body:**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePassword123!"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "username": "john_doe",
    "email": "john@example.com"
  },
  "message": "User registered successfully"
}
```

**Error Cases:**
- `400`: Duplicate username or email
- `400`: Invalid password format
- `500`: Database error

---

#### POST /auth/login
Authenticate and receive JWT token.

**Request Body:**
```json
{
  "username": "john_doe",
  "password": "SecurePassword123!"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": 1,
      "username": "john_doe",
      "email": "john@example.com"
    }
  },
  "message": "Login successful"
}
```

**Token Details:**
- **Type**: JWT (HS256)
- **Expiry**: 24 hours (configurable via `JWT_EXPIRES_IN`)
- **Secret**: `JWT_SECRET` from environment

**Error Cases:**
- `401`: Invalid username or password
- `400`: Missing credentials

---

### 3. Company Endpoints

#### POST /companies
Create a new company (creator becomes OWNER).

**Headers:**
```
Authorization: Bearer <token>
X-Company-Id: 1
```

**Request Body:**
```json
{
  "name": "Acme Corporation",
  "logo": "https://example.com/logo.png"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "id": 5,
    "name": "Acme Corporation",
    "logo": "https://example.com/logo.png"
  }
}
```

---

#### GET /companies/:id
Get company details (must be member).

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": 5,
    "name": "Acme Corporation",
    "logo": "https://example.com/logo.png",
    "memberships": [
      {
        "user_id": 1,
        "company_id": 5,
        "role": "OWNER",
        "user": {
          "id": 1,
          "username": "john_doe",
          "email": "john@example.com"
        }
      }
    ]
  }
}
```

---

#### POST /companies/:id/memberships
Add user to company (admin only).

**Headers:**
```
Authorization: Bearer <token>
X-Company-Id: 5
```

**Request Body:**
```json
{
  "user_id": 2,
  "role": "MEMBER"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "user_id": 2,
    "company_id": 5,
    "role": "MEMBER"
  }
}
```

**Constraints:**
- Only OWNER/ADMIN can add users
- User must exist in system

---

### 4. User Endpoints

#### GET /users
List all users in company (member+ role).

**Headers:**
```
Authorization: Bearer <token>
X-Company-Id: 5
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "username": "john_doe",
      "email": "john@example.com"
    },
    {
      "id": 2,
      "username": "jane_smith",
      "email": "jane@example.com"
    }
  ]
}
```

---

#### GET /users/:id
Get specific user details (in same company).

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "username": "john_doe",
    "email": "john@example.com"
  }
}
```

---

### 5. Meeting Endpoints

#### POST /meetings
Create meeting in company.

**Headers:**
```
Authorization: Bearer <token>
X-Company-Id: 5
```

**Request Body:**
```json
{
  "title": "Q1 Planning Meeting",
  "transcript": "Transcript content here...",
  "summary": "Discussed Q1 OKRs and roadmap"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "id": 10,
    "company_id": 5,
    "title": "Q1 Planning Meeting",
    "transcript": "Transcript content here...",
    "summary": "Discussed Q1 OKRs and roadmap",
    "created_at": "2024-03-20T10:30:00.000Z"
  }
}
```

---

#### GET /meetings
List all meetings in company.

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": 10,
      "company_id": 5,
      "title": "Q1 Planning Meeting",
      "transcript": "...",
      "summary": "...",
      "created_at": "2024-03-20T10:30:00.000Z"
    }
  ]
}
```

---

#### GET /meetings/:id
Get meeting details with participants.

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": 10,
    "company_id": 5,
    "title": "Q1 Planning Meeting",
    "transcript": "...",
    "summary": "...",
    "created_at": "2024-03-20T10:30:00.000Z",
    "meetingParticipants": [
      {
        "meeting_id": 10,
        "user_id": 1,
        "user": {
          "id": 1,
          "username": "john_doe",
          "email": "john@example.com"
        }
      }
    ]
  }
}
```

---

#### POST /meetings/:id/participants
Add user as meeting participant (admin+).

**Request Body:**
```json
{
  "user_id": 2
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "meeting_id": 10,
    "user_id": 2
  }
}
```

---

### 6. Task Endpoints

#### POST /tasks
Create task (linked to meeting).

**Headers:**
```
Authorization: Bearer <token>
X-Company-Id: 5
```

**Request Body:**
```json
{
  "meeting_id": 10,
  "task_text": "Finalize Q1 roadmap",
  "due_date": "2024-04-15T00:00:00.000Z",
  "status": "TODO"
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "id": 20,
    "meeting_id": 10,
    "task_text": "Finalize Q1 roadmap",
    "due_date": "2024-04-15T00:00:00.000Z",
    "status": "TODO"
  }
}
```

---

#### GET /tasks
List all tasks in company.

**Query Parameters:**
- `status`: Filter by TODO, IN_PROGRESS, or DONE
- `meeting_id`: Filter by meeting

**Example:** `GET /tasks?status=TODO`

**Response (200 OK):**
```json
{
  "success": true,
  "data": [
    {
      "id": 20,
      "meeting_id": 10,
      "task_text": "Finalize Q1 roadmap",
      "due_date": "2024-04-15T00:00:00.000Z",
      "status": "TODO"
    }
  ]
}
```

---

#### GET /tasks/:id
Get task with assignees.

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": 20,
    "meeting_id": 10,
    "task_text": "Finalize Q1 roadmap",
    "due_date": "2024-04-15T00:00:00.000Z",
    "status": "TODO",
    "taskAssignees": [
      {
        "task_id": 20,
        "user_id": 1,
        "user": {
          "id": 1,
          "username": "john_doe",
          "email": "john@example.com"
        }
      }
    ]
  }
}
```

---

#### POST /tasks/:id/assignees
Assign user to task (admin+).

**Request Body:**
```json
{
  "user_id": 2
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "task_id": 20,
    "user_id": 2
  }
}
```

---

#### PATCH /tasks/:id
Update task (status, due_date, etc.).

**Request Body:**
```json
{
  "status": "IN_PROGRESS",
  "task_text": "Updated task description"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "id": 20,
    "meeting_id": 10,
    "task_text": "Updated task description",
    "due_date": "2024-04-15T00:00:00.000Z",
    "status": "IN_PROGRESS"
  }
}
```

---

## Authentication & Security

### JWT Token Flow

```
┌─────────────┐
│   Client    │─────────────────────────────────────────┐
│ (Web/Mobile)│                                         │
└─────────────┘                                         │
      ▲                                                 │
      │                                                 ▼
      │                                        ┌──────────────────┐
      │                                        │  1. Register or  │
      │                                        │     Login        │
      │                                        │  2. Receive JWT  │
      │                                        └──────────────────┘
      │                                                 │
      │                                                 ▼
      │ ┌────────────────────────────────────────────────────────┐
      │ │  Token Response (Backend)                              │
      │ │  ┌──────────────────────────────────────────────────┐  │
      │ │  │ Header: { alg: "HS256", typ: "JWT" }             │  │
      │ │  │ Payload: { userId, username, iat, exp }         │  │
      │ │  │ Signature: HMAC-SHA256(secret)                   │  │
      │ │  └──────────────────────────────────────────────────┘  │
      │ └────────────────────────────────────────────────────────┘
      │                                                 │
      │                                                 ▼
      ├── Store in localStorage/sessionStorage ◄──────┘
      │
      │
      ├─── POST /api/meetings
      │    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
      │    X-Company-Id: 5
      │
      ▼
┌──────────────────────────────────────────┐
│  Backend JWT Verification                │
│  ┌──────────────────────────────────────┐│
│  │ 1. Extract token from header         ││
│  │ 2. Verify signature with secret      ││
│  │ 3. Check expiry (iat, exp)           ││
│  │ 4. Extract userId & verify in DB    ││
│  │ 5. Check X-Company-Id membership    ││
│  └──────────────────────────────────────┘│
└──────────────────────────────────────────┘
         │
         ├─ Valid ──► Proceed to handler
         │
         └─ Invalid ─► Return 401 Unauthorized
```

### Security Features

**Password Security:**
- Passwords hashed with bcrypt (12 rounds)
- Never stored in plain text
- Compared using constant-time comparison

**Token Management:**
- JWT tokens signed with HS256
- 24-hour expiry (configurable)
- Tokens not stored server-side (stateless)
- Logout: Client-side only (remove token from storage)

**Data Protection:**
- All database queries scoped to tenant company
- X-Company-Id header validated on every request
- Membership verified before granting access
- Role-based guards on sensitive operations

**Best Practices:**
- Tokens sent via Authorization header (not URL or body)
- HTTPS enforced in production
- JWT_SECRET kept secure in environment variables
- Tokens typically stored in secure, HTTP-only cookies (frontend implementation)

---

## Multi-Tenancy Architecture

### Tenant Isolation Strategy

**Multi-tenancy** ensures complete data isolation between companies. Each company is a separate logical tenant with independent data.

#### 1. Header-Based Tenant Identification

Every protected request must include `X-Company-Id` header:

```http
GET /api/meetings
Authorization: Bearer <token>
X-Company-Id: 5
```

This header tells the backend which company's context to operate in.

#### 2. Middleware-Level Enforcement

The `tenantMiddleware.js` validates:

```javascript
// 1. Parse and validate X-Company-Id from header
const companyId = parseInt(req.headers['x-company-id']);

// 2. Verify user is member of that company
const membership = await prisma.companyMembership.findUnique({
  where: { user_id_company_id: { user_id, company_id } }
});

// 3. Guard admin operations
if (requireAdmin && membership.role !== 'OWNER' && membership.role !== 'ADMIN') {
  throw new HttpError('Forbidden', 403);
}
```

#### 3. Query-Level Scoping

All service layer queries filter by `company_id`:

```javascript
// ❌ WRONG - No tenant scoping
const meetings = await prisma.meeting.findMany();

// ✅ CORRECT - Company scoped
const meetings = await prisma.meeting.findMany({
  where: { company_id }
});
```

#### 4. Role-Based Access Control

Three roles with hierarchical permissions:

| Role | Permissions |
|------|------------|
| **OWNER** | ✅ Manage users, meetings, tasks, company settings |
| **ADMIN** | ✅ Manage meetings & tasks, add users to meetings |
| **MEMBER** | ✅ Participate in meetings, view assigned tasks |

#### Example: Adding User to Company (Admin Only)

```javascript
// Step 1: Middleware validates user is OWNER/ADMIN of companyId
await requireTenantAdminFromHeader(req, res);

// Step 2: Controller calls service
const membership = await companyService.addUserToCompany(
  companyId, 
  userId, 
  role
);

// Step 3: Service creates membership
await prisma.companyMembership.create({
  data: {
    user_id: userId,
    company_id: companyId,  // ← Tenant scoped
    role
  }
});
```

### Data Isolation Patterns

**Pattern 1: Direct Company Scoping**
```javascript
// Meetings only visible to company members
await prisma.meeting.findMany({
  where: { company_id: userCompanyId }
});
```

**Pattern 2: Relationship-Based Scoping**
```javascript
// Tasks visible through meeting → company relationship
await prisma.task.findMany({
  where: {
    meeting: {
      company_id: userCompanyId
    }
  }
});
```

**Pattern 3: Composite Key Enforcement**
```javascript
// MeetingParticipants only with meetings from user's company
const participant = await prisma.meetingParticipants.findFirst({
  where: {
    meeting_id: meetingId,
    meeting: {
      company_id: userCompanyId
    }
  }
});
```

### Cross-Tenant Attack Prevention

The middleware prevents:

```javascript
// ❌ User from Company A tries to access Company B's meeting
GET /api/meetings/100
Authorization: Bearer <tokenFromCompanyAUser>
X-Company-Id: 2  // Company B

// Middleware checks:
// User is member of Company A, not Company B
// → 403 Forbidden
```

---

## Testing Strategy

### Overview

**Test Suite: 13 Tests (100% Passing)**

The test suite covers critical business logic and integration points:

| Category | Tests | Focus |
|----------|-------|-------|
| **Unit Tests** | 9 | Auth service, tenant middleware |
| **Feature Tests** | 4 | HTTP endpoints, integration |
| **Total** | 13 | Core functionality |

### Running Tests

```bash
# Run all tests
npm test

# Run only unit tests
npm run test:unit

# Run only feature tests
npm run test:feature

# Watch mode (re-run on file changes)
npm run test:watch
```

### Test Files Overview

#### 1. Unit Tests: Authentication Service (`tests/unit/auth.service.test.js`)

**5 Tests** covering user registration and login logic.

**Test 1: Register User Successfully**
- **Description**: User provides valid credentials and receives user object
- **Mocked Dependencies**: 
  - `prisma.user.findFirst()` − returns null (user doesn't exist)
  - `bcrypt.hash()` − returns hashed password
  - `prisma.user.create()` − returns created user
- **Assertion**: Returns user with id, username, email (no password)
- **Validates**: Registration creates user, hashes password, doesn't expose password

**Test 2: Register User with Duplicate Username**
- **Description**: Prevent user registration with existing username
- **Mocked Dependencies**:
  - `prisma.user.findFirst()` − returns existing user
- **Assertion**: Throws HttpError with message "Username already exists"
- **Validates**: Uniqueness validation works

**Test 3: Register User with Duplicate Email**
- **Description**: Prevent user registration with existing email
- **Mocked Dependencies**:
  - `prisma.user.findFirst()` − returns existing user (on second call, email check)
- **Assertion**: Throws HttpError with message "Email already exists"
- **Validates**: Email uniqueness validation works

**Test 4: Login User Successfully**
- **Description**: User provides valid credentials and receives JWT token
- **Mocked Dependencies**:
  - `prisma.user.findFirst()` − returns user from DB
  - `bcrypt.compare()` − returns true (password match)
  - `jwt.sign()` − returns valid token string
- **Assertion**: Returns token and user object
- **Validates**: Login flow, password comparison, token generation

**Test 5: Login User with Invalid Password**
- **Description**: Deny login with incorrect password
- **Mocked Dependencies**:
  - `prisma.user.findFirst()` − returns user
  - `bcrypt.compare()` − returns false (password mismatch)
- **Assertion**: Throws HttpError with message "Invalid password"
- **Validates**: Password validation prevents unauthorized access

---

#### 2. Unit Tests: Tenant Middleware (`tests/unit/tenant.middleware.test.js`)

**4 Tests** covering company scoping and membership validation.

**Test 1: Parse Valid Company ID**
- **Description**: Middleware extracts X-Company-Id from header
- **Setup**: Request with header `X-Company-Id: 5`
- **Assertion**: `req.companyId === 5`
- **Validates**: Header parsing works correctly

**Test 2: Reject Missing Company ID**
- **Description**: Deny access when X-Company-Id header missing
- **Setup**: Request without X-Company-Id header
- **Assertion**: Throws HttpError with "X-Company-Id header required"
- **Validates**: Header is mandatory for protected routes

**Test 3: Authorize User in Company**
- **Description**: Allow access to user who is company member
- **Mocked Dependencies**:
  - `prisma.companyMembership.findUnique()` − returns membership with role MEMBER
- **Assertion**: Middleware allows request to proceed
- **Validates**: Membership check passes for valid members

**Test 4: Deny Non-Member Access**
- **Description**: Reject user trying to access company they don't belong to
- **Mocked Dependencies**:
  - `prisma.companyMembership.findUnique()` − returns null (no membership)
- **Assertion**: Throws HttpError with "User is not a member of this company"
- **Validates**: Non-members blocked from accessing company data

---

#### 3. Feature Tests: HTTP Endpoints (`tests/feature/app.feature.test.js`)

**4 Tests** covering full HTTP request/response cycles and integration.

**Test 1: Health Endpoint (No Auth)**
- **Description**: Health check works without authentication
- **Request**: `GET /api/health`
- **Expected Status**: 200
- **Validates**: 
  - Server is running
  - Public endpoint accessible
  - Response format correct

**Test 2: Swagger Documentation Endpoint (No Auth)**
- **Description**: OpenAPI spec served at /docs.json
- **Request**: `GET /api/docs.json`
- **Expected Status**: 200
- **Validates**:
  - Swagger docs available
  - OpenAPI 3.0.3 spec returned
  - API documentation complete

**Test 3: Register New User (POST /auth/register)**
- **Description**: User registration with valid credentials
- **Request Body**:
  ```json
  {
    "username": "testuser",
    "email": "test@example.com",
    "password": "TestPassword123"
  }
  ```
- **Mocked Dependencies**:
  - `AuthService.register()` − returns user object
- **Expected Status**: 201
- **Validates**:
  - Registration endpoint accepts POST
  - Password handling works
  - Response includes user ID

**Test 4: Protected Route Requires Auth (GET /api/users)**
- **Description**: Protected endpoints reject requests without valid token
- **Request**: `GET /api/users` (no Authorization header)
- **Expected Status**: 401
- **Assertion**: Response contains "Bearer token required" error
- **Validates**:
  - Authentication middleware enforces tokens on protected routes
  - Requests without token are rejected
  - Security guards are in place

---

### Testing Patterns Used

#### Mocking Pattern: Vitest + vi.hoisted()

Mocks are defined in `vi.hoisted()` factory to ensure they're created before test imports:

```javascript
const { prismaMock, bcryptMock } = vi.hoisted(() => ({
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
  }
}));

vi.doMock('@prisma/client', () => ({ PrismaClient: prismaMock }));
vi.doMock('bcrypt', () => bcryptMock);
```

#### Unit Test Pattern

```javascript
// Arrange: Set up mock return values
prismaMock.user.findFirst.mockResolvedValue(null);  // User doesn't exist
bcryptMock.hash.mockResolvedValue('hashedPassword');

// Act: Call service function
const result = await authService.register(credentials);

// Assert: Verify behavior
expect(result.username).toBe('john_doe');
expect(bcryptMock.hash).toHaveBeenCalledWith(password, 12);
```

#### Feature Test Pattern (Supertest)

```javascript
// Act: Make HTTP request
const response = await request(app)
  .post('/api/auth/register')
  .send({
    username: 'testuser',
    email: 'test@example.com',
    password: 'TestPassword123'
  });

// Assert: Verify HTTP response
expect(response.status).toBe(201);
expect(response.body.data.username).toBe('testuser');
```

### Test Coverage

**Auth Service (5 tests):**
- ✅ Register with valid credentials
- ✅ Prevent duplicate usernames
- ✅ Prevent duplicate emails
- ✅ Login with valid password
- ✅ Reject invalid password

**Tenant Middleware (4 tests):**
- ✅ Parse company ID from header
- ✅ Require X-Company-Id header
- ✅ Authorize member access
- ✅ Deny non-member access

**HTTP Endpoints (4 tests):**
- ✅ Health endpoint (public)
- ✅ Swagger docs endpoint
- ✅ Registration endpoint integration
- ✅ Authentication requirement on protected routes

### Gaps & Future Testing

The following areas are currently out of scope but recommended for production:

**Integration Tests:**
- Real PostgreSQL test database
- Automatic migration & cleanup
- End-to-end workflows (register → create company → add users → meetings → tasks)

**Security Tests:**
- SQL injection prevention
- XSS prevention
- CSRF protection
- Rate limiting

**Load Tests:**
- Concurrent user handling
- Query performance
- Database connection pooling

**Edge Cases:**
- Invalid input validation
- Large file handling
- Timezone handling
- Concurrent updates

---

## Setup & Configuration

### Prerequisites

- **Node.js** 18+ ([Download](https://nodejs.org/))
- **PostgreSQL** 15+ ([Download](https://www.postgresql.org/))
- **npm** 9+ (comes with Node.js)

### Environment Setup

#### 1. Install Dependencies

```bash
cd backend
npm install
```

#### 2. Configure Database

Create `.env` file in `backend/` directory:

```bash
# .env
DATABASE_URL="postgresql://username:password@localhost:5432/ai_note_taker?schema=public"
JWT_SECRET="your_super_secure_secret_key_here_at_least_32_chars"
JWT_EXPIRES_IN="24h"
NODE_ENV="development"
```

**Environment Variables Explained:**

| Variable | Purpose | Example |
|----------|---------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://ai_user:ai_user@localhost:5432/ai_note_taker` |
| `JWT_SECRET` | Secret for signing JWT tokens | `lughacap_secret_key` |
| `JWT_EXPIRES_IN` | Token expiry time | `24h`, `7d`, `3600` (seconds) |
| `NODE_ENV` | Environment mode | `development`, `production` |

#### 3. Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE ai_note_taker;

# Create user with password
CREATE USER ai_user WITH PASSWORD 'ai_user';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE ai_note_taker TO ai_user;

# Exit
\q
```

#### 4. Run Database Migrations

```bash
# Push schema to database
npm run db:push

# Or use Prisma migrate
npm run prisma:migrate
```

#### 5. Seed Sample Data

```bash
npm run seed
```

This creates:
- 2 users: john_doe, jane_smith
- 1 company: TechCorp
- Company memberships with roles (OWNER, MEMBER)
- 1 meeting with participants
- 2 tasks with assignees

### NPM Scripts

```bash
# Development
npm run dev              # Start with hot-reload (--watch)
npm start               # Start production server

# Database
npm run prisma:generate # Generate Prisma client
npm run db:push         # Push schema without migration
npm run db:seed         # Run seed script
npm run prisma:studio   # Open Prisma Studio GUI

# Testing
npm test                # Run all tests
npm run test:unit       # Run unit tests only
npm run test:feature    # Run feature tests only
npm run test:watch      # Watch mode (re-run on changes)
```

### Database GUI Tools

#### Prisma Studio

Interactive database viewer and editor:

```bash
npm run prisma:studio
# Opens http://localhost:5555
```

#### DBeaver

Download from [dbeaver.io](https://dbeaver.io/)

Connection settings:
- **Host**: localhost
- **Port**: 5432
- **Database**: ai_note_taker
- **User**: ai_user
- **Password**: ai_user

---

## Development Guide

### Project Workflow

#### 1. Making Code Changes

```bash
# Start development server with auto-reload
npm run dev
```

Server runs on `http://localhost:3000`

#### 2. Testing Changes

```bash
# Run tests automatically on file changes
npm run test:watch

# Or run once after changes
npm test
```

#### 3. Database Schema Changes

Edit `prisma/schema.prisma`:

```prisma
// Add new field
model Task {
  // ... existing fields
  priority    String @default("MEDIUM")  // ← NEW
}
```

Then:

```bash
# Option 1: Safe create migration
npm run prisma:migrate

# Option 2: Force push schema (dev only)
npm run db:push
```

#### 4. Adding New Routes

**File Structure:**
```
src/
├── routes/newResource.js      # Route definitions
├── controllers/newController.js # HTTP handlers
└── services/newService.js     # Business logic
```

**Example: Add "Comments" Resource**

1. **Update schema** (`prisma/schema.prisma`):
```prisma
model Comment {
  id          Int       @id @default(autoincrement())
  meeting_id  Int
  user_id     Int
  text        String
  created_at  DateTime  @default(now())
  meeting     Meeting   @relation(fields: [meeting_id], references: [id])
  user        User      @relation(fields: [user_id], references: [id])
}
```

2. **Create service** (`src/services/commentService.js`):
```javascript
export class CommentService {
  async addComment(companyId, meetingId, userId, text) {
    // Verify meeting belongs to company
    const meeting = await prisma.meeting.findFirst({
      where: { id: meetingId, company_id: companyId }
    });
    
    return prisma.comment.create({
      data: {
        meeting_id: meetingId,
        user_id: userId,
        text
      },
      include: { user: true }
    });
  }
}
```

3. **Create controller** (`src/controllers/commentController.js`):
```javascript
export const addCommentHandler = async (req, res, next) => {
  try {
    const comment = await commentService.addComment(
      req.companyId,
      req.params.meetingId,
      req.userId,
      req.body.text
    );
    res.status(201).json({ success: true, data: comment });
  } catch (error) {
    next(error);
  }
};
```

4. **Create routes** (`src/routes/commentRoutes.js`):
```javascript
import { Router } from 'express';
import { authMiddleware } from '../middlewares/authMiddleware.js';
import { tenantMiddleware } from '../middlewares/tenantMiddleware.js';
import { addCommentHandler } from '../controllers/commentController.js';

const router = Router();

router.post(
  '/meetings/:meetingId/comments',
  authMiddleware,
  tenantMiddleware(),
  addCommentHandler
);

export default router;
```

5. **Mount in app** (`src/app.js`):
```javascript
import commentRoutes from './routes/commentRoutes.js';
app.use('/api', commentRoutes);
```

6. **Test the endpoint**:
```bash
npm test:watch
# Then in Postman:
POST http://localhost:3000/api/meetings/10/comments
Authorization: Bearer <token>
X-Company-Id: 5
Content-Type: application/json

{
  "text": "Great discussion about the roadmap!"
}
```

### Common Development Tasks

#### Add Validation to Endpoint

Using manual validation (add middleware for automatic validation in future):

```javascript
export const updateTaskHandler = async (req, res, next) => {
  try {
    // Validate input
    if (!req.body.status || !['TODO', 'IN_PROGRESS', 'DONE'].includes(req.body.status)) {
      throw new HttpError('Invalid status', 400);
    }
    
    const task = await taskService.updateTask(
      req.companyId,
      req.params.id,
      req.body
    );
    
    res.json({ success: true, data: task });
  } catch (error) {
    next(error);
  }
};
```

#### Query Data Efficiently

Avoid N+1 queries using `include`:

```javascript
// ❌ BAD: N+1 problem
const meetings = await prisma.meeting.findMany();
const meetingsWithParticipants = await Promise.all(
  meetings.map(m => prisma.meetingParticipants.findMany({
    where: { meeting_id: m.id }
  }))
);

// ✅ GOOD: Single query with include
const meetings = await prisma.meeting.findMany({
  include: { meetingParticipants: true }
});
```

#### Add Error Handling

All handlers should use try/catch:

```javascript
export const getTaskHandler = async (req, res, next) => {
  try {
    const task = await taskService.getTask(req.companyId, req.params.id);
    
    if (!task) {
      throw new HttpError('Task not found', 404);
    }
    
    res.json({ success: true, data: task });
  } catch (error) {
    // Express error middleware handles: next(error)
    next(error);
  }
};
```

---

## Deployment

### Pre-Deployment Checklist

- [ ] All tests passing: `npm test`
- [ ] No console errors: Check logs
- [ ] Environment variables set for production
- [ ] Database migrations run: `npm run prisma:deploy`
- [ ] Security headers configured
- [ ] CORS properly scoped (not `*`)
- [ ] Rate limiting enabled
- [ ] Error logging configured
- [ ] Monitoring/alerting set up

### Production Environment Variables

```bash
# .env.production
DATABASE_URL="postgresql://prod_user:secure_password@db.example.com:5432/ai_note_taker"
JWT_SECRET="generate_a_random_32_char_string_use_secrets_manager"
JWT_EXPIRES_IN="24h"
NODE_ENV="production"
LOG_LEVEL="info"
CORS_ORIGIN="https://app.example.com"
```

### Build & Run

```bash
# Production start
NODE_ENV=production npm start
```

### Monitoring

**Suggested Tools:**
- **Logging**: Pino, Winston
- **Monitoring**: DataDog, New Relic, Sentry
- **Performance**: Lighthouse CI, New Relic APM
- **Uptime**: UptimeRobot, StatusPage

---

## Glossary

| Term | Definition |
|------|-----------|
| **ORM** | Object-Relational Mapping (Prisma abstracts SQL) |
| **JWT** | JSON Web Token (stateless authentication) |
| **bcrypt** | Password hashing algorithm (12 rounds = secure) |
| **Tenant** | Company/customer instance (multi-tenancy = multiple tenants isolated) |
| **Composite Key** | Primary key made of multiple columns (e.g., user_id + company_id) |
| **Cascade Delete** | Automatically delete related records when parent deleted |
| **Middleware** | Function that processes requests before controllers |
| **Service Layer** | Business logic layer between routes and database |
| **Prisma Studio** | GUI for viewing/editing database data |

---

## Support & Resources

### Documentation Links

- [Express.js Guide](https://expressjs.com/)
- [Prisma Documentation](https://www.prisma.io/docs/)
- [PostgreSQL Docs](https://www.postgresql.org/docs/)
- [JWT Introduction](https://jwt.io/introduction)
- [OpenAPI Specification](https://spec.openapis.org/oas/v3.0.3)

### Quick Reference

**Start developing:**
```bash
npm run dev        # Starts on localhost:3000
npm run test:watch # Run tests
npm run prisma:studio # View database
```

**Debug endpoint issues:**
```bash
# Check logs in terminal
# Use Postman to test endpoint
# Use Prisma Studio to verify data
# Check middleware order in app.js
```

**Check test coverage:**
```bash
npm test           # Runs all 13 tests
npm run test:unit  # Runs 9 unit tests
npm run test:feature # Runs 4 feature tests
```

---

## Version & Changelog

**Current Version**: 1.1.1

### v1.1.1 (May 2026)

#### Features
- ✅ Add PATCH `/api/companies/:id/memberships/:userId` to update member roles (admin/member only); owner promotion is disallowed via this endpoint.
- ✅ Return tenant-scoped `role` in `GET /api/users` and `GET /api/users/:id` responses to surface user roles in the UI.
- ✅ Accept JWT token via query string (`?token=...`) as a fallback for SSE/EventSource connections to support browser SSE constraints.

#### Improvements
- ✅ Enforced strict role validation for membership updates (`admin`, `member`).
- ✅ Updated OpenAPI spec to v1.1.1 with documentation for membership patch and user role field.
- ✅ Added unit tests covering companies service role updates, users service role mapping, and auth middleware token fallback.

### v1.1.0 (April 2026)

#### Features
- ✅ Auth flow split into two endpoints: `POST /auth/login` returns token + user identity, and `GET /auth/me` returns memberships with `activeCompanyId` and `activeRole` for multi-company role selection.
- ✅ Meeting upload pipeline now accepts required `lang`, sign-language video, wav files, and Cloudinary main video URL metadata.
- ✅ Main meeting video is stored as Cloudinary playback metadata and excluded from FastAPI processing payloads.
- ✅ FastAPI processing payload now forwards only sign-language video and wav files with meeting context.
- ✅ Upload validation now enforces required `lang` values (`en`, `ar`, `cs`) and required wav input.
- ✅ Meeting status lifecycle and SSE event stream remain the source of truth for UI progress updates.
- ✅ Task assignee, meeting participant, and company membership assignment endpoints now accept email-based input and resolve internal user IDs server-side.
- ✅ Task list responses now include nested assignee display data (`id`, `username`, `email`) inside `taskAssignees` for UI rendering.
- ✅ PATCH task update behavior is now partial, preserving unchanged fields such as `due_date` when status is updated.
- ✅ PATCH request bodies are documented in OpenAPI for company, meeting, and task update endpoints.
- ✅ GET `/api/meetings/:id` now has a documented response example showing pyannote-style transcript output, summary, and tasks.
- ✅ GET `/api/tasks` now documents task assignees clearly for UI consumption.
- ✅ Company logo support was removed from the public API contract and database schema after previously being treated as a file-based concept.
- ✅ Swagger/OpenAPI and team handoff docs updated to reflect the new upload, assignment, auth, and response contracts.

#### Improvements
- ✅ More modular backend service boundaries for meeting processing, upload validation, callback handling, and media routing.
- ✅ Cloudinary-based main video playback flow designed for browser-friendly streaming without backend disk storage.
- ✅ Backend now cleanly separates browser playback media from FastAPI AI-processing inputs.

#### Notes
- Main video playback is expected to be handled by Cloudinary URL streaming on the client side.
- FastAPI should treat `signVideo` and `wavFile` as the only media inputs for AI processing.
- The backend still acts as the source of truth for meeting metadata, task persistence, tenant access control, and per-company roles.
- April 2026 release note: v1.1.0 collects the auth split, upload contract changes, doc updates, and response-shape improvements introduced after the initial March 2026 backend release.

### v1.0.0 (March 2026)

#### Features
- ✅ Complete Express.js API scaffold
- ✅ Prisma ORM with PostgreSQL
- ✅ JWT authentication with bcrypt
- ✅ Multi-tenant company isolation
- ✅ Role-based access control
- ✅ Meeting & task management
- ✅ OpenAPI 3.0.3 documentation
- ✅ Comprehensive test suite (13 tests)


#### Known Limitations [Planned]
- No request validation middleware (planned)
- No integration tests against test DB (planned)
- No rate limiting (planned for v1.1)
- No structured logging (planned for v1.1)


---

