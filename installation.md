# Installation Guide

This guide sets up PostgreSQL, Prisma, environment variables, and the backend API for AI-NoteTaker.

## 1) Prerequisites

- Node.js 20+ recommended
- npm 10+
- PostgreSQL 15+ installed locally

## 2) Create PostgreSQL Database

### Option A: Use psql (recommended)

Open PowerShell and run:

```powershell
psql -U postgres
```

Then run SQL:

```sql
CREATE DATABASE ai_note_taker;
```

Optional dedicated user:

```sql
CREATE USER ai_user WITH PASSWORD 'strong_password_here';
GRANT ALL PRIVILEGES ON DATABASE ai_note_taker TO ai_user;
```

Exit with:

```sql
\q
```

### Option B: Use pgAdmin

- Open pgAdmin
- Right-click Databases
- Create > Database
- Name: ai_note_taker
- Save

## 3) Configure Backend Environment

Copy the backend environment template and update values:

```powershell
cd backend
Copy-Item .env.example .env
```

Update backend/.env with your real credentials.

Example with postgres user:

```env
DATABASE_URL="postgresql://postgres:your_postgres_password@localhost:5432/ai_note_taker?schema=public"
```

Example with dedicated ai_user:

```env
DATABASE_URL="postgresql://ai_user:strong_password_here@localhost:5432/ai_note_taker?schema=public"
```

Also set:

- JWT_SECRET to a long random value
- PORT if you do not want default 3000

## 4) Install Dependencies

From backend folder:

```powershell
npm install
```

If PowerShell blocks npm scripts, use:

```powershell
npm.cmd install
```

## 5) Create DB Tables with Prisma

From backend folder:

```powershell
npm run prisma:generate
npm run prisma:migrate
```

Optional: seed sample data

```powershell
npm run db:seed
```

This creates all schema tables:

- User
- Company
- CompanyMembership
- Meeting
- MeetingParticipants
- Task
- TaskAssignees

Seeded sample users (when running db:seed):

- admin@example.com / Admin@123
- member@example.com / User@123

## 6) Run Backend

Development mode:

```powershell
npm run dev
```

Production mode:

```powershell
npm run start
```

Health check:

```text
GET http://localhost:3000/health
```

Expected response:

```json
{
  "status": "ok",
  "env": "development"
}
```

Swagger UI docs:

```text
http://localhost:3000/docs
```

OpenAPI JSON:

```text
http://localhost:3000/docs.json
```

## 7) API Overview

Base URL:

```text
http://localhost:3000/api
```

Public endpoints:

- POST /auth/register
- POST /auth/login

Protected endpoints (Bearer token required):

- /users
- /companies
- /meetings
- /tasks

Tenant isolation header for protected tenant routes:

```text
X-Company-Id: <company_id>
```

Use this header for:

- /users
- /meetings
- /tasks

Company admin actions (add/remove members) are restricted to users with role owner/admin in that company.

## 8) Exact Verification Steps (Run In Order)

1. Start PostgreSQL service.
2. Create database ai_note_taker.
3. In backend folder, create .env from .env.example.
4. Set DATABASE_URL and JWT_SECRET in .env.
5. Run:

```powershell
npm.cmd install
npm.cmd run prisma:generate
npm.cmd run prisma:migrate
npm.cmd run db:seed
npm.cmd run dev
```

6. Verify health endpoint:
  - GET http://localhost:3000/health
7. Login with seeded admin:
  - POST http://localhost:3000/api/auth/login
  - body: { "email": "admin@example.com", "password": "Admin@123" }
8. Copy returned token into Authorization header:
  - Bearer <token>
9. Set tenant header:
  - X-Company-Id: 1
10. Test tenant-scoped endpoint:
  - GET http://localhost:3000/api/users


## 9) Useful Prisma Commands

Run from backend folder:

```powershell
npm run prisma:studio
npm run prisma:deploy
npm run db:push
```

## 10) Common Troubleshooting

- Authentication failed for postgres user:
  - Verify DATABASE_URL username/password
- Database does not exist:
  - Create ai_note_taker first in psql or pgAdmin
- P1001 cannot reach database server:
  - Ensure PostgreSQL service is running and port 5432 is open
- npm blocked by PowerShell execution policy:
  - Use npm.cmd instead of npm
- Prisma client out of date after schema changes:
  - Run npm run prisma:generate again
