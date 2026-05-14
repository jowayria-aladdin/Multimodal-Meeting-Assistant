import { prisma } from "../config/prisma.js";
import { httpError } from "../utils/httpError.js";

const assigneeSelection = {
  task_id: true,
  user_id: true,
  user: {
    select: {
      id: true,
      username: true,
      email: true
    }
  }
};

const resolveUserByEmail = async (email) => {
  const normalizedEmail = typeof email === "string" ? email.trim() : "";

  if (!normalizedEmail) {
    throw httpError(400, "email is required");
  }

  const user = await prisma.user.findUnique({
    where: { email: normalizedEmail }
  });

  if (!user) {
    throw httpError(404, "User not found");
  }

  return user;
};

export const createTask = async (companyId, { meeting_id, task_text, due_date, status }) => {
  if (!meeting_id || !task_text) {
    throw httpError(400, "meeting_id and task_text are required");
  }

  const meeting = await prisma.meeting.findFirst({
    where: {
      id: meeting_id,
      company_id: companyId
    }
  });
  if (!meeting) {
    throw httpError(404, "Meeting not found in your company");
  }

  return prisma.task.create({
    data: {
      meeting_id,
      task_text,
      due_date: due_date ? new Date(due_date) : null,
      status
    }
  });
};

export const listTasks = async (companyId) => prisma.task.findMany({
  where: {
    meeting: {
      company_id: companyId
    }
  },
  orderBy: { id: "asc" },
  include: {
    taskAssignees: {
      select: assigneeSelection
    }
  }
});

export const getTaskById = async (id, companyId) => {
  const task = await prisma.task.findFirst({
    where: {
      id,
      meeting: {
        company_id: companyId
      }
    },
    include: {
      taskAssignees: {
        select: assigneeSelection
      }
    }
  });

  if (!task) {
    throw httpError(404, "Task not found");
  }

  return task;
};

export const updateTask = async (id, companyId, payload) => {
  await getTaskById(id, companyId);

  if (payload.meeting_id !== undefined) {
    const meeting = await prisma.meeting.findFirst({
      where: {
        id: payload.meeting_id,
        company_id: companyId
      }
    });

    if (!meeting) {
      throw httpError(400, "meeting_id must belong to your company");
    }
  }

  const data = {};

  if (payload.meeting_id !== undefined) {
    data.meeting_id = payload.meeting_id;
  }

  if (payload.task_text !== undefined) {
    data.task_text = payload.task_text;
  }

  if (Object.prototype.hasOwnProperty.call(payload, "due_date")) {
    data.due_date = payload.due_date ? new Date(payload.due_date) : null;
  }

  if (payload.status !== undefined) {
    data.status = payload.status;
  }

  if (!Object.keys(data).length) {
    throw httpError(400, "At least one field is required for update");
  }

  return prisma.task.update({
    where: { id },
    data
  });
};

export const deleteTask = async (id, companyId) => {
  await getTaskById(id, companyId);
  await prisma.task.delete({ where: { id } });
};

export const addTaskAssignee = async (taskId, email, companyId) => {
  const targetUser = await resolveUserByEmail(email);

  const [task, user] = await Promise.all([
    prisma.task.findFirst({
      where: {
        id: taskId,
        meeting: {
          company_id: companyId
        }
      }
    }),
    prisma.user.findFirst({
      where: {
        id: targetUser.id,
        companyMemberships: {
          some: {
            company_id: companyId
          }
        }
      }
    })
  ]);

  if (!task) {
    throw httpError(404, "Task not found");
  }

  if (!user) {
    throw httpError(404, "User not found");
  }

  return prisma.taskAssignees.upsert({
    where: {
      task_id_user_id: {
        task_id: taskId,
        user_id: targetUser.id
      }
    },
    update: {},
    create: {
      task_id: taskId,
      user_id: targetUser.id
    }
  });
};

export const removeTaskAssignee = async (taskId, userId, companyId) => {
  await getTaskById(taskId, companyId);

  const existing = await prisma.taskAssignees.findUnique({
    where: {
      task_id_user_id: {
        task_id: taskId,
        user_id: userId
      }
    }
  });

  if (!existing) {
    throw httpError(404, "Assignee not found for task");
  }

  await prisma.taskAssignees.delete({
    where: {
      task_id_user_id: {
        task_id: taskId,
        user_id: userId
      }
    }
  });
};
