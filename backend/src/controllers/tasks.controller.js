import { asyncHandler } from "../utils/asyncHandler.js";
import * as tasksService from "../services/tasks.service.js";

export const createTask = asyncHandler(async (req, res) => {
  const task = await tasksService.createTask(req.tenant.companyId, req.body);
  res.status(201).json(task);
});

export const listTasks = asyncHandler(async (req, res) => {
  const tasks = await tasksService.listTasks(req.tenant.companyId);
  res.status(200).json(tasks);
});

export const getTaskById = asyncHandler(async (req, res) => {
  const task = await tasksService.getTaskById(Number(req.params.id), req.tenant.companyId);
  res.status(200).json(task);
});

export const updateTask = asyncHandler(async (req, res) => {
  const task = await tasksService.updateTask(Number(req.params.id), req.tenant.companyId, req.body);
  res.status(200).json(task);
});

export const deleteTask = asyncHandler(async (req, res) => {
  await tasksService.deleteTask(Number(req.params.id), req.tenant.companyId);
  res.status(204).send();
});

export const addTaskAssignee = asyncHandler(async (req, res) => {
  const assignee = await tasksService.addTaskAssignee(
    Number(req.params.id),
    Number(req.body.user_id),
    req.tenant.companyId
  );
  res.status(201).json(assignee);
});

export const removeTaskAssignee = asyncHandler(async (req, res) => {
  await tasksService.removeTaskAssignee(
    Number(req.params.id),
    Number(req.params.userId),
    req.tenant.companyId
  );
  res.status(204).send();
});
