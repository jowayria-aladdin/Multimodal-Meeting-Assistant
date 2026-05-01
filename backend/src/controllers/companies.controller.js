import { asyncHandler } from "../utils/asyncHandler.js";
import * as companiesService from "../services/companies.service.js";

export const createCompany = asyncHandler(async (req, res) => {
  const company = await companiesService.createCompany(req.body, Number(req.user.id));
  res.status(201).json(company);
});

export const listCompanies = asyncHandler(async (req, res) => {
  const companies = await companiesService.listCompanies(Number(req.user.id));
  res.status(200).json(companies);
});

export const getCompanyById = asyncHandler(async (req, res) => {
  const company = await companiesService.getCompanyById(Number(req.params.id), Number(req.user.id));
  res.status(200).json(company);
});

export const updateCompany = asyncHandler(async (req, res) => {
  const company = await companiesService.updateCompany(
    Number(req.params.id),
    req.body,
    Number(req.user.id)
  );
  res.status(200).json(company);
});

export const deleteCompany = asyncHandler(async (req, res) => {
  await companiesService.deleteCompany(Number(req.params.id), Number(req.user.id));
  res.status(204).send();
});

export const addMembership = asyncHandler(async (req, res) => {
  const membership = await companiesService.addMembership(Number(req.params.id), req.body);
  res.status(201).json(membership);
});

export const updateMembershipRole = asyncHandler(async (req, res) => {
  const membership = await companiesService.updateMembershipRole(
    Number(req.params.id),
    Number(req.params.userId),
    req.body
  );
  res.status(200).json(membership);
});

export const removeMembership = asyncHandler(async (req, res) => {
  await companiesService.removeMembership(Number(req.params.id), Number(req.params.userId));
  res.status(204).send();
});
