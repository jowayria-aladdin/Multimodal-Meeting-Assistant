import express from "express";
import swaggerUi from "swagger-ui-express";
import { env } from "./config/env.js";
import { apiRouter } from "./routes/index.js";
import { errorHandler, notFoundHandler } from "./middlewares/errorHandler.js";
import { openApiSpec } from "./docs/openapi.js";

const app = express();

app.use(express.json());

app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok", env: env.nodeEnv });
});

app.get("/docs.json", (req, res) => {
  res.status(200).json(openApiSpec);
});

app.use("/docs", swaggerUi.serve, swaggerUi.setup(openApiSpec));

app.use("/api", apiRouter);
app.use(notFoundHandler);
app.use(errorHandler);

export default app;
