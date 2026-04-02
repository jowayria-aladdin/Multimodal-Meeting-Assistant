import app from "./app.js";
import { env } from "./config/env.js";
import { prisma } from "./config/prisma.js";

const server = app.listen(env.port, () => {
  console.log(`Backend server listening on port ${env.port}`);
});

const shutdown = async () => {
  console.log("Shutting down server...");
  server.close(async () => {
    await prisma.$disconnect();
    process.exit(0);
  });
};

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
