import dotenv from "dotenv";

dotenv.config();

const required = ["DATABASE_URL", "JWT_SECRET", "GEMINI_API_KEY"];

for (const key of required) {
  if (!process.env[key]) {
    throw new Error(`Missing required environment variable: ${key}`);
  }
}

export const env = {
  nodeEnv: process.env.NODE_ENV || "development",
  port: Number(process.env.PORT || 3000),
  databaseUrl: process.env.DATABASE_URL,
  jwtSecret: process.env.JWT_SECRET,
  jwtExpiresIn: process.env.JWT_EXPIRES_IN || "7d",
  corsOrigin: process.env.CORS_ORIGIN || "*",
  fastApiProcessUrl: process.env.FASTAPI_PROCESS_URL || "http://localhost:5000/process-audio",
  internalCallbackSecret: process.env.INTERNAL_CALLBACK_SECRET || "some_long_random_secret_here",
  geminiApiKey: process.env.GEMINI_API_KEY,
  geminiEmbeddingModel: process.env.GEMINI_EMBEDDING_MODEL || "gemini-embedding-004",
  geminiChatModel: process.env.GEMINI_CHAT_MODEL || "gemini-2.0-flash"
};
