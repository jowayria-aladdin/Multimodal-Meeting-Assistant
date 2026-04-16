import path from "path";
import { mkdirSync } from "fs";
import multer from "multer";

const uploadDir = path.resolve(process.cwd(), "tmp", "uploads");
mkdirSync(uploadDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const safeBase = file.originalname.replace(/[^a-zA-Z0-9_.-]/g, "_");
    cb(null, `${Date.now()}-${safeBase}`);
  }
});

const allowedMimes = new Set(["audio/webm", "audio/wav", "audio/wave", "audio/x-wav"]);

const fileFilter = (req, file, cb) => {
  if (!allowedMimes.has(file.mimetype)) {
    const error = new Error(`Unsupported file type: ${file.mimetype}`);
    error.status = 400;
    cb(error);
    return;
  }

  cb(null, true);
};

export const uploadMeetingAudio = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 50 * 1024 * 1024,
    files: 12
  }
}).fields([
  { name: "webmFile", maxCount: 1 },
  { name: "wavFiles", maxCount: 10 }
]);
