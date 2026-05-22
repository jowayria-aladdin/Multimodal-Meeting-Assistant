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

const signVideoMimes = new Set(["video/webm", "audio/webm"]);
const wavMimes = new Set(["audio/wav", "audio/wave", "audio/x-wav", "audio/webm", "video/webm"]);

const fileFilter = (req, file, cb) => {
  if (file.fieldname === "signVideo") {
    if (signVideoMimes.has(file.mimetype)) {
      cb(null, true);
      return;
    }

    const error = new Error(`signVideo must be webm. Received: ${file.mimetype}`);
    error.status = 400;
    cb(error);
    return;
  }

  if (file.fieldname === "wavFile") {
    if (wavMimes.has(file.mimetype)) {
      cb(null, true);
      return;
    }

    const error = new Error(`wavFile must be wav. Received: ${file.mimetype}`);
    error.status = 400;
    cb(error);
    return;
  }

  const error = new Error(`Unsupported upload field: ${file.fieldname}`);
  error.status = 400;
  cb(error);
};

export const uploadMeetingAudio = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 50 * 1024 * 1024,
    files: 2
  }
}).fields([
  { name: "signVideo", maxCount: 1 },
  { name: "wavFile", maxCount: 1 }
]);
