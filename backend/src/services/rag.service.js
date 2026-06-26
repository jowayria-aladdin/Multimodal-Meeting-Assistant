import { GoogleGenerativeAI } from "@google/generative-ai";
import { env } from "../config/env.js";
import { prisma } from "../config/prisma.js";
import { httpError } from "../utils/httpError.js";

const genAI = new GoogleGenerativeAI(env.geminiApiKey);
const embeddingModel = genAI.getGenerativeModel({ model: env.geminiEmbeddingModel });
const chatModel = genAI.getGenerativeModel({
  model: env.geminiChatModel,
  systemInstruction:
    "You are an AI assistant for meeting notes. Answer questions using only the provided meeting context. " +
    "If the answer is not in the context, say clearly that you could not find relevant information. " +
    "Be concise and cite the meeting title when relevant."
});

const WORD_LIMIT = 400;

function chunkTranscript(segments) {
  if (!Array.isArray(segments) || segments.length === 0) return [];

  const chunks = [];
  let buffer = [];
  let wordCount = 0;
  let chunkIndex = 0;

  const flush = () => {
    if (buffer.length === 0) return;
    const speaker = buffer[0].speaker || "Unknown";
    const text = buffer.map(s => s.text || "").join(" ").trim();
    if (!text) { buffer = []; wordCount = 0; return; }
    chunks.push({
      content: `[${speaker}]: ${text}`,
      type: "transcript",
      metadata: { speaker, start: buffer[0].start, end: buffer[buffer.length - 1].end, chunk_index: chunkIndex++ }
    });
    buffer = [];
    wordCount = 0;
  };

  for (const seg of segments) {
    const segWords = (seg.text || "").split(/\s+/).filter(Boolean).length;
    const speakerChanged = buffer.length > 0 && buffer[0].speaker !== seg.speaker;

    if (speakerChanged || (wordCount + segWords > WORD_LIMIT && buffer.length > 0)) {
      flush();
    }

    buffer.push(seg);
    wordCount += segWords;
  }
  flush();

  return chunks;
}

function chunkSummary(summary) {
  if (!summary || typeof summary !== "string" || !summary.trim()) return [];

  return summary
    .split(/\n\n+/)
    .map(p => p.trim())
    .filter(Boolean)
    .map((text, i) => ({
      content: text,
      type: "summary",
      metadata: { chunk_index: i }
    }));
}

function chunkTasks(tasks) {
  if (!Array.isArray(tasks) || tasks.length === 0) return [];

  return tasks.map((task, i) => {
    const parts = [`Task: ${task.task_text}`];
    if (task.assignee) parts.push(`Assignee: ${task.assignee}`);
    if (task.due_date) parts.push(`Due: ${new Date(task.due_date).toLocaleDateString()}`);
    parts.push(`Status: ${task.status}`);
    return {
      content: parts.join(". "),
      type: "task",
      metadata: { task_id: task.id, chunk_index: i }
    };
  });
}

async function generateEmbeddings(texts) {
  const embeddings = [];
  for (const text of texts) {
    const result = await embeddingModel.embedContent(text);
    embeddings.push(result.embedding.values);
  }
  return embeddings;
}

function cosineSimilarity(a, b) {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom === 0 ? 0 : dot / denom;
}

export const indexMeeting = async (meetingId, companyId) => {
  const meeting = await prisma.meeting.findFirst({
    where: { id: meetingId, company_id: companyId },
    include: { tasks: true }
  });

  if (!meeting) throw httpError(404, "Meeting not found");
  if (meeting.processing_status !== "COMPLETED") {
    throw httpError(400, "Meeting must be COMPLETED before indexing");
  }

  await prisma.meetingChunk.deleteMany({ where: { meeting_id: meetingId } });

  const allChunks = [
    ...chunkTranscript(meeting.transcript || []),
    ...chunkSummary(meeting.summary || ""),
    ...chunkTasks(meeting.tasks)
  ];

  if (allChunks.length === 0) return { meetingId, chunksCreated: 0 };

  const embeddings = await generateEmbeddings(allChunks.map(c => c.content));

  await prisma.meetingChunk.createMany({
    data: allChunks.map((chunk, i) => ({
      meeting_id: meetingId,
      company_id: meeting.company_id,
      chunk_type: chunk.type,
      content: chunk.content,
      embedding: embeddings[i],
      metadata: chunk.metadata
    }))
  });

  return { meetingId, chunksCreated: allChunks.length };
};

export const indexMeetingSafe = async (meetingId) => {
  try {
    const row = await prisma.meeting.findUnique({
      where: { id: meetingId },
      select: { company_id: true }
    });
    if (!row) return;
    const result = await indexMeeting(meetingId, row.company_id);
    console.log(`[RAG] Indexed meeting ${meetingId}: ${result.chunksCreated} chunks`);
  } catch (err) {
    console.error(`[RAG] Failed to index meeting ${meetingId}:`, err.message);
  }
};

export const queryRag = async (companyId, question, options = {}) => {
  if (!question || typeof question !== "string" || !question.trim()) {
    throw httpError(400, "question is required");
  }
  if (question.length > 1000) {
    throw httpError(400, "question must be 1000 characters or less");
  }

  const { meetingIds, topK = 5 } = options;

  const [queryEmbedding] = await generateEmbeddings([question.trim()]);

  const chunks = await prisma.meetingChunk.findMany({
    where: {
      company_id: companyId,
      ...(meetingIds?.length ? { meeting_id: { in: meetingIds.map(Number) } } : {})
    },
    select: {
      id: true,
      meeting_id: true,
      chunk_type: true,
      content: true,
      embedding: true,
      metadata: true,
      meeting: { select: { id: true, title: true, created_at: true } }
    }
  });

  if (chunks.length === 0) {
    return {
      answer: "No indexed meeting content found. Please wait for your meetings to finish processing and be indexed.",
      sources: []
    };
  }

  const ranked = chunks
    .map(chunk => ({ ...chunk, similarity: cosineSimilarity(queryEmbedding, chunk.embedding) }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);

  const context = ranked
    .map((c, i) => `[${i + 1}] Meeting: "${c.meeting.title}" (${c.chunk_type})\n${c.content}`)
    .join("\n\n");

  const prompt = `Context from meeting notes:\n\n${context}\n\nQuestion: ${question}`;

  const result = await chatModel.generateContent({
    contents: [{ role: "user", parts: [{ text: prompt }] }]
  });

  return {
    answer: result.response.text(),
    sources: ranked.map(c => ({
      chunkId: c.id,
      meetingId: c.meeting_id,
      meetingTitle: c.meeting.title,
      chunkType: c.chunk_type,
      content: c.content,
      similarity: Math.round(c.similarity * 1000) / 1000
    }))
  };
};

export const listIndexedMeetings = async (companyId) => {
  return prisma.meeting.findMany({
    where: { company_id: companyId, chunks: { some: {} } },
    select: { id: true, title: true, created_at: true },
    orderBy: { created_at: "desc" }
  });
};

export const getIndexStatus = async (meetingId, companyId) => {
  const meeting = await prisma.meeting.findFirst({
    where: { id: meetingId, company_id: companyId },
    select: { id: true, title: true }
  });

  if (!meeting) throw httpError(404, "Meeting not found");

  const chunkCount = await prisma.meetingChunk.count({ where: { meeting_id: meetingId } });

  return { meetingId, meetingTitle: meeting.title, indexed: chunkCount > 0, chunkCount };
};
