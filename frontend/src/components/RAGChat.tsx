"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  MessageSquare, Send, SlidersHorizontal, Loader2,
  ChevronDown, ChevronUp, X, AlertCircle, BrainCircuit
} from "lucide-react";
import { apiFetch } from "@/lib/api";

interface Meeting {
  id: string | number;
  title: string;
  created_at?: string;
}

interface Source {
  chunkId: number;
  meetingId: number;
  meetingTitle: string;
  chunkType: string;
  content: string;
  similarity: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  isLoading?: boolean;
  isError?: boolean;
}

interface Props {
  meetings: Meeting[];
  isLoading: boolean;
}

const EXAMPLE_QUESTIONS = [
  "What were the main topics discussed?",
  "Who was assigned tasks from the last meeting?",
  "What decisions were made about the project?",
  "Summarize the key action items."
];

const CHUNK_TYPE_STYLES: Record<string, string> = {
  transcript: "bg-blue-50 text-blue-700 border-blue-100",
  summary: "bg-green-50 text-green-700 border-green-100",
  task: "bg-amber-50 text-amber-700 border-amber-100"
};

function SourcesAccordion({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-2 border border-slate-100 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-sm text-slate-500 hover:bg-slate-50 transition-colors"
      >
        <span className="font-medium">{sources.length} source{sources.length !== 1 ? "s" : ""}</span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {open && (
        <div className="border-t border-slate-100 divide-y divide-slate-100">
          {sources.map(src => (
            <div key={src.chunkId} className="px-4 py-3">
              <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                <span className="text-xs font-semibold text-slate-700 truncate max-w-[180px]">
                  {src.meetingTitle}
                </span>
                <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${CHUNK_TYPE_STYLES[src.chunkType] || "bg-slate-50 text-slate-600 border-slate-100"}`}>
                  {src.chunkType}
                </span>
                <span className="text-xs text-slate-400 ml-auto">
                  {Math.round(src.similarity * 100)}% match
                </span>
              </div>
              <p className="text-xs text-slate-500 line-clamp-3 leading-relaxed">{src.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function RAGChat({ meetings, isLoading }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [selectedMeetingIds, setSelectedMeetingIds] = useState<number[]>([]);
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [indexedMeetings, setIndexedMeetings] = useState<Meeting[]>([]);
  const [isLoadingIndexed, setIsLoadingIndexed] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const filterRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

  useEffect(() => {
    const fetchIndexed = async () => {
      try {
        const data = await apiFetch<Meeting[]>("/rag/meetings");
        setIndexedMeetings(Array.isArray(data) ? data : []);
      } catch {
        setIndexedMeetings([]);
      } finally {
        setIsLoadingIndexed(false);
      }
    };
    fetchIndexed();
  }, []);

  // Close filter dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (filterRef.current && !filterRef.current.contains(e.target as Node)) {
        setIsFilterOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const toggleMeetingFilter = (id: number) => {
    setSelectedMeetingIds(prev =>
      prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
    );
  };

  const handleSubmit = async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed || isSending) return;

    const userMsgId = crypto.randomUUID();
    const assistantMsgId = crypto.randomUUID();

    setMessages(prev => [
      ...prev,
      { id: userMsgId, role: "user", content: trimmed },
      { id: assistantMsgId, role: "assistant", content: "", isLoading: true }
    ]);
    setInput("");
    setIsSending(true);

    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = "auto";

    try {
      const result = await apiFetch<{ answer: string; sources: Source[] }>("/rag/query", {
        method: "POST",
        data: {
          question: trimmed,
          ...(selectedMeetingIds.length ? { meetingIds: selectedMeetingIds } : {})
        }
      });

      setMessages(prev => prev.map(m =>
        m.id === assistantMsgId
          ? { ...m, content: result.answer, sources: result.sources, isLoading: false }
          : m
      ));
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Something went wrong. Please try again.";
      setMessages(prev => prev.map(m =>
        m.id === assistantMsgId
          ? { ...m, content: msg, isLoading: false, isError: true }
          : m
      ));
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(input);
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
  };

  const hasNoIndexed = !isLoadingIndexed && indexedMeetings.length === 0;
  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full max-h-[calc(100vh-160px)] bg-white rounded-2xl border border-slate-200 overflow-hidden">

      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-brand-maroon/10 text-brand-maroon rounded-xl flex items-center justify-center">
            <BrainCircuit size={18} />
          </div>
          <div>
            <h2 className="font-semibold text-slate-800 text-sm">Ask AI</h2>
            <p className="text-xs text-slate-400">
              {isLoadingIndexed
                ? "Loading..."
                : indexedMeetings.length === 0
                  ? "No indexed meetings yet"
                  : `${indexedMeetings.length} meeting${indexedMeetings.length !== 1 ? "s" : ""} indexed`}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
              title="Clear conversation"
            >
              <X size={16} />
            </button>
          )}

          {/* Meeting filter */}
          <div className="relative" ref={filterRef}>
            <button
              onClick={() => setIsFilterOpen(v => !v)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedMeetingIds.length > 0
                  ? "bg-brand-maroon/10 text-brand-maroon"
                  : "text-slate-500 hover:bg-slate-100"
              }`}
            >
              <SlidersHorizontal size={14} />
              {selectedMeetingIds.length > 0 ? `${selectedMeetingIds.length} filter${selectedMeetingIds.length !== 1 ? "s" : ""}` : "Filter"}
            </button>

            {isFilterOpen && (
              <div className="absolute right-0 top-full mt-2 w-72 bg-white border border-slate-200 rounded-xl shadow-lg z-10 overflow-hidden">
                <div className="px-4 py-3 border-b border-slate-100">
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Filter by meeting</p>
                  <p className="text-xs text-slate-400 mt-0.5">Leave empty to search all indexed meetings</p>
                </div>
                <div className="max-h-56 overflow-y-auto">
                  {indexedMeetings.length === 0 ? (
                    <p className="px-4 py-3 text-sm text-slate-400">No indexed meetings available</p>
                  ) : (
                    indexedMeetings.map(m => (
                      <label
                        key={m.id}
                        className="flex items-center gap-3 px-4 py-3 hover:bg-slate-50 cursor-pointer"
                      >
                        <input
                          type="checkbox"
                          checked={selectedMeetingIds.includes(Number(m.id))}
                          onChange={() => toggleMeetingFilter(Number(m.id))}
                          className="rounded border-slate-300 text-brand-maroon focus:ring-brand-maroon"
                        />
                        <span className="text-sm text-slate-700 truncate">{m.title}</span>
                      </label>
                    ))
                  )}
                </div>
                {selectedMeetingIds.length > 0 && (
                  <div className="px-4 py-2 border-t border-slate-100">
                    <button
                      onClick={() => setSelectedMeetingIds([])}
                      className="text-xs text-brand-maroon hover:underline"
                    >
                      Clear filters
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* No indexed meetings banner */}
      {hasNoIndexed && (
        <div className="mx-4 mt-4 px-4 py-3 bg-amber-50 border border-amber-200 rounded-xl flex items-start gap-3 shrink-0">
          <AlertCircle size={16} className="text-amber-500 mt-0.5 shrink-0" />
          <p className="text-sm text-amber-700">
            No meetings have been indexed yet. Meetings are indexed automatically once processing completes.
          </p>
        </div>
      )}

      {/* Messages / Empty state */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {isEmpty ? (
          <div className="h-full flex flex-col items-center justify-center text-center py-8">
            <div className="w-16 h-16 bg-brand-maroon/10 text-brand-maroon rounded-2xl flex items-center justify-center mb-4">
              <MessageSquare size={28} />
            </div>
            <h3 className="font-semibold text-slate-700 mb-1">Ask about your meetings</h3>
            <p className="text-sm text-slate-400 mb-6 max-w-xs">
              Ask anything about your meeting transcripts, summaries, and action items.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-md">
              {EXAMPLE_QUESTIONS.map(q => (
                <button
                  key={q}
                  onClick={() => handleSubmit(q)}
                  disabled={isSending || hasNoIndexed}
                  className="text-left px-4 py-3 border border-brand-gold/40 rounded-xl text-sm text-slate-600 hover:bg-brand-gold/10 hover:border-brand-gold transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map(msg => (
              <div
                key={msg.id}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className={`max-w-[80%] ${msg.role === "user" ? "order-1" : ""}`}>
                  <div
                    className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                      msg.role === "user"
                        ? "bg-brand-maroon text-white rounded-tr-sm"
                        : msg.isError
                          ? "bg-red-50 text-red-700 border border-red-100 rounded-tl-sm"
                          : "bg-slate-50 text-slate-800 border border-slate-100 rounded-tl-sm"
                    }`}
                  >
                    {msg.isLoading ? (
                      <Loader2 size={16} className="animate-spin text-slate-400" />
                    ) : (
                      <span className="whitespace-pre-wrap">{msg.content}</span>
                    )}
                  </div>
                  {!msg.isLoading && msg.sources && msg.sources.length > 0 && (
                    <SourcesAccordion sources={msg.sources} />
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input area */}
      <div className="px-6 py-4 border-t border-slate-100 shrink-0">
        <div className="flex items-end gap-3 bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 focus-within:border-brand-gold focus-within:ring-1 focus-within:ring-brand-gold transition-all">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            placeholder={hasNoIndexed ? "No meetings indexed yet..." : "Ask about your meetings..."}
            disabled={isSending || hasNoIndexed}
            rows={1}
            className="flex-1 bg-transparent text-sm text-slate-800 placeholder:text-slate-400 resize-none outline-none disabled:cursor-not-allowed"
            style={{ maxHeight: "120px" }}
          />
          <button
            onClick={() => handleSubmit(input)}
            disabled={!input.trim() || isSending || hasNoIndexed}
            className="w-8 h-8 bg-brand-maroon text-white rounded-lg flex items-center justify-center shrink-0 hover:bg-brand-gold transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isSending ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
          </button>
        </div>
        <p className="text-xs text-slate-400 mt-1.5 text-center">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
