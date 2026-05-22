"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter, useParams } from "next/navigation";
import { ArrowLeft, Play, Pause, Volume2, VolumeX, Maximize, Minimize, FileText, CheckSquare, MessageSquare, Clock, Loader2, AlertTriangle, X } from "lucide-react";
import { EventSourcePolyfill } from "event-source-polyfill";
import { apiFetch } from "@/lib/api";

interface TranscriptLine {
  time?: string;
  seconds?: number;
  start?: number; 
  end?: number;  
  speaker: string;
  text: string;
}

interface RealMeeting { 
  id: string | number; 
  title: string; 
  created_at?: string; 
  status?: string; 
  processing_status?: string; 
  main_video_url?: string;
  summary?: string | string[] | { text: string } | { summary: { text: string } }; 
  transcript?: string | TranscriptLine[]; // UPDATED: Now accepts the single string block    
  transcription?: TranscriptLine[]; 
  sign_transcript?: TranscriptLine[];
  name_recognition?: { 
    mappings: { speaker_id: string; predicted_name: string | null; confidence?: number }[] 
  };    
}

interface RealTask { 
  id: string | number; 
  task_text: string; 
  status?: string; 
  due_date?: string; 
  meeting_id?: string | number; 
}

// Typed API responses
type MeetingResponse = RealMeeting | { data?: RealMeeting };
type TasksResponse = RealTask[] | { data?: RealTask[] };

export default function MeetingDetailsWorkstation() {
  const router = useRouter();
  const params = useParams();
  const meetingId = params.id;
  
  const [leftTab, setLeftTab] = useState<'SUMMARY' | 'TASKS'>('SUMMARY');
  const [rightTab, setRightTab] = useState<'AUDIO' | 'SIGN'>('AUDIO');

  const [meeting, setMeeting] = useState<RealMeeting | null>(null);
  const [tasks, setTasks] = useState<RealTask[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  const [progressPercent, setProgressPercent] = useState(0); 
  const [liveStatus, setLiveStatus] = useState<string | null>(null);
  const [updatingTaskId, setUpdatingTaskId] = useState<string | number | null>(null);

  // NATIVE MEDIA REFS 
  const playerContainerRef = useRef<HTMLDivElement>(null); 
  const mainVideoRef = useRef<HTMLVideoElement>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [playedProgress, setPlayedProgress] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isFixingDuration, setIsFixingDuration] = useState(false); 

  // FALLBACK MOCK MEDI testing purposr
  const FALLBACK_MAIN_VIDEO = "https://media.w3.org/2010/05/sintel/trailer_hd.mp4"; 

  const toggleFullScreen = () => {
    if (!document.fullscreenElement) {
      playerContainerRef.current?.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const togglePlay = () => {
    if (!mainVideoRef.current) return;
    if (isPlaying) {
      mainVideoRef.current.pause(); 
    } else {
      mainVideoRef.current.play().catch(e => console.warn("Main video blocked:", e)); 
    }
  };


  const handleLoadedMetadata = () => {
    const video = mainVideoRef.current;
    if (!video) return;

    if (video.duration === Infinity) {
      setIsFixingDuration(true);
      video.currentTime = 1e101; 
      
      const onSeeked = () => {
        video.currentTime = 0; 
        setIsFixingDuration(false);
        video.removeEventListener('seeked', onSeeked);
      };
      
      video.addEventListener('seeked', onSeeked);
    }
  };

  // duration getter
  const getSafeDuration = useCallback(() => {
    const videoDuration = mainVideoRef.current?.duration;
    
    if (videoDuration && isFinite(videoDuration) && videoDuration > 0) {
      return videoDuration;
    }

    const rawTranscript = meeting?.transcript || meeting?.transcription;
    if (rawTranscript && Array.isArray(rawTranscript) && rawTranscript.length > 0) {
      const lastLine = rawTranscript[rawTranscript.length - 1];
      const inferredTime = lastLine.end || lastLine.start || lastLine.seconds || 0;
      if (inferredTime > 0) {
        return inferredTime + 2; 
      }
    }

    return 0; 
  }, [meeting]);

  const handleTimeUpdate = () => {
    if (isFixingDuration) return; 

    const video = mainVideoRef.current;
    if (!video) return;

    const total = getSafeDuration();
    if (total > 0 && isFinite(total)) {
      setPlayedProgress(Math.min(100, Math.max(0, (video.currentTime / total) * 100)));
    }
  };

  const handleProgressBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isFixingDuration) return; 

    const video = mainVideoRef.current;
    if (!video) return; 

    const total = getSafeDuration();
    if (total <= 0 || !isFinite(total)) return; 

    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return; 

    const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const newTime = percent * total;

    if (isFinite(newTime)) {
      setPlayedProgress(percent * 100);
      video.currentTime = newTime; 
    }
  };

  //  CLICK-TO-SEEK TRANSCRIPT 
  const handleTranscriptClick = (startTime?: number) => {
    const video = mainVideoRef.current;
    const total = getSafeDuration();
    if (!video || typeof startTime !== 'number' || total <= 0 || !isFinite(total)) return;

    if (isFinite(startTime)) {
      video.currentTime = startTime;
      setPlayedProgress((startTime / total) * 100);
      video.play().catch(() => {});
      setIsPlaying(true);
    }
  };

  const fetchRealData = useCallback(async () => {
    try {
      // Fetch Meeting Details via apiFetch
      const meetingData = await apiFetch<MeetingResponse>(`/meetings/${meetingId}`);
      const actualMeeting = ('data' in meetingData && meetingData.data) ? meetingData.data : meetingData as RealMeeting;
      setMeeting(actualMeeting);

      //  Fetch Tasks via apiFetch
      const tasksData = await apiFetch<TasksResponse>(`/tasks`);
      const allTasks = Array.isArray(tasksData) ? tasksData : tasksData.data || [];
      
      // Filter tasks to only show ones matching this specific meeting ID
      setTasks(allTasks.filter((t: RealTask) => String(t.meeting_id) === String(meetingId)));

    } catch (err) {
      console.error("Failed to fetch data:", err);
    } finally {
      setIsLoading(false);
    }
  }, [meetingId]);

  useEffect(() => {
    if (meetingId) {
      setIsLoading(true);
      fetchRealData();
    }
  }, [meetingId, fetchRealData]);

  // Subscribe to Server-Sent Events
  useEffect(() => {
    if (!meetingId) return;
    
    const dbStatus = (meeting?.processing_status || meeting?.status || '').toUpperCase();

    if (dbStatus === 'COMPLETED' || dbStatus === 'FAILED' || dbStatus === 'CANCELLED') {
       return;
    }

    const token = localStorage.getItem("token");
    const companyId = localStorage.getItem("companyId");
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3000/api";
    
    const url = `${API_URL}/meetings/${meetingId}/events`;
    
    const es = new EventSourcePolyfill(url, {
      headers: {
        "Authorization": `Bearer ${token}`,
        "X-Company-Id": companyId || ""
      },
      heartbeatTimeout: 300000
    });

    const handleUpdate = (status: string) => {
       setLiveStatus(status);
       if (status === 'COMPLETED' || status === 'FAILED') {
         fetchRealData();
         es.close();
       }
    };

    es.addEventListener('meeting.queued', () => handleUpdate('QUEUED'));

    (es as unknown as EventTarget).addEventListener('meeting.progress', (e: Event) => {
      handleUpdate('PROCESSING');
      try {
        const messageEvent = e as MessageEvent;
        const data = JSON.parse(messageEvent.data);
        if (data.progress) setProgressPercent(data.progress);
      } catch (err) {
        console.error("Failed to parse progress data", err);
      }
    });

    es.addEventListener('meeting.completed', () => handleUpdate('COMPLETED'));
    es.addEventListener('meeting.failed', () => handleUpdate('FAILED'));
    es.addEventListener('meeting.cancelled', () => handleUpdate('CANCELLED'));

    return () => {
       es.close();
    };
  }, [meetingId, meeting?.status, meeting?.processing_status, fetchRealData]);

  const toggleTaskStatus = async (task: RealTask) => {
    const newStatus = (task.status === "DONE" || task.status === "COMPLETED") ? "TODO" : "DONE";
    setUpdatingTaskId(task.id);
    try {
      await apiFetch<void>(`/tasks/${task.id}`, {
        method: "PATCH",
        data: { 
          task_text: task.task_text, 
          meeting_id: task.meeting_id ? Number(task.meeting_id) : null, 
          due_date: task.due_date, 
          status: newStatus 
        }
      });
      setTasks(currentTasks => currentTasks.map(t => t.id === task.id ? { ...t, status: newStatus } : t));
    } catch (err) {
      console.error(err);
    } finally {
      setUpdatingTaskId(null);
    }
  };

  if (isLoading) {
    return <div className="h-screen flex items-center justify-center bg-slate-50"><Loader2 className="w-12 h-12 text-brand-maroon animate-spin" /></div>;
  }

  const activeMainVideoUrl = meeting?.main_video_url || FALLBACK_MAIN_VIDEO;

  const dbStatus = meeting?.processing_status || meeting?.status || 'UPLOADED';
  const currentDisplayStatus = (liveStatus || dbStatus).toUpperCase();
  
  const isProcessing = ['UPLOADED', 'QUEUED', 'PROCESSING'].includes(currentDisplayStatus);
  const isFailed = currentDisplayStatus === 'FAILED';
  const isCancelled = currentDisplayStatus === 'CANCELLED';

  const formatTime = (seconds?: number) => {
    if (seconds === undefined) return "0:00";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const renderSummary = () => {
    let s = meeting?.summary;
    if (!s) return [];

    // Safety fallback for deeply nested JSON structures from FastAPI
    if (typeof s === 'object' && s !== null && 'summary' in s) {
      s = (s as { summary: { text: string } }).summary;
    }

    if (Array.isArray(s)) return s;
    if (typeof s === 'string') return s.split('\n').filter(Boolean);

    const sObj = s as Record<string, unknown>;
    if (sObj && typeof sObj.text === 'string') {
      // Split by newlines so it renders as neat list items if the AI returns paragraphs
      return sObj.text.split('\n').filter(Boolean);
    }

    return [];
  };

  const renderTranscript = () => {
    const rawTranscript = meeting?.transcript || meeting?.transcription;
    if (!rawTranscript) return []; 

    // Handle single string block by splitting by newlines
    if (typeof rawTranscript === 'string') {
      return rawTranscript.split('\n').filter(Boolean).map(lineText => ({
        text: lineText.trim(),
        speaker: "Speaker",
        displaySpeaker: "Speaker",
        displayTime: "-",
        start: undefined
      }));
    }
    
    if (!Array.isArray(rawTranscript)) return [];

    const mappings = meeting?.name_recognition?.mappings || [];

    return rawTranscript.map(line => {
      const match = mappings.find(m => m.speaker_id === line.speaker);
      return {
        ...line,
        displaySpeaker: match?.predicted_name || line.speaker,
        displayTime: line.time || formatTime(line.start)
      };
    });
  };

  const renderSignTranscript = () => {
    if (meeting?.sign_transcript && Array.isArray(meeting.sign_transcript)) return meeting.sign_transcript;
    return []; 
  };

  return (
    <div className="h-screen flex flex-col bg-slate-50 overflow-hidden">
      
      <header className="h-16 bg-white border-b border-slate-200 flex items-center px-6 shrink-0 z-10">
        <button onClick={() => router.push('/dashboard')} className="p-2 text-slate-400 hover:text-brand-maroon hover:bg-brand-maroon/5 rounded-lg transition-colors mr-4">
          <ArrowLeft size={20} />
        </button>
        <div className="h-6 w-px bg-slate-200 mr-4"></div>
        <div>
          <h1 className="font-bold text-slate-900 leading-tight">{meeting?.title || "Unknown Meeting"}</h1>
          <p className="text-xs text-slate-500 uppercase tracking-wide font-medium">
            {meeting?.created_at ? new Date(meeting.created_at).toLocaleDateString() : "No date"} • <span className={`${
                currentDisplayStatus === 'COMPLETED' ? 'text-green-600' : 
                (currentDisplayStatus === 'PROCESSING' || currentDisplayStatus === 'QUEUED') ? 'text-yellow-600' : 
                currentDisplayStatus === 'FAILED' ? 'text-red-600' : 'text-slate-500' 
              }`}>{currentDisplayStatus}</span>
          </p>
        </div>
      </header>

      <main className="flex-1 flex overflow-hidden">
        
        <section className="w-2/3 flex flex-col border-r border-slate-200 bg-black relative">
          
          <div ref={playerContainerRef} className="h-[60%] bg-black relative group flex items-center justify-center overflow-hidden">
            
            {/* SINGLE MAIN VIDEO PLAYER */}
            <video 
              ref={mainVideoRef}
              src={activeMainVideoUrl} 
              muted={isMuted}
              playsInline
              preload="auto"
              className="w-full h-full object-contain"
              onLoadedMetadata={handleLoadedMetadata} 
              onTimeUpdate={handleTimeUpdate}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
            />

            {/* CONTROL BAR */}
            <div className="absolute bottom-0 inset-x-0 h-20 bg-linear-to-t from-black/90 via-black/40 to-transparent flex flex-col justify-end px-6 pb-4 opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-auto">
              
              <div 
                className="w-full h-1.5 bg-white/30 rounded-full mb-3 overflow-hidden cursor-pointer" 
                onClick={handleProgressBarClick}
              >
                <div className="h-full bg-brand-maroon transition-all duration-75 ease-out" style={{ width: `${playedProgress}%` }}></div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <button onClick={togglePlay} className="text-white hover:text-brand-gold transition-colors focus:outline-none relative z-50 cursor-pointer">
                    {isPlaying ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" />}
                  </button>
                  <button onClick={() => setIsMuted(!isMuted)} className="text-white hover:text-brand-gold transition-colors focus:outline-none relative z-50 cursor-pointer">
                    {isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
                  </button>
                </div>
                <button onClick={toggleFullScreen} className="text-white hover:text-brand-gold transition-colors focus:outline-none relative z-50 cursor-pointer">
                  {isFullscreen ? <Minimize size={20} /> : <Maximize size={20} />}
                </button>
              </div>
            </div>
          </div>

          <div className="h-[40%] flex flex-col bg-slate-50 border-t border-slate-200">
            <div className="flex border-b border-slate-200 bg-white shrink-0">
              <button onClick={() => setLeftTab('SUMMARY')} className={`flex-1 py-3 text-sm font-bold flex items-center justify-center gap-2 border-b-2 transition-colors ${leftTab === 'SUMMARY' ? 'border-brand-maroon text-brand-maroon' : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50'}`}>
                <FileText size={16} /> AI Summary
              </button>
              <button onClick={() => setLeftTab('TASKS')} className={`flex-1 py-3 text-sm font-bold flex items-center justify-center gap-2 border-b-2 transition-colors ${leftTab === 'TASKS' ? 'border-brand-maroon text-brand-maroon' : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50'}`}>
                <CheckSquare size={16} /> Extracted Tasks
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6">
              {isProcessing ? (
                <div className="flex flex-col items-center justify-center h-full text-slate-500 space-y-3">
                  <Loader2 className="w-8 h-8 animate-spin text-brand-maroon/50" />
                  <p className="font-medium text-sm">
                    {currentDisplayStatus === 'QUEUED' 
                      ? 'Waiting in queue...' 
                      : `AI is analyzing this meeting... ${progressPercent}%`}
                  </p>
                </div>
              ) : isFailed ? (
                <div className="flex flex-col items-center justify-center h-full text-center px-6 space-y-3">
                  <div className="w-16 h-16 bg-red-50 rounded-full flex items-center justify-center mb-2">
                    <AlertTriangle className="w-8 h-8 text-red-500" />
                  </div>
                  <h3 className="text-lg font-bold text-slate-900">Processing Failed</h3>
                  <p className="text-sm font-medium text-slate-500 max-w-sm">
                    The AI encountered an error analyzing this meeting. The temporary media files have been cleared from the server.
                  </p>

                  
                </div>
                
              ) : isCancelled ? (
                <div className="flex flex-col items-center justify-center h-full text-slate-500 space-y-3">
                  <X className="w-10 h-10" />
                  <p className="font-medium text-sm">This processing job was cancelled.</p>
                </div>
              ) : leftTab === 'SUMMARY' ? (
                <ul className="space-y-3">
                  {renderSummary().length === 0 ? (
                    <p className="text-sm text-slate-500 text-center mt-4">No summary available.</p>
                  ) : (
                    renderSummary().map((point: string, idx: number) => (
                      <li key={idx} className="flex gap-3 text-slate-700 text-sm">
                        <div className="w-1.5 h-1.5 rounded-full bg-brand-gold mt-1.5 shrink-0"></div>
                        <p>{point}</p>
                      </li>
                    ))
                  )}
                </ul>
              ) : (
                <div className="space-y-3">
                  {tasks.length === 0 ? (
                    <p className="text-sm text-slate-500 text-center mt-4">No tasks extracted for this meeting yet.</p>
                  ) : (
                    tasks.map(task => (
                      <div key={task.id} className="flex items-center justify-between p-3 bg-white rounded-lg border border-slate-200 shadow-sm">
                        <div className="flex items-center gap-3">
                          {updatingTaskId === task.id ? (
                            <Loader2 className="w-4 h-4 text-brand-maroon animate-spin" />
                          ) : (
                            <input 
                              type="checkbox" 
                              checked={task.status === 'DONE' || task.status === 'COMPLETED'} 
                              onChange={() => toggleTaskStatus(task)}
                              className="w-4 h-4 text-brand-maroon rounded border-slate-300 focus:ring-brand-maroon cursor-pointer transition-colors" 
                            />
                          )}
                          <span className={`text-sm font-medium ${(task.status === 'DONE' || task.status === 'COMPLETED') ? 'line-through text-slate-400' : 'text-slate-700'}`}>{task.task_text}</span>
                        </div>
                        {task.due_date && <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded font-medium">Due: {new Date(task.due_date).toLocaleDateString()}</span>}
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="w-1/3 flex flex-col bg-white">
          <div className="flex border-b border-slate-200 shrink-0">
            <button onClick={() => setRightTab('AUDIO')} className={`flex-1 py-4 text-sm font-bold flex items-center justify-center gap-2 border-b-2 transition-colors ${rightTab === 'AUDIO' ? 'border-brand-maroon text-brand-maroon bg-brand-maroon/5' : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50'}`}>
              <MessageSquare size={16} /> Audio Transcript
            </button>
            <button onClick={() => setRightTab('SIGN')} className={`flex-1 py-4 text-sm font-bold flex items-center justify-center gap-2 border-b-2 transition-colors ${rightTab === 'SIGN' ? 'border-brand-gold text-yellow-700 bg-brand-gold/10' : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50'}`}>
               Sign Translation
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {isProcessing ? (
               <div className="flex flex-col items-center justify-center h-full text-slate-500 space-y-3">
                 <Loader2 className="w-8 h-8 animate-spin text-brand-maroon/50" />
                 <p className="font-medium text-sm">Generating transcript... {progressPercent}%</p>
               </div>
            ) : isFailed ? (
               <div className="flex flex-col items-center justify-center h-full text-center px-6 space-y-3">
                 <AlertTriangle className="w-10 h-10 text-slate-300" />
                 <p className="font-medium text-sm text-slate-500">Transcript generation failed.</p>
               </div>
            ) : isCancelled ? (
               <div className="flex flex-col items-center justify-center h-full text-slate-500 space-y-3">
                 <X className="w-10 h-10" />
                 <p className="font-medium text-sm">Processing was cancelled.</p>
               </div>
            ) : rightTab === 'AUDIO' ? (
              renderTranscript().length === 0 ? (
                 <p className="text-sm text-slate-500 text-center mt-4">No transcript available.</p>
              ) : (
                renderTranscript().map((line: TranscriptLine & { displayTime: string; displaySpeaker: string }, idx: number) => (
                  <div 
                    key={idx} 
                    onClick={() => handleTranscriptClick(line.start)}
                    className="flex gap-4 group cursor-pointer hover:bg-slate-50 p-2 -mx-2 rounded-lg transition-colors"
                  >
                    <div className="flex flex-col items-center gap-1 shrink-0 mt-0.5">
                      <span className="text-xs font-mono font-medium text-brand-maroon bg-brand-maroon/10 px-1.5 py-0.5 rounded group-hover:bg-brand-maroon group-hover:text-white transition-colors">{line.displayTime}</span>
                      <Clock size={12} className="text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                    <div>
                      <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">{line.displaySpeaker}</span>
                      <p className="text-sm text-slate-700 mt-0.5 leading-relaxed group-hover:text-slate-900 transition-colors">
                        {typeof line === 'string' ? line : line.text}
                      </p>
                    </div>
                  </div>
                ))
              )
            ) : (
              renderSignTranscript().length === 0 ? (
                 <p className="text-sm text-slate-500 text-center mt-4">No sign language translation available yet.</p>
              ) : (
                renderSignTranscript().map((line: TranscriptLine, idx: number) => (
                  <div 
                    key={idx} 
                    onClick={() => handleTranscriptClick(line.start)}
                    className="flex gap-4 group cursor-pointer hover:bg-slate-50 p-2 -mx-2 rounded-lg transition-colors"
                  >
                    <div className="flex flex-col items-center gap-1 shrink-0 mt-0.5">
                      <span className="text-xs font-mono font-bold text-yellow-700 bg-brand-gold/10 border border-brand-gold/20 px-1.5 py-0.5 rounded group-hover:bg-yellow-700 group-hover:text-white transition-colors">{line.time || "0:00"}</span>
                      <Clock size={12} className="text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                    <div>
                      <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">{line.speaker || "Signer"}</span>
                      <p className="text-sm text-slate-700 mt-0.5 leading-relaxed group-hover:text-slate-900 transition-colors">
                        {typeof line === 'string' ? line : line.text}
                      </p>
                    </div>
                  </div>
                ))
              )
            )}
          </div>
        </section>

      </main>
    </div>
  );
}