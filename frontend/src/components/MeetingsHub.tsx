"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { 
  Calendar, Play, MoreVertical, Loader2, Trash2, 
  AlertTriangle, Edit2, Check, X, ArrowUpDown, Search, 
  Users, Shield 
} from "lucide-react";
import { EventSourcePolyfill } from "event-source-polyfill";
import { apiFetch } from "@/lib/api";

interface Participant {
  id?: number | string;
  user_id?: number | string;
  userId?: number | string;
  email?: string;
  user?: {
    id?: number | string;
    email?: string;
    username?: string;
  };
}

interface Meeting {
  id: string | number;
  title: string;
  created_at?: string;
  summary?: string;
  status?: string;
  processing_status?: string;
  participants?: Participant[];
  meetingParticipants?: Participant[]; 
  MeetingParticipants?: Participant[]; 
}

interface Member {
  id: number | string;
  username?: string;
  email?: string;
  role?: string;
}

interface CurrentUser {
  id?: string | number;
  email?: string;
  username?: string;
}

interface AuthResponse {
  activeRole?: string;
  user?: CurrentUser;
}

type UsersResponse = Member[] | { data?: Member[] };
type MeetingResponse = Meeting | { data?: Meeting };

interface Props {
  meetings: Meeting[];
  isLoading: boolean;
  onMeetingUpdated: () => void; 
}

type SortOrder = 'NEWEST' | 'OLDEST';

export default function MeetingsHub({ meetings, isLoading, onMeetingUpdated }: Props) {
  const localUsername = typeof window !== "undefined" ? localStorage.getItem("username") : null;

  // States
  const [openMenuId, setOpenMenuId] = useState<string | number | null>(null);
  const [meetingToDelete, setMeetingToDelete] = useState<Meeting | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [editingMeetingId, setEditingMeetingId] = useState<string | number | null>(null);
  const [editTitleValue, setEditTitleValue] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortOrder, setSortOrder] = useState<SortOrder>('NEWEST');
  const [accessModalMeeting, setAccessModalMeeting] = useState<Meeting | null>(null);
  const [companyMembers, setCompanyMembers] = useState<Member[]>([]);
  const [isLoadingMembers, setIsLoadingMembers] = useState(false);
  const [meetingParticipants, setMeetingParticipants] = useState<Set<string>>(new Set());
  const [togglingUserId, setTogglingUserId] = useState<string | number | null>(null);
  const [activeRole, setActiveRole] = useState<string | null>(null);
  const [currentUserData, setCurrentUserData] = useState<CurrentUser | null>(null);
  const [liveStatuses, setLiveStatuses] = useState<Record<string, string>>({});

  useEffect(() => {
    const fetchAuth = async () => {
      try {
        const data = await apiFetch<AuthResponse>("/auth/me");
        setActiveRole(data.activeRole ? data.activeRole.toUpperCase() : "MEMBER");
        if (data.user) setCurrentUserData(data.user);
      } catch (err) {
        console.error("Failed to fetch auth", err);
      }
    };
    fetchAuth();
  }, []);

  useEffect(() => {
    const activeMeetings = meetings.filter(m => {
      const s = (m.processing_status || m.status || 'UPLOADED').toUpperCase();
      return ['UPLOADED', 'QUEUED', 'PROCESSING'].includes(s);
    });
    
    if (activeMeetings.length === 0) return;

    const token = localStorage.getItem("token");
    const companyId = localStorage.getItem("companyId");
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3000/api";
    const eventSources: EventSource[] = [];

    activeMeetings.forEach(meeting => {
      const url = `${API_URL}/meetings/${meeting.id}/events`;

      const es = new EventSourcePolyfill(url, {
        headers: {
          "Authorization": `Bearer ${token}`,
          "X-Company-Id": companyId || ""
        },
        heartbeatTimeout: 300000
      });

      const handleUpdate = (status: string) => {
        setLiveStatuses(prev => ({ ...prev, [meeting.id]: status }));
        if (status === 'COMPLETED' || status === 'FAILED' || status === 'CANCELLED') {
          onMeetingUpdated(); 
          es.close();
        }
      };

      const target = es as unknown as EventTarget;
      target.addEventListener('meeting.queued', () => handleUpdate('QUEUED'));
      target.addEventListener('meeting.progress', () => handleUpdate('PROCESSING'));
      target.addEventListener('meeting.completed', () => handleUpdate('COMPLETED'));
      target.addEventListener('meeting.failed', () => handleUpdate('FAILED'));
      target.addEventListener('meeting.cancelled', () => handleUpdate('CANCELLED'));

      eventSources.push(es);
    });

    return () => {
      eventSources.forEach(es => es.close());
    };
  }, [meetings, onMeetingUpdated]);

  const isAdminOrOwner = activeRole === 'ADMIN' || activeRole === 'OWNER';

  const toggleMenu = (id: string | number) => {
    setOpenMenuId(openMenuId === id ? null : id);
  };

  const executeDelete = async () => {
    if (!meetingToDelete) return;
    setIsDeleting(true);

    try {
      await apiFetch<void>(`/meetings/${meetingToDelete.id}`, { method: "DELETE" });
      onMeetingUpdated();
      setMeetingToDelete(null);
    } catch (err) {
      console.error(err);
      alert("Failed to delete meeting.");
    } finally {
      setIsDeleting(false);
    }
  };

  const executeEdit = async (meetingId: string | number) => {
    if (!editTitleValue.trim()) return;
    setIsEditing(true);

    try {
      await apiFetch<Meeting>(`/meetings/${meetingId}`, {
        method: "PATCH",
        data: { title: editTitleValue.trim() }
      });
      onMeetingUpdated();
      setEditingMeetingId(null);
    } catch (err) {
      console.error(err);
      alert(`Title Edit Failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setIsEditing(false);
    }
  };

  const openAccessModal = async (meeting: Meeting) => {
    setAccessModalMeeting(meeting);
    setOpenMenuId(null);
    setIsLoadingMembers(true);

    try {
      const usersData = await apiFetch<UsersResponse>("/users");
      const usersList = Array.isArray(usersData) ? usersData : usersData.data || [];
      setCompanyMembers(usersList);

      const mData = await apiFetch<MeetingResponse>(`/meetings/${meeting.id}`);
      const actualMeetingData = ('data' in mData && mData.data) ? mData.data : mData as Meeting;
        
      const pIds = new Set<string>();
      // Removed "any" cast by referencing strictly typed interfaces
      const participantsArray = actualMeetingData.participants || actualMeetingData.meetingParticipants || actualMeetingData.MeetingParticipants || [];
        
      if (Array.isArray(participantsArray)) {
        participantsArray.forEach((p: Participant) => {
          if (p.email) pIds.add(p.email.toLowerCase());
          if (p.user?.email) pIds.add(p.user.email.toLowerCase());
          const validId = p.user_id || p.userId || p.id;
          if (validId) pIds.add(String(validId));
        });
      }
      setMeetingParticipants(pIds);
    } catch (err) {
      console.error("Failed to load access data", err);
    } finally {
      setIsLoadingMembers(false);
    }
  };

  const toggleParticipantAccess = async (member: Member, isCurrentlyParticipant: boolean) => {
    if (!accessModalMeeting) return;
    if (!member.email) {
      alert("Cannot assign this user because they do not have a valid email address.");
      return;
    }
    
    setTogglingUserId(member.id);

    try {
      if (isCurrentlyParticipant) {
        await apiFetch<void>(`/meetings/${accessModalMeeting.id}/participants/${member.id}`, { 
          method: "DELETE",
          data: { email: member.email }
        });
        setMeetingParticipants(prev => {
          const next = new Set(prev);
          next.delete(String(member.id));
          if (member.email) next.delete(member.email.toLowerCase());
          return next;
        });
      } else {
        await apiFetch<void>(`/meetings/${accessModalMeeting.id}/participants`, {
          method: "POST",
          data: { email: member.email } 
        });
        setMeetingParticipants(prev => {
          const next = new Set(prev);
          next.add(String(member.id));
          if (member.email) next.add(member.email.toLowerCase());
          return next;
        });
      }
    } catch (err) {
      console.error(err);
      alert("Failed to update access.");
    } finally {
      setTogglingUserId(null);
    }
  };

  if (isLoading) {
    return (
      <div className="flex-1 h-full flex flex-col items-center justify-center min-h-100">
        <Loader2 className="w-10 h-10 text-brand-maroon animate-spin" />
        <p className="mt-4 text-slate-500 font-medium">Loading your meetings...</p>
      </div>
    );
  }

  const processedMeetings = meetings
    .filter(meeting => {
      if (!meeting.title.toLowerCase().includes(searchQuery.toLowerCase())) return false;

      if (!isAdminOrOwner && currentUserData) {
        // Removed "any" cast by referencing strictly typed interfaces
        const participantsArray = meeting.participants || meeting.meetingParticipants || meeting.MeetingParticipants || [];
        const isParticipant = participantsArray.some((p: Participant) =>
          String(p.user_id) === String(currentUserData.id) ||
          String(p.userId) === String(currentUserData.id) ||
          String(p.id) === String(currentUserData.id) ||
          (p.email && p.email.toLowerCase() === currentUserData.email?.toLowerCase()) ||
          (p.user?.email && p.user.email.toLowerCase() === currentUserData.email?.toLowerCase())
        );
        if (!isParticipant) return false;
      }

      return true;
    })
    .sort((a, b) => {
      const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
      const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
      return sortOrder === 'NEWEST' ? dateB - dateA : dateA - dateB;
    });

  return (
    <div className="space-y-6 relative">
      
      {/* HEADER WITH SEARCH & SORT */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
        <div>
          <h2 className="text-2xl font-serif font-bold text-slate-900">All Meetings</h2>
          <p className="text-slate-500 mt-1">Browse and search through your {meetings.length} recorded sessions.</p>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-3 w-full md:w-auto">
          <div className="relative w-full md:w-64">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-4 w-4 text-slate-400" />
            </div>
            <input 
              type="text" 
              placeholder="Search meetings..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold text-sm transition-colors" 
            />
          </div>
          
          <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-xl px-4 py-2.5 shadow-sm">
            <ArrowUpDown className="w-4 h-4 text-slate-400" />
            <select 
              value={sortOrder} 
              onChange={(e) => setSortOrder(e.target.value as SortOrder)} 
              className="bg-transparent text-sm font-medium text-slate-700 focus:outline-none cursor-pointer"
            >
              <option value="NEWEST">Date: Newest First</option>
              <option value="OLDEST">Date: Oldest First</option>
            </select>
          </div>
        </div>
      </div>

      {/* MEETINGS GRID */}
      {processedMeetings.length === 0 ? (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-12 text-center">
          <p className="text-slate-500">No meetings match your search query &quot;{searchQuery}&quot;.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {processedMeetings.map((meeting) => {
            
            const dbStatus = meeting.processing_status || meeting.status || 'UPLOADED';
            const currentStatus = (liveStatuses[meeting.id] || dbStatus).toUpperCase();

            return (
            <div key={meeting.id} className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-visible hover:shadow-md hover:border-brand-gold/50 transition-all group flex flex-col relative">
              
              {/* CLICKABLE THUMBNAIL */}
              <Link href={`/dashboard/meeting/${meeting.id}`} className="aspect-video bg-slate-100 relative flex items-center justify-center overflow-hidden rounded-t-2xl group/thumb">
                <div className="absolute top-3 left-3 z-10 flex items-center gap-2">
                  <span className={`px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider rounded-md backdrop-blur-md shadow-sm border ${
                    currentStatus === 'COMPLETED' ? 'bg-green-500/90 text-white border-green-600/20' : 
                    (currentStatus === 'PROCESSING' || currentStatus === 'QUEUED') ? 'bg-brand-gold/90 text-yellow-950 border-yellow-600/20' : 
                    currentStatus === 'FAILED' ? 'bg-red-500/90 text-white border-red-600/20' :
                    'bg-slate-800/80 text-white border-white/10'
                  }`}>
                    {currentStatus === 'PROCESSING' && <Loader2 size={10} className="inline mr-1 animate-spin" />}
                    {currentStatus}
                  </span>
                </div>

                <div className="absolute inset-0 bg-linear-to-tr from-brand-maroon/20 to-brand-gold/20 opacity-50"></div>
                <div className="w-12 h-12 bg-white/90 backdrop-blur-sm rounded-full flex items-center justify-center shadow-lg transform group-hover/thumb:scale-110 transition-transform cursor-pointer">
                  <Play className="w-5 h-5 text-brand-maroon ml-1" />
                </div>
              </Link>

              <div className="p-5 flex flex-col grow relative">
                <div className="flex items-start justify-between mb-3 min-h-12">
                  
                  {/* Inline Editing UI vs Normal Title */}
                  {editingMeetingId === meeting.id ? (
                    <div className="flex-1 mr-2">
                      <input 
                        autoFocus
                        value={editTitleValue}
                        onChange={(e) => setEditTitleValue(e.target.value)}
                        disabled={isEditing}
                        className="w-full text-sm font-bold text-slate-900 border-2 border-brand-gold/50 rounded-lg px-2 py-1 focus:outline-none focus:border-brand-gold bg-slate-50 transition-colors disabled:opacity-50"
                      />
                      <div className="flex items-center gap-2 mt-2">
                        <button onClick={() => executeEdit(meeting.id)} disabled={isEditing || !editTitleValue.trim()} className="bg-brand-maroon text-white p-1 rounded hover:bg-brand-gold transition-colors disabled:opacity-50">
                          {isEditing ? <Loader2 size={14} className="animate-spin" /> : <Check size={14} />}
                        </button>
                        <button onClick={() => setEditingMeetingId(null)} disabled={isEditing} className="bg-slate-200 text-slate-600 p-1 rounded hover:bg-slate-300 transition-colors disabled:opacity-50">
                          <X size={14} />
                        </button>
                      </div>
                    </div>
                  ) : (
                    <Link href={`/dashboard/meeting/${meeting.id}`} className="font-bold text-slate-900 line-clamp-2 leading-tight hover:text-brand-maroon transition-colors cursor-pointer pr-4">
                      {meeting.title}
                    </Link>
                  )}
                  
                  {/* 3-Dot Menu Button (Secured for Admins only) */}
                  {editingMeetingId !== meeting.id && isAdminOrOwner && (
                    <div className="relative">
                      <button onClick={() => toggleMenu(meeting.id)} className="text-slate-400 hover:text-slate-600 hover:bg-slate-50 transition-colors p-1.5 rounded-lg -mr-2 -mt-1">
                        <MoreVertical size={18} />
                      </button>

                      {/* Dropdown Menu (Secured for Admins!) */}
                      {openMenuId === meeting.id && (
                        <div className="absolute right-0 mt-1 w-40 bg-white rounded-xl shadow-lg border border-slate-100 overflow-hidden z-20 animate-in fade-in slide-in-from-top-2">
                          <button onClick={() => { setEditingMeetingId(meeting.id); setEditTitleValue(meeting.title); setOpenMenuId(null); }} className="w-full text-left px-4 py-2.5 text-sm text-slate-600 hover:bg-slate-50 hover:text-brand-maroon flex items-center gap-2 transition-colors">
                            <Edit2 size={14} /> Edit Title
                          </button>
                          
                          {isAdminOrOwner && (
                            <>
                              <button onClick={() => openAccessModal(meeting)} className="w-full text-left px-4 py-2.5 text-sm text-slate-600 hover:bg-slate-50 hover:text-brand-maroon flex items-center gap-2 transition-colors border-t border-slate-50">
                                <Users size={14} /> Manage Access
                              </button>

                              <button onClick={() => { setMeetingToDelete(meeting); setOpenMenuId(null); }} className="w-full text-left px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 flex items-center gap-2 transition-colors border-t border-slate-50">
                                <Trash2 size={14} /> Delete
                              </button>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                <div className="flex items-center gap-2 text-xs text-slate-500 mb-4 mt-auto">
                  <Calendar size={14} />
                  <span>{meeting.created_at ? new Date(meeting.created_at).toLocaleDateString() : "Recently uploaded"}</span>
                </div>
                
                <div className="pt-4 border-t border-slate-100 mt-auto">
                  <Link href={`/dashboard/meeting/${meeting.id}`} className="w-full text-sm font-medium text-brand-maroon hover:text-brand-gold transition-colors flex items-center justify-center gap-2">
                    View Details
                  </Link>
                </div>
              </div>

            </div>
          )})}
        </div>
      )}

      {/* MANAGE ACCESS MODAL */}
      {accessModalMeeting && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white max-w-lg w-full rounded-2xl shadow-xl overflow-hidden flex flex-col max-h-[85vh] animate-in zoom-in-95 duration-200">
            
            <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between shrink-0">
              <div>
                <h3 className="font-serif text-xl font-bold text-slate-900">Manage Access</h3>
                <p className="text-sm text-slate-500 mt-1 truncate max-w-75">&quot;{accessModalMeeting.title}&quot;</p>
              </div>
              <button onClick={() => setAccessModalMeeting(null)} className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-full transition-colors"><X size={20} /></button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 bg-slate-50/50">
              {isLoadingMembers ? (
                <div className="flex flex-col items-center justify-center py-12 text-slate-500">
                  <Loader2 className="w-8 h-8 text-brand-maroon animate-spin mb-4" />
                  <p>Loading team members...</p>
                </div>
              ) : companyMembers.length === 0 ? (
                <div className="text-center py-8 text-slate-500">No members found in this workspace.</div>
              ) : (
                <div className="space-y-3">
                  <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-1.5"><Shield size={14} /> Workspace Members</p>
                  
                  {companyMembers.map(member => {
                    const hasAccess = meetingParticipants.has(String(member.id)) || (member.email && meetingParticipants.has(member.email.toLowerCase()));
                    const isToggling = togglingUserId === member.id;
                    
                    const isMe = 
                      (currentUserData && member.email?.toLowerCase() === currentUserData.email?.toLowerCase()) ||
                      member.username?.toLowerCase() === localUsername?.toLowerCase() || 
                      member.email?.toLowerCase() === localUsername?.toLowerCase();
                    
                    // SMART ROLE CHECK: The backend now provides the role for every user!
                    const memberRole = String(member.role || 'MEMBER').toUpperCase();
                    const isWorkspaceAdmin = memberRole === 'ADMIN' || memberRole === 'OWNER';
                    
                    return (
                      <div key={member.id} className="flex items-center justify-between p-3 bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-md hover:border-slate-300 transition-all">
                        <div className="flex items-center gap-3">
                          <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm uppercase ${hasAccess || isWorkspaceAdmin ? 'bg-brand-maroon/10 text-brand-maroon' : 'bg-slate-100 text-slate-400'}`}>
                            {(member.username || member.email || "U")[0]}
                          </div>
                          <div>
                            <p className="font-medium text-slate-900 text-sm leading-none">
                              {member.username || member.email || "Unknown User"} 
                              {isMe && <span className="text-slate-400 font-normal ml-1">(You)</span>}
                            </p>
                            <p className="text-xs text-slate-500 mt-1 flex items-center gap-1.5">
                              {member.email}
                              {(hasAccess || isWorkspaceAdmin) && <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>}
                            </p>
                          </div>
                        </div>

                        {isWorkspaceAdmin ? (
                          <span className="px-3 py-1.5 rounded-lg text-xs font-bold text-yellow-700 bg-brand-gold/10 flex items-center gap-1.5 border border-brand-gold/20">
                            <Shield size={14} /> Full Access
                          </span>
                        ) : (
                          <button
                            onClick={() => toggleParticipantAccess(member, !!hasAccess)}
                            disabled={isToggling || !member.email}
                            className={`px-4 py-2 rounded-lg text-xs font-bold transition-all disabled:opacity-50 flex items-center gap-2 ${
                              hasAccess 
                                ? 'bg-slate-100 text-slate-600 hover:bg-red-50 hover:text-red-600' 
                                : 'bg-brand-maroon text-white hover:bg-brand-gold'
                            }`}
                          >
                            {isToggling ? <Loader2 size={14} className="animate-spin" /> : hasAccess ? "Remove" : "Grant Access"}
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="px-6 py-4 border-t border-slate-100 flex justify-end shrink-0 bg-white">
               <button onClick={() => setAccessModalMeeting(null)} className="px-6 py-2.5 bg-slate-900 text-white text-sm font-medium rounded-lg hover:bg-slate-800 transition-colors">Done</button>
            </div>
          </div>
        </div>
      )}

      {/* DELETE CONFIRMATION MODAL */}
      {meetingToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white max-w-md w-full rounded-2xl shadow-xl overflow-hidden animate-in zoom-in-95 duration-200">
            <div className="p-6">
              <div className="w-12 h-12 rounded-full bg-red-100 text-red-600 flex items-center justify-center mb-4">
                <AlertTriangle size={24} />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-2">Delete Meeting?</h3>
              <p className="text-slate-500 mb-6">
                Are you sure you want to delete <span className="font-semibold text-slate-700">&quot;{meetingToDelete.title}&quot;</span>? This will also permanently delete all associated transcripts, summaries, and action items.
              </p>
              <div className="flex gap-3 w-full">
                <button 
                  onClick={() => setMeetingToDelete(null)}
                  disabled={isDeleting}
                  className="flex-1 px-4 py-2.5 border border-slate-200 text-slate-700 font-medium rounded-xl hover:bg-slate-50 transition-colors"
                >
                  Cancel
                </button>
                <button 
                  onClick={executeDelete}
                  disabled={isDeleting}
                  className="flex-1 px-4 py-2.5 bg-red-600 text-white font-medium rounded-xl hover:bg-red-700 transition-colors flex items-center justify-center gap-2"
                >
                  {isDeleting ? <Loader2 size={18} className="animate-spin" /> : "Delete Meeting"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}