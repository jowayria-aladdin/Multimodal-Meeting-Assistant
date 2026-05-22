"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import { 
  Loader2, Plus, Users, Building, 
  Home, Video, CheckSquare, Settings as SettingsIcon, 
  LogOut, Upload,ArrowLeft
} from "lucide-react";

import { apiFetch } from "@/lib/api"; 
import DashboardHome from "@/components/DashboardHome";
import MeetingsHub from "@/components/MeetingsHub";
import ActionItems from "@/components/ActionItems";
import TeamSettings from "@/components/TeamSettings";
import UploadModal from "@/components/UploadModal";
interface ApiResponse<T> {
  data?: T;
}

interface Meeting { id: string | number; title: string; created_at?: string; summary?: string; status?: string; }
interface Task { id: string | number; task_text: string; status?: string; due_date?: string; meeting_id?: string | number;}
interface CurrentUser { id?: string | number; email?: string; username?: string; }

interface AuthMeResponse {
  activeRole?: string;
  user?: CurrentUser;
}

interface Company {
  id: string | number;
  name?: string;
}

interface CompanyResponse {
  data?: { name?: string };
  name?: string;
}

export default function Dashboard() {
  const router = useRouter();
  
  const [isChecking, setIsChecking] = useState(true);
  const [needsWorkspace, setNeedsWorkspace] = useState(false);
  const [token, setToken] = useState<string | null>(null);

  const [workspaceName, setWorkspaceName] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState("");
  
  const [username, setUsername] = useState<string>("");
  const [currentWorkspaceName, setCurrentWorkspaceName] = useState<string>(""); 
  
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("dashboard");

  // Dashboard Data
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [memberCount, setMemberCount] = useState<number>(0); 
  const [isLoadingData, setIsLoadingData] = useState(false);
  
  // RBAC State
  const [activeRole, setActiveRole] = useState<string | null>(null);
  const [currentUserData, setCurrentUserData] = useState<CurrentUser | null>(null);

  useEffect(() => {
    const initializeApp = async () => {
      const storedToken = localStorage.getItem("token");
      const storedCompanyId = localStorage.getItem("companyId");
      const storedUsername = localStorage.getItem("username");
      
      if (storedUsername) setUsername(storedUsername);
      
      if (!storedToken) {
        router.push("/signin");
        return;
      }

      setToken(storedToken);

      if (!storedCompanyId || ["null", "undefined", ""].includes(storedCompanyId)) {
        try {
          const companies = await apiFetch<Company[] | ApiResponse<Company[]>>("/companies");
          const companiesList = Array.isArray(companies) ? companies : companies.data || [];
          
          if (companiesList.length > 0) {
            const recoveredId = companiesList[0].id;
            localStorage.setItem("companyId", String(recoveredId));
            setNeedsWorkspace(false);
          } else {
            setNeedsWorkspace(true);
          }
        } catch (err) {
          console.error("Auto-recovery failed:", err);
          setNeedsWorkspace(true);
        }
      } 
      
      setIsChecking(false);
    };

    initializeApp();
  }, [router]);

  const fetchDashboardData = useCallback(async () => {
    setIsLoadingData(true);
    const companyId = localStorage.getItem("companyId");

    try {
      const authData = await apiFetch<AuthMeResponse>("/auth/me");
      setActiveRole(authData.activeRole ? authData.activeRole.toUpperCase() : "MEMBER");
      if (authData.user) setCurrentUserData(authData.user);

      if (companyId) {
        const cData = await apiFetch<CompanyResponse>(`/companies/${companyId}`);
        setCurrentWorkspaceName(cData.data?.name || cData.name || "Workspace");
      }

      const mData = await apiFetch<Meeting[] | ApiResponse<Meeting[]>>("/meetings");
      setMeetings(Array.isArray(mData) ? mData : mData.data || []); 

      const tData = await apiFetch<Task[] | ApiResponse<Task[]>>("/tasks");
      setTasks(Array.isArray(tData) ? tData : tData.data || []);

      const uData = await apiFetch<CurrentUser[] | ApiResponse<CurrentUser[]>>("/users");
      const membersArray = Array.isArray(uData) ? uData : uData.data || [];
      setMemberCount(membersArray.length);

    } catch (err) {
      console.error("Failed to fetch dashboard data:", err);
    } finally {
      setIsLoadingData(false);
    }
  }, []); 

  useEffect(() => {
    if (!token || needsWorkspace || isChecking) return;
    fetchDashboardData();
  }, [token, needsWorkspace, isChecking, fetchDashboardData]); 

  const handleCreateWorkspace = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsCreating(true);
    setError("");

    try {
      const data = await apiFetch<{ data?: { id: string | number }; id?: string | number }>("/companies", {
        method: "POST",
        data: { name: workspaceName },
      });

      const companyId = data.data?.id || data.id;
      if (!companyId) throw new Error("Workspace created, but ID was missing.");

      localStorage.setItem("companyId", String(companyId));
      setNeedsWorkspace(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create workspace.");
    } finally {
      setIsCreating(false);
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    router.push("/signin");
  };

  if (isChecking) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <Loader2 className="w-8 h-8 text-brand-maroon animate-spin" />
      </div>
    );
  }

  if (needsWorkspace) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 p-4">
        <div className="max-w-4xl w-full grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 flex flex-col">
            <div className="w-12 h-12 bg-brand-maroon/10 text-brand-maroon rounded-xl flex items-center justify-center mb-6">
              <Plus strokeWidth={2.5} />
            </div>
            
            <h2 className="text-2xl font-serif font-bold text-slate-900 mb-2">Create Workspace</h2>
            <form onSubmit={handleCreateWorkspace} className="flex flex-col gap-5 mt-4">
              <input 
                type="text" 
                placeholder="e.g., LughaCap Team" 
                value={workspaceName}
                onChange={(e) => setWorkspaceName(e.target.value)}
                required
                className="w-full px-4 py-3 rounded-lg border border-slate-200 focus:border-brand-gold focus:ring-1 focus:ring-brand-gold outline-none"
              />
              {error && <p className="text-sm text-red-500">{error}</p>}
              <button type="submit" disabled={isCreating} className="w-full bg-brand-maroon text-white py-3 rounded-lg font-medium hover:bg-brand-gold transition-colors">
                {isCreating ? <Loader2 className="w-5 h-5 animate-spin mx-auto" /> : "Start Workspace"}
              </button>
            </form>
          </div>
          <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 flex flex-col justify-between">
            <div className="w-12 h-12 bg-slate-100 text-slate-600 rounded-xl flex items-center justify-center mb-6">
              <Users strokeWidth={2.5} />
            </div>
            <h2 className="text-2xl font-serif font-bold text-slate-900 mb-2">Join a Team</h2>
            <p className="text-slate-500 mb-4 text-sm">Ask your administrator to invite you via their settings panel.</p>
            <div className="w-full bg-slate-50 border border-slate-100 p-4 rounded-lg flex items-center gap-3">
              <Building className="w-5 h-5 text-slate-400" />
              <p className="text-xs text-slate-600 font-medium">Verify your email address with your team lead.</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const isAdminOrOwner = activeRole === 'ADMIN' || activeRole === 'OWNER';

  return (
    <div className="min-h-screen bg-slate-50 flex overflow-hidden">
      <aside className="w-64 bg-white border-r border-slate-200 flex flex-col shrink-0">
        <div className="h-20 flex items-center px-6 border-b border-slate-200">
          <Image src="/LughaCap_Icon.png" alt="Icon" width={32} height={32} className="mr-3 object-contain" />
          <span className="font-serif text-2xl font-bold text-brand-maroon tracking-tight">LughaCap</span>
        </div>
        <nav className="flex-1 px-4 py-6 space-y-2">
          <button onClick={() => setActiveTab('dashboard')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium ${activeTab === 'dashboard' ? 'bg-brand-maroon/5 text-brand-maroon' : 'text-slate-600 hover:bg-slate-50'}`}>
            <Home size={20} /> Dashboard
          </button>
          <button onClick={() => setActiveTab('meetings')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium ${activeTab === 'meetings' ? 'bg-brand-maroon/5 text-brand-maroon' : 'text-slate-600 hover:bg-slate-50'}`}>
            <Video size={20} /> Meetings Hub
          </button>
          <button onClick={() => setActiveTab('tasks')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium ${activeTab === 'tasks' ? 'bg-brand-maroon/5 text-brand-maroon' : 'text-slate-600 hover:bg-slate-50'}`}>
            <CheckSquare size={20} /> Action Items
          </button>
        </nav>
        <div className="p-4 border-t border-slate-200 space-y-2">
          {/* Settings tab for Admins and Owners only */}
          {isAdminOrOwner && (
            <button onClick={() => setActiveTab('settings')} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium ${activeTab === 'settings' ? 'bg-brand-maroon/5 text-brand-maroon' : 'text-slate-600 hover:bg-slate-50'}`}>
              <SettingsIcon size={20} /> Settings
            </button>
          )}
          <button onClick={handleLogout} className="w-full flex items-center gap-3 px-4 py-3 text-red-600 hover:bg-red-50 rounded-xl font-medium">
            <LogOut size={20} /> Sign Out
          </button>
        </div>
      </aside>

      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        <header className="h-20 bg-white border-b border-slate-200 flex items-center justify-between px-8 shrink-0">
          <div className="flex items-center gap-4">
    <Link href="/" className="p-2 text-slate-400 hover:text-brand-maroon hover:bg-slate-100 rounded-lg transition-colors">
      <ArrowLeft size={20} />
    </Link>
    <h2 className="text-xl font-medium text-slate-800">
      {activeTab === 'dashboard' && `Welcome to ${currentWorkspaceName}${username ? `, ${username}` : ""}!`}
      {activeTab === 'meetings' && "Meetings Hub"}
      {activeTab === 'tasks' && "Action Items"}
      {activeTab === 'settings' && "Team Settings"}
    </h2>
  </div>
          
          {isAdminOrOwner && (
            <button 
              onClick={() => setIsUploadModalOpen(true)}
              className="bg-brand-maroon text-white px-6 py-2.5 rounded-xl font-medium flex items-center gap-2 hover:bg-brand-gold transition-all shadow-sm"
            >
              <Upload size={18} /> Upload Recording
            </button>
          )}
        </header>

        <div className="flex-1 overflow-y-auto p-8">
          {activeTab === 'dashboard' && <DashboardHome meetings={meetings} tasks={tasks} memberCount={memberCount} isLoading={isLoadingData} onUploadClick={() => setIsUploadModalOpen(true)} isAdminOrOwner={isAdminOrOwner} currentUserData={currentUserData} />}
          {activeTab === 'meetings' && <MeetingsHub meetings={meetings} isLoading={isLoadingData} onMeetingUpdated={fetchDashboardData} />}
          {activeTab === 'tasks' && <ActionItems tasks={tasks} meetings={meetings} isLoading={isLoadingData} onTaskUpdated={fetchDashboardData} />}
          {activeTab === 'settings' && isAdminOrOwner && <TeamSettings />}
        </div>
      </main>

      {isUploadModalOpen && (
        <UploadModal 
          onClose={() => setIsUploadModalOpen(false)} 
          onSuccess={() => {
            setIsUploadModalOpen(false);
            fetchDashboardData(); 
          }} 
        />
      )}
    </div>
  );
}