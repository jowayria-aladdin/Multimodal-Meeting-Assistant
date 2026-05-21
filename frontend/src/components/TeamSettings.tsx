"use client";

import { useState, useEffect, useCallback } from "react";
import { Users, UserPlus, Shield, Loader2, Trash2, AlertTriangle, Mail, Building, Check, Lock } from "lucide-react";
import { apiFetch } from "@/lib/api";

interface Membership {
  role?: string;
  companyId?: string | number;
  company_id?: string | number;
}

interface Member {
  id: string | number;
  username?: string;
  email?: string;
  role?: string;
  memberships?: Membership[]; 
  companyMemberships?: Membership[]; 
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

interface CompanyResponse {
  data?: { name?: string };
  name?: string;
}

type UsersResponse = Member[] | { data?: Member[] };

export default function TeamSettings() {
  const localUsername = typeof window !== "undefined" ? localStorage.getItem("username") : null;
  
  const [members, setMembers] = useState<Member[]>([]);
  const [activeRole, setActiveRole] = useState<string | null>(null);
  const [currentUserData, setCurrentUserData] = useState<CurrentUser | null>(null); 
  
  const [isLoadingMembers, setIsLoadingMembers] = useState(true);
  const [isLoadingAuth, setIsLoadingAuth] = useState(true);
  
  // Workspace States
  const [workspaceName, setWorkspaceName] = useState("");
  const [isUpdatingWorkspace, setIsUpdatingWorkspace] = useState(false);
  const [showDeleteWorkspaceModal, setShowDeleteWorkspaceModal] = useState(false);
  const [isDeletingWorkspace, setIsDeletingWorkspace] = useState(false);

  // Invite States
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("MEMBER");
  const [isInviting, setIsInviting] = useState(false);
  const [inviteError, setInviteError] = useState("");

  // Member Management States
  const [memberToRemove, setMemberToRemove] = useState<Member | null>(null);
  const [isRemoving, setIsRemoving] = useState(false);
  const [isChangingRoleId, setIsChangingRoleId] = useState<string | number | null>(null);

  const fetchAuth = useCallback(async () => {
    setIsLoadingAuth(true);
    try {
      const data = await apiFetch<AuthResponse>("/auth/me");
      setActiveRole(data.activeRole ? data.activeRole.toUpperCase() : "MEMBER");
      if (data.user) setCurrentUserData(data.user);
    } catch (err) {
      console.error("Failed to fetch auth", err);
      setActiveRole("MEMBER");
    } finally {
      setIsLoadingAuth(false);
    }
  }, []);

  const fetchMembers = useCallback(async () => {
    setIsLoadingMembers(true);
    try {
      const usersData = await apiFetch<UsersResponse>("/users");
      setMembers(Array.isArray(usersData) ? usersData : usersData.data || []);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoadingMembers(false);
    }
  }, []);

  const fetchCompanyDetails = useCallback(async () => {
    try {
      const companyId = localStorage.getItem("companyId");
      if (!companyId) return;

      const data = await apiFetch<CompanyResponse>(`/companies/${companyId}`);
      setWorkspaceName(data.data?.name || data.name || "");
    } catch (err) {
      console.error("Failed to fetch company details", err);
    }
  }, []);

  useEffect(() => {
    fetchAuth();
    fetchMembers();
    fetchCompanyDetails();
  }, [fetchAuth, fetchMembers, fetchCompanyDetails]);

  const handleUpdateWorkspace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!workspaceName.trim()) return;

    setIsUpdatingWorkspace(true);
    try {
      const companyId = localStorage.getItem("companyId");
      await apiFetch<void>(`/companies/${companyId}`, {
        method: "PATCH",
        data: { name: workspaceName.trim() }
      });
    } catch (err) {
      console.error(err);
      alert("Failed to update workspace name.");
    } finally {
      setIsUpdatingWorkspace(false);
    }
  };

  const executeDeleteWorkspace = async () => {
    setIsDeletingWorkspace(true);
    try {
      const companyId = localStorage.getItem("companyId");
      await apiFetch<void>(`/companies/${companyId}`, { method: "DELETE" });

      localStorage.removeItem("companyId");
      window.location.reload();
    } catch (err) {
      console.error(err);
      alert("Failed to delete workspace.");
      setIsDeletingWorkspace(false);
      setShowDeleteWorkspaceModal(false);
    }
  };

  const handleInvite = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inviteEmail.trim()) return;
    
    setIsInviting(true);
    setInviteError("");

    try {
      const companyId = localStorage.getItem("companyId");
      await apiFetch<void>(`/companies/${companyId}/memberships`, {
        method: "POST",
        data: { email: inviteEmail.trim(), role: inviteRole }
      });

      setInviteEmail("");
      setInviteRole("MEMBER");
      fetchMembers(); 
    } catch (err) {
      setInviteError(err instanceof Error ? err.message : "An unexpected error occurred.");
    } finally {
      setIsInviting(false);
    }
  };

  const executeRemove = async () => {
    if (!memberToRemove) return;
    setIsRemoving(true);

    try {
      const companyId = localStorage.getItem("companyId");
      await apiFetch<void>(`/companies/${companyId}/memberships/${memberToRemove.id}`, { method: "DELETE" });

      setMemberToRemove(null);
      fetchMembers(); 
    } catch (err) {
      console.error(err);
      alert("Failed to remove member.");
    } finally {
      setIsRemoving(false);
    }
  };

  const handleRoleChange = async (memberId: string | number, newRole: string) => {
    setIsChangingRoleId(memberId);
    try {
      const companyId = localStorage.getItem("companyId");
      await apiFetch<void>(`/companies/${companyId}/memberships/${memberId}`, {
        method: "PATCH",
        data: { role: newRole.toLowerCase() }
      });

      fetchMembers();
    } catch (err) {
      console.error(err);
      alert("Failed to update role.");
    } finally {
      setIsChangingRoleId(null);
    }
  };

  // --- STRICT RBAC ENFORCEMENT ---
  const isAdminOrOwner = activeRole === 'ADMIN' || activeRole === 'OWNER';

  if (isLoadingAuth) {
    return (
      <div className="flex flex-col items-center justify-center min-h-125">
        <Loader2 className="w-10 h-10 text-brand-maroon animate-spin" />
        <p className="mt-4 text-slate-500 font-medium">Verifying access...</p>
      </div>
    );
  }

  // Backup fallback UI in case they manage to manipulate their browser state
  if (!isAdminOrOwner) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] bg-white rounded-3xl border border-slate-200 shadow-sm p-12 text-center">
        <div className="w-24 h-24 bg-red-50 text-red-500 rounded-full flex items-center justify-center mb-6">
          <Lock size={40} strokeWidth={2} />
        </div>
        <h2 className="text-3xl font-serif font-bold text-slate-900 mb-3">Access Denied</h2>
        <p className="text-slate-500 max-w-md text-lg">
          You must be an <strong className="text-slate-700">Admin</strong> or <strong className="text-slate-700">Owner</strong> of this workspace to view and manage Team Settings.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in duration-300">
      
      <div>
        <h2 className="text-2xl font-serif font-bold text-slate-900">Team Settings</h2>
        <p className="text-slate-500 mt-1">Manage your workspace members and admin privileges.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        <div className="lg:col-span-1 space-y-6">
          
          {/* WORKSPACE PROFILE */}
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm transition-all">
            <div className="w-10 h-10 bg-brand-gold/10 text-yellow-700 rounded-lg flex items-center justify-center mb-4">
              <Building size={20} />
            </div>
            <h3 className="font-bold text-slate-900 mb-2">Workspace Profile</h3>
            <p className="text-sm text-slate-500 mb-6">
              Update your company name or permanently delete this workspace.
            </p>
            
            <form onSubmit={handleUpdateWorkspace} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">Workspace Name</label>
                <input 
                  type="text" 
                  value={workspaceName}
                  onChange={(e) => setWorkspaceName(e.target.value)}
                  disabled={isUpdatingWorkspace}
                  className="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold text-sm transition-colors"
                />
              </div>
              
              <button 
                type="submit" 
                disabled={isUpdatingWorkspace || !workspaceName.trim()}
                className="w-full bg-slate-900 text-white py-2.5 rounded-xl text-sm font-medium hover:bg-slate-800 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {isUpdatingWorkspace ? <Loader2 size={16} className="animate-spin" /> : <><Check size={16} /> Save Changes</>}
              </button>
            </form>

            <div className="mt-6 pt-6 border-t border-slate-100">
              <button 
                onClick={() => setShowDeleteWorkspaceModal(true)}
                className="w-full flex items-center justify-center gap-2 px-4 py-2.5 border-2 border-red-100 text-red-600 rounded-xl text-sm font-bold hover:bg-red-50 hover:border-red-200 transition-colors"
              >
                <Trash2 size={16} /> Delete Workspace
              </button>
            </div>
          </div>

          {/* INVITE MEMBER */}
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
            <div className="w-10 h-10 bg-brand-maroon/10 text-brand-maroon rounded-lg flex items-center justify-center mb-4">
              <UserPlus size={20} />
            </div>
            <h3 className="font-bold text-slate-900 mb-2">Invite Member</h3>
            <p className="text-sm text-slate-500 mb-6">
              Enter the Email Address of the person you want to add to your workspace.
            </p>
            
            <form onSubmit={handleInvite} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">Email Address</label>
                <input 
                  type="email" 
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  placeholder="e.g., lugha@example.com" 
                  disabled={isInviting}
                  className="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold text-sm transition-colors"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">Assign Role</label>
                <select 
                  value={inviteRole}
                  onChange={(e) => setInviteRole(e.target.value)}
                  disabled={isInviting}
                  className="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold text-sm transition-colors bg-white cursor-pointer"
                >
                  <option value="MEMBER">Member (Standard Access)</option>
                  <option value="ADMIN">Admin (Full Access)</option>
                </select>
              </div>
              
              {inviteError && <p className="text-xs text-red-500 font-medium">{inviteError}</p>}
              
              <button 
                type="submit" 
                disabled={isInviting || !inviteEmail.trim()}
                className="w-full bg-brand-maroon text-white py-2.5 rounded-xl text-sm font-medium hover:bg-brand-gold transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {isInviting ? <Loader2 size={16} className="animate-spin" /> : "Send Invitation"}
              </button>
            </form>
          </div>
        </div>

        {/* Right Column: Members Table */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden flex flex-col h-full">
            <div className="px-6 py-5 border-b border-slate-200 flex items-center justify-between bg-slate-50/50">
              <h3 className="font-bold text-slate-900 flex items-center gap-2">
                <Users size={18} className="text-brand-maroon" /> Workspace Members
              </h3>
              <span className="bg-slate-100 text-slate-600 text-xs font-bold px-3 py-1 rounded-full">
                {members.length} Total
              </span>
            </div>

            {isLoadingMembers ? (
              <div className="p-12 flex flex-col items-center justify-center text-slate-500">
                <Loader2 className="w-8 h-8 text-brand-maroon animate-spin mb-4" />
                <p>Loading team members...</p>
              </div>
            ) : members.length === 0 ? (
              <div className="p-12 text-center text-slate-500">
                No team members found. Invite someone to get started!
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="bg-slate-50 border-b border-slate-100 text-slate-500">
                    <tr>
                      <th className="px-6 py-4 font-medium">User</th>
                      <th className="px-6 py-4 font-medium">Role Privilege</th>
                      <th className="px-6 py-4 font-medium text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {members.map((member) => {
                      const isMe = 
                        (currentUserData && member.email?.toLowerCase() === currentUserData.email?.toLowerCase()) ||
                        member.username?.toLowerCase() === localUsername?.toLowerCase() || 
                        member.email?.toLowerCase() === localUsername?.toLowerCase();
                      
                      const extractedRole = member.role ? String(member.role).toUpperCase() : 'MEMBER';
                      const displayRole = isMe && activeRole ? activeRole.toUpperCase() : extractedRole;

                      return (
                        <tr key={member.id} className="hover:bg-slate-50/50 transition-colors group">
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 rounded-full bg-brand-maroon/10 text-brand-maroon flex items-center justify-center font-bold text-xs uppercase shrink-0">
                                {(member.username || member.email || "U")[0]}
                              </div>
                              <div className="min-w-0">
                                <p className="font-medium text-slate-900 truncate max-w-37.5">{member.username || `User #${member.id}`}</p>
                                {member.email && <p className="text-xs text-slate-500 flex items-center gap-1 mt-0.5 truncate max-w-37.5"><Mail size={10} className="shrink-0"/> {member.email}</p>}
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            {isMe || displayRole === 'OWNER' ? (
                               <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-bold ${displayRole === 'OWNER' || displayRole === 'ADMIN' ? 'bg-brand-gold/10 text-yellow-700' : 'bg-slate-100 text-slate-600'}`}>
                                 {(displayRole === 'ADMIN' || displayRole === 'OWNER') && <Shield size={10} />}
                                 {displayRole} {isMe && "(You)"}
                               </span>
                            ) : (
                               <div className="relative inline-block">
                                 {isChangingRoleId === member.id && (
                                    <div className="absolute -left-6 top-1/2 -translate-y-1/2">
                                      <Loader2 size={14} className="animate-spin text-brand-maroon" />
                                    </div>
                                 )}
                                 <select 
                                   value={displayRole}
                                   onChange={(e) => handleRoleChange(member.id, e.target.value)}
                                   disabled={isChangingRoleId === member.id}
                                   className={`cursor-pointer appearance-none px-3 py-1 pr-8 rounded-md text-xs font-bold border transition-colors outline-none focus:ring-2 focus:ring-brand-gold ${displayRole === 'ADMIN' ? 'bg-brand-gold/10 text-yellow-700 border-brand-gold/20 hover:bg-brand-gold/20' : 'bg-slate-100 text-slate-700 border-slate-200 hover:bg-slate-200'}`}
                                 >
                                   <option value="MEMBER">MEMBER</option>
                                   <option value="ADMIN">ADMIN</option>
                                 </select>
                               </div>
                            )}
                          </td>
                          <td className="px-6 py-4 text-right">
                            {!isMe && displayRole !== 'OWNER' && (
                              <button 
                                onClick={() => setMemberToRemove(member)}
                                className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
                                title="Remove Member"
                              >
                                <Trash2 size={16} />
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>

      {memberToRemove && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white max-w-md w-full rounded-2xl shadow-xl overflow-hidden animate-in zoom-in-95 duration-200">
            <div className="p-6">
              <div className="w-12 h-12 rounded-full bg-red-100 text-red-600 flex items-center justify-center mb-4">
                <AlertTriangle size={24} />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-2">Remove Team Member?</h3>
              <p className="text-slate-500 mb-6">
                Are you sure you want to remove <span className="font-semibold text-slate-700">{memberToRemove.username || memberToRemove.email || `User #${memberToRemove.id}`}</span> from the workspace? They will lose access to all meetings and tasks.
              </p>
              <div className="flex gap-3 w-full">
                <button 
                  onClick={() => setMemberToRemove(null)}
                  disabled={isRemoving}
                  className="flex-1 px-4 py-2.5 border border-slate-200 text-slate-700 font-medium rounded-xl hover:bg-slate-50 transition-colors"
                >
                  Cancel
                </button>
                <button 
                  onClick={executeRemove}
                  disabled={isRemoving}
                  className="flex-1 px-4 py-2.5 bg-red-600 text-white font-medium rounded-xl hover:bg-red-700 transition-colors flex items-center justify-center gap-2"
                >
                  {isRemoving ? <Loader2 size={18} className="animate-spin" /> : "Remove Member"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showDeleteWorkspaceModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white max-w-lg w-full rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
            <div className="p-8">
              <div className="w-14 h-14 rounded-full bg-red-100 text-red-600 flex items-center justify-center mb-6">
                <AlertTriangle size={28} strokeWidth={2.5} />
              </div>
              <h3 className="text-2xl font-serif font-bold text-slate-900 mb-3">Delete Workspace?</h3>
              <p className="text-slate-500 mb-6 leading-relaxed">
                You are about to permanently delete <span className="font-bold text-slate-900">&quot;{workspaceName}&quot;</span>. This action <strong className="text-red-600">cannot be undone</strong> and will immediately destroy all meetings, transcripts, and action items for all team members.
              </p>
              
              <div className="flex gap-3 w-full mt-8">
                <button 
                  onClick={() => setShowDeleteWorkspaceModal(false)}
                  disabled={isDeletingWorkspace}
                  className="flex-1 px-4 py-3 border-2 border-slate-200 text-slate-700 font-bold rounded-xl hover:bg-slate-50 transition-colors"
                >
                  Cancel
                </button>
                <button 
                  onClick={executeDeleteWorkspace}
                  disabled={isDeletingWorkspace}
                  className="flex-1 px-4 py-3 bg-red-600 text-white font-bold rounded-xl hover:bg-red-700 transition-colors flex items-center justify-center gap-2 shadow-sm"
                >
                  {isDeletingWorkspace ? <Loader2 size={18} className="animate-spin" /> : "Permanently Delete"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}