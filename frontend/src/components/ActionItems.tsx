"use client";

import { useState, useEffect } from "react";
import { CheckSquare, Circle, CheckCircle2, Loader2, Calendar, Video, ArrowUpDown, Trash2, AlertTriangle, Plus, X, Edit2, User } from "lucide-react";
import { apiFetch } from "@/lib/api";

interface Meeting { id: string | number; title: string; }

interface TaskAssignee {
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

interface Task {
  id: string | number;
  task_text: string;
  status?: string;
  due_date?: string;
  meeting_id?: string | number; 
  assignees?: TaskAssignee[]; 
  taskassignees?: TaskAssignee[]; 
  taskAssignees?: TaskAssignee[]; 
  TaskAssignees?: TaskAssignee[];
}

interface Member {
  id: string | number;
  username?: string;
  email?: string;
  role?: string; 
}

interface Props {
  tasks: Task[];
  meetings: Meeting[]; 
  isLoading: boolean;
  onTaskUpdated: () => void; 
}

interface CreateTaskPayload {
  task_text: string;
  status: string;
  meeting_id?: number;
  due_date?: string;
}

interface AuthResponse {
  activeRole?: string;
}

type UsersResponse = Member[] | { data?: Member[] };

interface CreateTaskResponse {
  data?: { id: string | number };
  id?: string | number;
}

type FilterType = 'ALL' | 'PENDING' | 'COMPLETED';
type SortOrder = 'NEAREST' | 'LATEST';

export default function ActionItems({ tasks, meetings, isLoading, onTaskUpdated }: Props) {
  const currentUsername = typeof window !== "undefined" ? localStorage.getItem("username") : null;

  const [filter, setFilter] = useState<FilterType>('ALL');
  const [sortOrder, setSortOrder] = useState<SortOrder>('NEAREST');
  const [updatingTaskId, setUpdatingTaskId] = useState<string | number | null>(null);
  
  const [taskToDelete, setTaskToDelete] = useState<Task | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newTaskText, setNewTaskText] = useState("");
  const [newTaskDueDate, setNewTaskDueDate] = useState("");
  const [newTaskMeetingId, setNewTaskMeetingId] = useState("");
  const [newTaskAssigneeIds, setNewTaskAssigneeIds] = useState<string[]>([]);
  const [isCreating, setIsCreating] = useState(false);

  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [editTaskData, setEditTaskData] = useState({
    id: "",
    task_text: "",
    due_date: "",
    meeting_id: "",
    assignee_ids: [] as string[],
    initial_assignee_ids: [] as string[] 
  });

  const [members, setMembers] = useState<Member[]>([]);
  const [activeRole, setActiveRole] = useState<string | null>(null);

  useEffect(() => {
    const fetchAuthAndMembers = async () => {
      try {
        const usersData = await apiFetch<UsersResponse>("/users");
        setMembers(Array.isArray(usersData) ? usersData : usersData.data || []);

        const authData = await apiFetch<AuthResponse>("/auth/me");
        setActiveRole(authData.activeRole || "MEMBER");
      } catch (err) {
        console.error("Failed to fetch data", err);
      }
    };
    fetchAuthAndMembers();
  }, []);

  //Checks every possible backend naming convention
  const getTaskAssigneesList = (task: Task): TaskAssignee[] => {
    return task.taskAssignees || task.taskassignees || task.assignees || task.TaskAssignees || [];
  };

  // Guarantees we find the exact Member ID for the checkboxes
  const resolveMemberId = (a: TaskAssignee): string | null => {
    const directUid = a.user?.id || a.userId || a.user_id;
    let match = members.find(m => String(m.id) === String(directUid));
    
    if (!match && (a.email || a.user?.email)) {
      const targetEmail = a.email || a.user?.email;
      match = members.find(m => m.email === targetEmail);
    }
    return match ? String(match.id) : null;
  };

  const handleCreateTask = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTaskText.trim()) return;

    setIsCreating(true);

    try {
      const payload: CreateTaskPayload = {
        task_text: newTaskText.trim(),
        status: "TODO"
      };

      if (newTaskMeetingId) payload.meeting_id = parseInt(newTaskMeetingId);
      if (newTaskDueDate) payload.due_date = new Date(newTaskDueDate).toISOString();

      const createdTaskData = await apiFetch<CreateTaskResponse>("/tasks", {
        method: "POST",
        data: payload 
      });

      const newTaskId = createdTaskData.data?.id || createdTaskData.id;

      if (newTaskAssigneeIds.length > 0 && newTaskId) {
        const assignmentPromises = newTaskAssigneeIds.map(async (id) => {
          const selectedMember = members.find(m => String(m.id) === id);
          if (selectedMember && selectedMember.email) {
            return apiFetch<void>(`/tasks/${newTaskId}/assignees`, {
              method: "POST",
              data: { email: selectedMember.email }
            });
          }
        });
        
        await Promise.all(assignmentPromises);
      }

      setNewTaskText(""); setNewTaskDueDate(""); setNewTaskMeetingId(""); setNewTaskAssigneeIds([]);
      setIsCreateModalOpen(false);
      onTaskUpdated();
      
    } catch (err) {
      console.error(err);
      alert("Failed to create task."); 
    } finally {
      setIsCreating(false);
    }
  };

  const openEditModal = (task: Task) => {
    let formattedDate = "";
    if (task.due_date) {
      formattedDate = new Date(task.due_date).toISOString().split('T')[0];
    }

    const assigneesList = getTaskAssigneesList(task);
    // Maps the nested data to actual UI Member IDs so the checkboxes turn ON
    const existingAssigneeIds = assigneesList.map(resolveMemberId).filter(Boolean) as string[];

    setEditTaskData({
      id: String(task.id),
      task_text: task.task_text,
      due_date: formattedDate,
      meeting_id: task.meeting_id ? String(task.meeting_id) : "",
      assignee_ids: existingAssigneeIds,
      initial_assignee_ids: existingAssigneeIds
    });
    
    setIsUpdateModalOpen(true);
  };

  const handleUpdateTaskDetails = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editTaskData.task_text.trim()) return;

    setIsUpdating(true);

    try {
      await apiFetch<void>(`/tasks/${editTaskData.id}`, {
        method: "PATCH",
        data: { 
          task_text: editTaskData.task_text.trim(),
          meeting_id: editTaskData.meeting_id ? parseInt(editTaskData.meeting_id) : null,
          due_date: editTaskData.due_date ? new Date(editTaskData.due_date).toISOString() : null
        }
      });

      const idsToAdd = editTaskData.assignee_ids.filter(id => !editTaskData.initial_assignee_ids.includes(id));
      const idsToRemove = editTaskData.initial_assignee_ids.filter(id => !editTaskData.assignee_ids.includes(id));

      const assignPromises: Promise<void>[] = [];

      // Add new ones
      idsToAdd.forEach(id => {
        const selectedMember = members.find(m => String(m.id) === id);
        if (selectedMember && selectedMember.email) {
          assignPromises.push(
            apiFetch<void>(`/tasks/${editTaskData.id}/assignees`, {
              method: "POST",
              data: { email: selectedMember.email }
            })
          );
        }
      });

      // Delete unchecked ones
      idsToRemove.forEach(id => {
        const selectedMember = members.find(m => String(m.id) === id);
        if (selectedMember) {
          assignPromises.push(
            apiFetch<void>(`/tasks/${editTaskData.id}/assignees/${selectedMember.id}`, {
              method: "DELETE"
            })
          );
        }
      });

      await Promise.all(assignPromises);

      setIsUpdateModalOpen(false);
      onTaskUpdated();

    } catch (err) {
      console.error(err);
      alert(`Update Failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setIsUpdating(false);
    }
  };

  const handleToggleStatus = async (task: Task) => {
    const isCurrentlyCompleted = task.status === "DONE" || task.status === "COMPLETED";
    setUpdatingTaskId(task.id);
    try {
      await apiFetch<void>(`/tasks/${task.id}`, {
        method: "PATCH",
        data: { status: isCurrentlyCompleted ? "TODO" : "DONE" }
      });
      
      onTaskUpdated();
    } catch (err) {
      console.error(err);
      alert("Failed to update status."); 
    } finally {
      setUpdatingTaskId(null);
    }
  };

  const executeDelete = async () => {
    if (!taskToDelete) return;
    setIsDeleting(true);
    try {
      await apiFetch<void>(`/tasks/${taskToDelete.id}`, { method: "DELETE" });
      onTaskUpdated();
      setTaskToDelete(null);
    } catch (err) {
      console.error(err);
      alert("Failed to delete task.");
    } finally {
      setIsDeleting(false);
    }
  };

  const handleToggleAssignee = (idStr: string, isEditing: boolean) => {
    if (isEditing) {
      setEditTaskData(prev => ({
        ...prev,
        assignee_ids: prev.assignee_ids.includes(idStr) 
          ? prev.assignee_ids.filter(i => i !== idStr) 
          : [...prev.assignee_ids, idStr]
      }));
    } else {
      setNewTaskAssigneeIds(prev => 
        prev.includes(idStr) ? prev.filter(i => i !== idStr) : [...prev, idStr]
      );
    }
  };

  if (isLoading) {
    return (
      <div className="flex-1 h-full flex flex-col items-center justify-center min-h-100">
        <Loader2 className="w-10 h-10 text-brand-maroon animate-spin" />
        <p className="mt-4 text-slate-500 font-medium">Loading your action items...</p>
      </div>
    );
  }

  const currentUser = members.find(m => 
    m.username?.toLowerCase() === currentUsername?.toLowerCase() || 
    m.email?.toLowerCase() === currentUsername?.toLowerCase()
  );
  
  //ROLE CHECK: lock the UI based on the exact role from the updated users list
  const memberRole = currentUser?.role ? String(currentUser.role).toUpperCase() : (activeRole?.toUpperCase() || 'MEMBER');
  const isAdmin = memberRole === 'ADMIN' || memberRole === 'OWNER';

  const processedTasks = tasks.filter(task => {
    const isCompleted = task.status === "DONE" || task.status === "COMPLETED";
    
    if (filter === 'PENDING' && isCompleted) return false;
    if (filter === 'COMPLETED' && !isCompleted) return false;

    if (!isAdmin && currentUser) {
      const assigneesList = getTaskAssigneesList(task);
      const isAssignedToMe = assigneesList.some(a => resolveMemberId(a) === String(currentUser.id));
      if (!isAssignedToMe) return false;
    }

    return true; 
  }).sort((a, b) => {
    if (!a.due_date) return 1;
    if (!b.due_date) return -1;
    const dateA = new Date(a.due_date).getTime();
    const dateB = new Date(b.due_date).getTime();
    return sortOrder === 'NEAREST' ? dateA - dateB : dateB - dateA;
  });

  const getMeetingName = (meetingId?: string | number) => {
    if (!meetingId) return "Unknown Meeting";
    const meeting = meetings.find(m => m.id == meetingId); 
    return meeting ? meeting.title : "Unknown Meeting";
  };

  return (
    <div className="space-y-6 relative">
      
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h2 className="text-2xl font-serif font-bold text-slate-900">Action Items</h2>
          <p className="text-slate-500 mt-1">Track and manage tasks generated from your meetings.</p>
        </div>
        {isAdmin && (
          <button 
            onClick={() => setIsCreateModalOpen(true)}
            className="bg-brand-maroon text-white px-4 py-2.5 rounded-xl font-medium flex items-center gap-2 hover:bg-brand-gold transition-colors shadow-sm w-full sm:w-auto justify-center"
          >
            <Plus size={18} strokeWidth={2.5} />
            New Task
          </button>
        )}
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white p-2 rounded-2xl border border-slate-200 shadow-sm">
        <div className="flex items-center p-1 bg-slate-50 rounded-xl border border-slate-100">
          <button onClick={() => setFilter('ALL')} className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${filter === 'ALL' ? 'bg-white text-slate-900 shadow-sm border border-slate-200/50' : 'text-slate-500 hover:text-slate-700'}`}>All Tasks</button>
          <button onClick={() => setFilter('PENDING')} className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${filter === 'PENDING' ? 'bg-white text-brand-maroon shadow-sm border border-slate-200/50' : 'text-slate-500 hover:text-slate-700'}`}>Pending</button>
          <button onClick={() => setFilter('COMPLETED')} className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${filter === 'COMPLETED' ? 'bg-white text-green-700 shadow-sm border border-slate-200/50' : 'text-slate-500 hover:text-slate-700'}`}>Completed</button>
        </div>
        <div className="flex items-center gap-2 px-3">
          <ArrowUpDown className="w-4 h-4 text-slate-400" />
          <select value={sortOrder} onChange={(e) => setSortOrder(e.target.value as SortOrder)} className="bg-transparent text-sm font-medium text-slate-700 focus:outline-none cursor-pointer">
            <option value="NEAREST">Due: Nearest First</option>
            <option value="LATEST">Due: Latest First</option>
          </select>
        </div>
      </div>

      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        {tasks.length === 0 ? (
          <div className="p-16 text-center flex flex-col items-center justify-center min-h-100">
            <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mb-6">
              <CheckSquare className="w-10 h-10 text-slate-300" />
            </div>
            <h3 className="text-2xl font-serif font-bold text-slate-900 mb-2">No action items yet</h3>
            <p className="text-slate-500 max-w-md">Create a new manual task, or upload a meeting to have the AI extract them automatically.</p>
          </div>
        ) : processedTasks.length === 0 ? (
          <div className="p-12 text-center text-slate-500">No tasks found for your account matching this filter.</div>
        ) : (
          <div className="divide-y divide-slate-100">
            {processedTasks.map((task) => {
              const isCompleted = task.status === "DONE" || task.status === "COMPLETED";
              const isUpdatingThisTask = updatingTaskId === task.id; 
              
              const assigneesList = getTaskAssigneesList(task);
              const assignedUsers = assigneesList.map(a => {
                const uid = resolveMemberId(a);
                const memberMatch = members.find(m => String(m.id) === uid);
                // Fallback rendering in case member isn't explicitly found
                return memberMatch || { id: uid || a.id, username: a.user?.username, email: a.user?.email || a.email };
              });

              return (
                <div key={task.id} className={`p-4 sm:p-5 flex items-start sm:items-center gap-4 hover:bg-slate-50 transition-colors group ${isCompleted ? 'opacity-70' : ''}`}>
                  
                  <button onClick={() => handleToggleStatus(task)} disabled={isUpdatingThisTask} className="mt-1 sm:mt-0 shrink-0 focus:outline-none disabled:opacity-50">
                    {isUpdatingThisTask ? <Loader2 className="w-6 h-6 text-brand-maroon animate-spin" /> : isCompleted ? <CheckCircle2 className="w-6 h-6 text-green-500 hover:text-green-600 transition-colors" /> : <Circle className="w-6 h-6 text-slate-300 group-hover:text-brand-gold transition-colors" />}
                  </button>

                  <div className="flex-1 min-w-0">
                    <p className={`font-medium truncate ${isCompleted ? 'text-slate-500 line-through' : 'text-slate-900'}`}>{task.task_text}</p>
                    <div className="flex flex-wrap items-center gap-3 mt-2 text-xs">
                      
                      {assignedUsers.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                          {assignedUsers.map((u, idx) => (
                            <span key={idx} className={`flex items-center gap-1.5 px-2 py-1 rounded-md font-medium ${isCompleted ? 'bg-slate-100 text-slate-500' : 'bg-brand-gold/10 text-yellow-700 border border-brand-gold/20'}`}>
                              <User size={12} />
                              {u.username || u.email || `User #${u.id}`}
                            </span>
                          ))}
                        </div>
                      )}

                      {task.meeting_id && (
                        <span className="flex items-center gap-1.5 bg-brand-maroon/5 text-brand-maroon px-2 py-1 rounded-md font-medium">
                          <Video size={12} />
                          <span className="truncate max-w-37.5 sm:max-w-50">{getMeetingName(task.meeting_id)}</span>
                        </span>
                      )}

                      {task.due_date && (
                        <span className={`flex items-center gap-1.5 px-2 py-1 rounded-md font-medium ${isCompleted ? 'bg-slate-100 text-slate-500' : new Date(task.due_date) < new Date() ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-600'}`}>
                          <Calendar size={12} />
                          {new Date(task.due_date).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                    {isAdmin && (
                      <>
                        <button 
                          onClick={() => openEditModal(task)}
                          className="p-2 text-slate-400 hover:text-brand-maroon hover:bg-brand-maroon/10 rounded-lg transition-colors shrink-0"
                          title="Edit task"
                        >
                          <Edit2 size={18} />
                        </button>
                        <button 
                          onClick={() => setTaskToDelete(task)}
                          className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors shrink-0"
                          title="Delete task"
                        >
                          <Trash2 size={18} />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {isCreateModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white max-w-lg w-full rounded-2xl shadow-xl overflow-hidden animate-in zoom-in-95 duration-200 flex flex-col">
            <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between shrink-0">
              <div>
                <h3 className="font-serif text-xl font-bold text-slate-900">New Action Item</h3>
              </div>
              <button onClick={() => setIsCreateModalOpen(false)} className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-full transition-colors"><X size={20} /></button>
            </div>

            <div className="p-6 overflow-y-auto">
              <form id="createTaskForm" onSubmit={handleCreateTask} className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Task Description *</label>
                  <input type="text" required value={newTaskText} onChange={(e) => setNewTaskText(e.target.value)} placeholder="e.g., Update the architecture diagram" className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold transition-colors" />
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                  <div className="sm:col-span-2">
                    <label className="block text-sm font-medium text-slate-700 mb-2">Related Meeting *</label>
                    <select required value={newTaskMeetingId} onChange={(e) => setNewTaskMeetingId(e.target.value)} className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold transition-colors text-slate-700 bg-white">
                      <option value="" disabled>Select a meeting...</option>
                      {meetings.map(m => <option key={m.id} value={m.id}>{m.title}</option>)}
                    </select>
                  </div>
                  
                  <div className="sm:col-span-2">
                    <label className="block text-sm font-medium text-slate-700 mb-2">Assign To (Select multiple)</label>
                    <div className="w-full max-h-32 overflow-y-auto px-4 py-2 rounded-xl border border-slate-200 bg-slate-50 space-y-2">
                      {members.map(m => (
                        <label key={m.id} className="flex items-center gap-3 cursor-pointer p-1 hover:bg-slate-100 rounded">
                          <input 
                            type="checkbox" 
                            checked={newTaskAssigneeIds.includes(String(m.id))}
                            onChange={() => handleToggleAssignee(String(m.id), false)}
                            className="w-4 h-4 rounded border-slate-300 text-brand-maroon focus:ring-brand-maroon cursor-pointer"
                          />
                          <span className="text-sm font-medium text-slate-700">{m.username || m.email || `User #${m.id}`}</span>
                        </label>
                      ))}
                      {members.length === 0 && <span className="text-sm text-slate-500">No members available.</span>}
                    </div>
                  </div>

                  <div className="sm:col-span-2">
                    <label className="block text-sm font-medium text-slate-700 mb-2">Due Date (Optional)</label>
                    <input type="date" value={newTaskDueDate} onChange={(e) => setNewTaskDueDate(e.target.value)} className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold transition-colors text-slate-700" />
                  </div>
                </div>
              </form>
            </div>

            <div className="px-6 py-4 border-t border-slate-100 bg-slate-50 flex items-center justify-end gap-3 shrink-0">
              <button type="button" onClick={() => setIsCreateModalOpen(false)} className="px-5 py-2.5 text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors" disabled={isCreating}>Cancel</button>
              <button form="createTaskForm" type="submit" disabled={isCreating || !newTaskText.trim() || !newTaskMeetingId} className="px-6 py-2.5 bg-brand-maroon text-white text-sm font-medium rounded-lg hover:bg-brand-gold transition-colors flex items-center gap-2 disabled:opacity-70">
                {isCreating ? <><Loader2 size={16} className="animate-spin" /> Saving...</> : "Create Task"}
              </button>
            </div>
          </div>
        </div>
      )}

      {isUpdateModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white max-w-lg w-full rounded-2xl shadow-xl overflow-hidden animate-in zoom-in-95 duration-200 flex flex-col">
            <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between shrink-0">
              <div>
                <h3 className="font-serif text-xl font-bold text-slate-900">Edit Action Item</h3>
              </div>
              <button onClick={() => setIsUpdateModalOpen(false)} className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-full transition-colors"><X size={20} /></button>
            </div>

            <div className="p-6 overflow-y-auto">
              <form id="updateTaskForm" onSubmit={handleUpdateTaskDetails} className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Task Description *</label>
                  <input type="text" required value={editTaskData.task_text} onChange={(e) => setEditTaskData({...editTaskData, task_text: e.target.value})} className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold transition-colors" />
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                  <div className="sm:col-span-2">
                    <label className="block text-sm font-medium text-slate-700 mb-2">Related Meeting *</label>
                    <select required value={editTaskData.meeting_id} onChange={(e) => setEditTaskData({...editTaskData, meeting_id: e.target.value})} className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold transition-colors text-slate-700 bg-white">
                      <option value="" disabled>Select a meeting...</option>
                      {meetings.map(m => <option key={m.id} value={m.id}>{m.title}</option>)}
                    </select>
                  </div>
                  
                  <div className="sm:col-span-2">
                    <label className="block text-sm font-medium text-slate-700 mb-2">Assign To (Select multiple)</label>
                    <div className="w-full max-h-32 overflow-y-auto px-4 py-2 rounded-xl border border-slate-200 bg-slate-50 space-y-2">
                      {members.map(m => (
                        <label key={m.id} className="flex items-center gap-3 cursor-pointer p-1 hover:bg-slate-100 rounded">
                          <input 
                            type="checkbox" 
                            checked={editTaskData.assignee_ids.includes(String(m.id))}
                            onChange={() => handleToggleAssignee(String(m.id), true)}
                            className="w-4 h-4 rounded border-slate-300 text-brand-maroon focus:ring-brand-maroon cursor-pointer"
                          />
                          <span className="text-sm font-medium text-slate-700">{m.username || m.email || `User #${m.id}`}</span>
                        </label>
                      ))}
                      {members.length === 0 && <span className="text-sm text-slate-500">No members available.</span>}
                    </div>
                  </div>

                  <div className="sm:col-span-2">
                    <label className="block text-sm font-medium text-slate-700 mb-2">Due Date (Optional)</label>
                    <input type="date" value={editTaskData.due_date} onChange={(e) => setEditTaskData({...editTaskData, due_date: e.target.value})} className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:border-brand-gold focus:ring-1 focus:ring-brand-gold transition-colors text-slate-700" />
                  </div>
                </div>
              </form>
            </div>

            <div className="px-6 py-4 border-t border-slate-100 bg-slate-50 flex items-center justify-end gap-3 shrink-0">
              <button type="button" onClick={() => setIsUpdateModalOpen(false)} className="px-5 py-2.5 text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors" disabled={isUpdating}>Cancel</button>
              <button form="updateTaskForm" type="submit" disabled={isUpdating || !editTaskData.task_text.trim() || !editTaskData.meeting_id} className="px-6 py-2.5 bg-brand-maroon text-white text-sm font-medium rounded-lg hover:bg-brand-gold transition-colors flex items-center gap-2 disabled:opacity-70">
                {isUpdating ? <><Loader2 size={16} className="animate-spin" /> Updating...</> : "Save Changes"}
              </button>
            </div>
          </div>
        </div>
      )}

      {taskToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white max-w-md w-full rounded-2xl shadow-xl overflow-hidden animate-in zoom-in-95 duration-200">
            <div className="p-6">
              <div className="w-12 h-12 rounded-full bg-red-100 text-red-600 flex items-center justify-center mb-4">
                <AlertTriangle size={24} />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-2">Delete Action Item?</h3>
              <p className="text-slate-500 mb-6">Are you sure you want to delete <span className="font-semibold text-slate-700">&quot;{taskToDelete.task_text}&quot;</span>? This action cannot be undone.</p>
              <div className="flex gap-3 w-full">
                <button onClick={() => setTaskToDelete(null)} disabled={isDeleting} className="flex-1 px-4 py-2.5 border border-slate-200 text-slate-700 font-medium rounded-xl hover:bg-slate-50 transition-colors">Cancel</button>
                <button onClick={executeDelete} disabled={isDeleting} className="flex-1 px-4 py-2.5 bg-red-600 text-white font-medium rounded-xl hover:bg-red-700 transition-colors flex items-center justify-center gap-2">
                  {isDeleting ? <Loader2 size={18} className="animate-spin" /> : "Delete Task"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}