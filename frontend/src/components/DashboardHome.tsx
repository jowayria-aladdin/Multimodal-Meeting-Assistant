import { Loader2, Video, Calendar } from "lucide-react";
import Link from "next/link";

interface CurrentUser { id?: string | number; email?: string; username?: string; }
interface Participant { id?: number | string; user_id?: number | string; userId?: number | string; email?: string; user?: { email?: string; }; }
interface TaskAssignee { id?: number | string; user_id?: number | string; userId?: number | string; email?: string; user?: { email?: string; }; }

interface Meeting { 
  id: string | number; 
  title: string; 
  created_at?: string; 
  summary?: string; 
  status?: string; 
  participants?: Participant[]; 
  meetingParticipants?: Participant[]; 
  MeetingParticipants?: Participant[]; 
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

interface Props {
  meetings: Meeting[];
  tasks: Task[];
  memberCount: number;
  isLoading: boolean;
  onUploadClick: () => void;
  isAdminOrOwner: boolean;
  currentUserData: CurrentUser | null;
}

export default function DashboardHome({ meetings, tasks, memberCount, isLoading, isAdminOrOwner, currentUserData }: Props) {
  
  // RBAC FILTER FOR TASKS: Show all for Admins, otherwise only show assigned tasks
  const processedTasks = tasks.filter(task => {
    if (isAdminOrOwner) return true;
    if (!currentUserData) return false;

    const assigneesList = task.taskAssignees || task.taskassignees || task.assignees || task.TaskAssignees || [];
    return assigneesList.some((a: TaskAssignee) => 
      String(a.user_id) === String(currentUserData.id) ||
      String(a.userId) === String(currentUserData.id) ||
      String(a.id) === String(currentUserData.id) ||
      (a.email && currentUserData.email && a.email.toLowerCase() === currentUserData.email.toLowerCase()) ||
      (a.user?.email && currentUserData.email && a.user.email.toLowerCase() === currentUserData.email.toLowerCase())
    );
  });

  //RBAC FILTER FOR MEETINGS: Show all for Admins, otherwise only show meetings participated in
  const processedMeetings = meetings.filter(meeting => {
    if (isAdminOrOwner) return true;
    if (!currentUserData) return false;

    const participantsArray = meeting.participants || meeting.meetingParticipants || meeting.MeetingParticipants || [];
    return participantsArray.some((p: Participant) => 
      String(p.user_id) === String(currentUserData.id) ||
      String(p.userId) === String(currentUserData.id) ||
      String(p.id) === String(currentUserData.id) ||
      (p.email && currentUserData.email && p.email.toLowerCase() === currentUserData.email.toLowerCase()) ||
      (p.user?.email && currentUserData.email && p.user.email.toLowerCase() === currentUserData.email.toLowerCase())
    );
  });

  const pendingTasksCount = processedTasks.filter(task => !task.status || task.status === "TODO" || task.status === "IN_PROGRESS").length;
  const totalMeetingsCount = processedMeetings.length;

  return (
    <>
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
          <span className="text-slate-500 font-medium mb-1">Total Meetings</span>
          <span className="text-3xl font-serif font-bold text-brand-maroon">
            {isLoading ? <Loader2 size={24} className="animate-spin text-slate-300 mt-1" /> : totalMeetingsCount}
          </span>
        </div>
        
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
          <span className="text-slate-500 font-medium mb-1">Pending Tasks</span>
          <span className="text-3xl font-serif font-bold text-brand-maroon">
            {isLoading ? <Loader2 size={24} className="animate-spin text-slate-300 mt-1" /> : pendingTasksCount}
          </span>
        </div>
        
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col">
          <span className="text-slate-500 font-medium mb-1">Team Members</span>
          <span className="text-3xl font-serif font-bold text-brand-maroon">
            {isLoading ? <Loader2 size={24} className="animate-spin text-slate-300 mt-1" /> : memberCount}
          </span>
        </div>
      </div>

      {/* Recent Activity Table */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-6 py-5 border-b border-slate-200">
          <h3 className="font-serif text-xl font-bold text-slate-900">Recent Activity</h3>
        </div>
        
        {isLoading ? (
            <div className="p-12 flex justify-center"><Loader2 className="w-8 h-8 text-slate-300 animate-spin" /></div>
        ) : processedMeetings.length === 0 ? (
          <div className="p-12 flex flex-col items-center justify-center text-center">
            <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-4">
              <Video className="w-8 h-8 text-slate-300" />
            </div>
            <h4 className="text-lg font-medium text-slate-900 mb-2">No meetings analyzed yet</h4>
            <p className="text-slate-500 max-w-sm mb-6">
              Upload your first audio or video recording to see AI insights, transcripts, and automated tasks here.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {processedMeetings.slice(0, 4).map((meeting) => (
              <div key={meeting.id} className="p-6 hover:bg-slate-50 transition-colors flex items-center justify-between group">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-brand-maroon/10 text-brand-maroon rounded-lg flex items-center justify-center">
                    <Video size={20} />
                  </div>
                  <div>
                    <h4 className="font-medium text-slate-900 group-hover:text-brand-maroon transition-colors">{meeting.title}</h4>
                    <div className="flex items-center gap-2 text-sm text-slate-500 mt-1">
                      <Calendar size={14} />
                      <span>{meeting.created_at ? new Date(meeting.created_at).toLocaleDateString() : "Recently processed"}</span>
                    </div>
                  </div>
                </div>
                <Link 
                  href={`/dashboard/meeting/${meeting.id}`} 
                  className="text-sm font-medium text-brand-maroon hover:text-brand-gold transition-colors px-4 py-2 rounded-lg hover:bg-brand-maroon/5 border border-transparent hover:border-brand-maroon/10"
                >
                  View Analysis
                </Link>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}