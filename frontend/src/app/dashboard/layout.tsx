"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const [isAuthorized, setIsAuthorized] = useState(false);

  useEffect(() => {
    const verifySecureConnection = async () => {
      const token = localStorage.getItem("token");
      
      // dashboard page itself handle the missing companyId.
      if (!token) {
        router.replace("/signin");
      } else {
        setIsAuthorized(true);
      }
    };

    verifySecureConnection();
  }, [router]);

  if (!isAuthorized) {
    return (
      <div className="h-screen w-full flex flex-col items-center justify-center bg-slate-50">
        <Loader2 className="w-10 h-10 text-brand-maroon animate-spin mb-4" />
        <p className="text-slate-500 font-medium">Verifying secure connection...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col md:flex-row">
      <main className="flex-1 w-full overflow-y-auto">
        {children}
      </main>
    </div>
  );
}