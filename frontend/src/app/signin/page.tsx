"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { ArrowLeft, Loader2 } from "lucide-react";
import { apiFetch } from "@/lib/api";

interface LoginResponse {
  data?: {
    token?: string;
  };
  user?: {
    username?: string;
  };
  token?: string;
  username?: string;
}

interface Company {
  id: number;
}

type CompaniesResponse = { data: Company[] } | Company[];

export default function SignIn() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    try {
  // Log In to get the token
  const loginData = await apiFetch<LoginResponse>("/auth/login", {
    method: "POST",
    data: { email: formData.email, password: formData.password },
  });

  const token = loginData?.data?.token || loginData?.token;
  if (!token) throw new Error("Verification failed. Please try again.");

  localStorage.setItem("token", token);
  
  const fetchedName = loginData?.user?.username || loginData?.username;
  if (fetchedName) localStorage.setItem("username", fetchedName);

  // Fetch the user's companies
  const companiesData = await apiFetch<CompaniesResponse>("/companies", {
    method: "GET",
  });
  const finalCompanies = Array.isArray(companiesData)
    ? companiesData
    : companiesData?.data || [];

  if (finalCompanies.length > 0) {
    // Existing User: Save company ID
    localStorage.setItem("companyId", finalCompanies[0].id.toString());
  } else {
    // New User: clear companyId to trigger the Choice Page in Dashboard
    localStorage.removeItem("companyId");
    console.log("New user detected: No companies assigned.");
  }

  // Navigate to Dashboard
  router.push("/dashboard");
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("We ran into an unexpected issue. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-slate-100 p-4 md:p-8">
      
      {/* Centered Container Card */}
      <div className="w-full max-w-6xl h-full max-h-225 flex bg-white rounded-3xl overflow-hidden shadow-[0_20px_60px_-15px_rgba(0,0,0,0.1)]">
        
        {/* LEFT SIDE: The Action (Form) */}
        <div className="w-full lg:w-1/2 flex flex-col relative px-8 sm:px-12 md:px-16 py-12 justify-center">
          
          {/* Back to Home Button */}
          <Link 
            href="/" 
            className="absolute top-8 left-8 sm:left-12 flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-brand-maroon transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to home
          </Link>

          <div className="w-full max-w-md mx-auto">
            {/* Form Header: Logo & Name */}
            <div className="mb-10 text-center flex flex-col items-center">
              <Image 
                src="/LughaCap_Icon.png" 
                alt="LughaCap Icon" 
                width={70} 
                height={70} 
                className="object-contain"
              />
              <Image 
                src="/LughaCap_Name.png" 
                alt="LughaCap Name" 
                width={180} 
                height={80} 
                className="object-contain" 
              />
            </div>

            {/* Email / Password Form */}
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div className="relative">
                <input 
                  type="email" 
                  id="email" 
                  value={formData.email}
                  onChange={handleChange}
                  className="peer block w-full px-4 pb-2 pt-6 text-slate-900 bg-slate-50/50 border border-slate-200 rounded-xl appearance-none focus:outline-none focus:ring-0 focus:border-brand-gold transition-colors" 
                  placeholder=" " 
                  required 
                  disabled={isLoading}
                />
                <label 
                  htmlFor="email" 
                  className="absolute text-slate-400 duration-300 transform -translate-y-3 scale-75 top-4 z-10 origin-left left-4 peer-placeholder-shown:scale-100 peer-placeholder-shown:translate-y-0 peer-focus:scale-75 peer-focus:-translate-y-3 peer-focus:text-brand-gold cursor-text"
                >
                  Email address
                </label>
              </div>

              <div className="relative">
                <input 
                  type="password" 
                  id="password" 
                  value={formData.password}
                  onChange={handleChange}
                  className="peer block w-full px-4 pb-2 pt-6 text-slate-900 bg-slate-50/50 border border-slate-200 rounded-xl appearance-none focus:outline-none focus:ring-0 focus:border-brand-gold transition-colors" 
                  placeholder=" " 
                  required 
                  disabled={isLoading}
                />
                <label 
                  htmlFor="password" 
                  className="absolute text-slate-400 duration-300 transform -translate-y-3 scale-75 top-4 z-10 origin-left left-4 peer-placeholder-shown:scale-100 peer-placeholder-shown:translate-y-0 peer-focus:scale-75 peer-focus:-translate-y-3 peer-focus:text-brand-gold cursor-text"
                >
                  Password
                </label>
              </div>

              {/* Error Message Display */}
              {error && (
                <div className="text-red-500 text-sm font-medium mt-2 bg-red-50 p-3 rounded-lg border border-red-100">
                  {error}
                </div>
              )}

              <button 
                type="submit" 
                disabled={isLoading}
                className="w-full flex items-center justify-center bg-brand-maroon text-white rounded-xl px-6 py-4 font-medium hover:bg-brand-gold transition-all duration-300 hover:shadow-[0_8px_25px_rgba(200,178,148,0.3)] hover:-translate-y-0.5 mt-4 disabled:opacity-70 disabled:hover:translate-y-0 disabled:hover:shadow-none"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Signing in...
                  </>
                ) : (
                  "Sign In"
                )}
              </button>
            </form>

            <p className="mt-8 text-center text-sm text-slate-500">
              Don&apos;t have an account?{" "}
              <Link href="/signup" className="font-semibold text-brand-maroon hover:text-brand-gold transition-colors">
                Sign up
              </Link>
            </p>
          </div>
        </div>

        {/* RIGHT SIDE: The Brand (Pattern contained in card) */}
        <div className="hidden lg:flex lg:w-1/2 bg-brand-maroon relative items-center justify-center overflow-hidden">
          <Image 
            src="/Saas.png" 
            alt="LughaCap Pattern"
            fill
            className="object-cover opacity-80"
            priority
          />
          <div className="absolute inset-0 bg-brand-maroon/20"></div>
          
          {/* Ambient Glows to maintain premium feel inside the card */}
          <div className="absolute top-1/4 -left-10 w-64 h-64 bg-brand-gold/30 rounded-full blur-[80px] pointer-events-none"></div>
          <div className="absolute bottom-1/4 -right-10 w-64 h-64 bg-white/10 rounded-full blur-[80px] pointer-events-none"></div>
        </div>

      </div>
    </div>
  );
}