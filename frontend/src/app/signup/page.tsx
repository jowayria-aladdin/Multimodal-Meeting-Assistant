"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { ArrowLeft, Loader2 } from "lucide-react";
import { apiFetch } from "@/lib/api"; 

type LoginResponse = {
  data?: {
    token?: string;
  };
  token?: string;
  user?: {
    username?: string;
  };
  username?: string;
};

export default function SignUp() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  
  const [formData, setFormData] = useState({
    username: "",
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
      // Register the User using apiFetch
      await apiFetch("/auth/register", {
        method: "POST",
        data: {
          username: formData.username,
          email: formData.email,
          password: formData.password,
        },
      });
      
      // Log In immediately to get the JWT token
      const loginData = await apiFetch<LoginResponse>("/auth/login", {
        method: "POST",
        data: {
          email: formData.email,
          password: formData.password,
        },
      });

      // Extract the token safely
      const token = loginData?.data?.token || loginData?.token;

      if (!token) {
        throw new Error("Account created successfully, but we couldn't log you in automatically. Please sign in.");
      }
      
      // Store credentials for protected routes
      localStorage.setItem("token", token); 
      
      // Smart Fallback: Use backend name, or default to what they just typed
      const fetchedName = loginData?.user?.username || loginData?.username || formData.username;
      if (fetchedName) {
        localStorage.setItem("username", fetchedName);
      }
      
      // Success! Redirect to the dashboard
      router.push("/dashboard");

    } catch (err) {
      if (err instanceof Error) {
        // apiFetch passes the exact backend error message through
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
      
      <div className="w-full max-w-6xl h-full max-h-237.5 flex bg-white rounded-3xl overflow-hidden shadow-[0_20px_60px_-15px_rgba(0,0,0,0.1)]">
        
        {/* LEFT SIDE: The Action (Form) */}
        <div className="w-full lg:w-1/2 flex flex-col relative px-8 sm:px-12 md:px-16 py-12 justify-center overflow-y-auto">
          
          {/* Back to Home Button */}
          <Link 
            href="/" 
            className="absolute top-8 left-8 sm:left-12 flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-brand-maroon transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to home
          </Link>

          <div className="w-full max-w-md mx-auto mt-8">
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
            
            {/* Form */}
            <form className="space-y-4" onSubmit={handleSubmit}>
              
              {/* Row 1: Username Input */}
              <div className="relative">
                <input 
                  type="text" 
                  id="username" 
                  value={formData.username}
                  onChange={handleChange}
                  className="peer block w-full px-4 pb-2 pt-6 text-slate-900 bg-slate-50/50 border border-slate-200 rounded-xl appearance-none focus:outline-none focus:ring-0 focus:border-brand-gold transition-colors" 
                  placeholder=" " 
                  required 
                  disabled={isLoading}
                />
                <label 
                  htmlFor="username" 
                  className="absolute text-slate-400 duration-300 transform -translate-y-3 scale-75 top-4 z-10 origin-left left-4 peer-placeholder-shown:scale-100 peer-placeholder-shown:translate-y-0 peer-focus:scale-75 peer-focus:-translate-y-3 peer-focus:text-brand-gold cursor-text"
                >
                  Username
                </label>
              </div>

              {/* Row 2: Email Input */}
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

              {/* Row 3: Password Input */}
              <div className="relative mb-6">
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
                    Creating Account...
                  </>
                ) : (
                  "Sign Up"
                )}
              </button>
            </form>

            <p className="mt-8 text-center text-sm text-slate-500">
              Already have an account?{" "}
              <Link href="/signin" className="font-semibold text-brand-maroon hover:text-brand-gold transition-colors">
                Sign in
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
          
          <div className="absolute top-1/4 -left-10 w-64 h-64 bg-brand-gold/30 rounded-full blur-[80px] pointer-events-none"></div>
          <div className="absolute bottom-1/4 -right-10 w-64 h-64 bg-white/10 rounded-full blur-[80px] pointer-events-none"></div>
        </div>

      </div>
    </div>
  );
}