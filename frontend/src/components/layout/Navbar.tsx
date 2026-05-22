"use client";

import Link from "next/link";
import Image from "next/image";
import { useState, useEffect } from "react";
import { Menu, X } from "lucide-react";

export default function Navbar() {
  const [activeSection, setActiveSection] = useState("");
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    setTimeout(() => {
      setIsLoggedIn(!!localStorage.getItem("token"));
    }, 0);
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      const sections = ["features", "about", "contact"];
      let current = "";

      for (const section of sections) {
        const element = document.getElementById(section);
        if (element) {
          const rect = element.getBoundingClientRect();
          if (rect.top <= 150 && rect.bottom >= 150) {
            current = section;
          }
        }
      }
      setActiveSection(current);
    };

    window.addEventListener("scroll", handleScroll);
    handleScroll(); 

    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navLinks = [
    { name: "Features", href: "#features", id: "features" },
    { name: "About", href: "#about", id: "about" },
    { name: "Contact", href: "#contact", id: "contact" },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-gray-100 bg-white">
      <div className="container relative mx-auto px-6 h-24 flex items-center justify-between">
        
        {/* Logo Section */}
        <Link href="/" className="flex items-center gap-3">
          <Image 
            src="/LughaCap_Icon.png" 
            alt="LughaCap Icon" 
            width={64} 
            height={64} 
            className="object-contain"
          />
          <Image 
            src="/LughaCap_Name.png" 
            alt="LughaCap Name" 
            width={180} 
            height={60} 
            className="object-contain mt-2" 
          />
        </Link>

        {/* Center Navigation Links */}
        <nav className="hidden lg:flex absolute z-10 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 items-center gap-10 text-lg font-medium">
          {navLinks.map((link) => {
            const isActive = activeSection === link.id;
            return (
              <a 
                key={link.id}
                href={link.href} 
                className={`relative py-2 transition-colors duration-300 group cursor-pointer ${
                  isActive ? "text-brand-maroon" : "text-gray-600 hover:text-brand-maroon"
                }`}
              >
                {link.name}
                <span 
                  className={`absolute left-0 bottom-0 h-0.5 bg-brand-maroon transition-all duration-300 ease-out ${
                    isActive ? "w-full" : "w-0 group-hover:w-full"
                  }`}
                />
              </a>
            );
          })}
        </nav>

        {/* Right Side: Desktop Auth Buttons & Mobile Hamburger Toggle */}
        <div className="flex items-center gap-4">
          
          {isLoggedIn ? (
            <Link 
              href="/dashboard" 
              className="hidden lg:block bg-brand-maroon text-white border-2 border-brand-maroon px-6 py-2 rounded-md font-medium hover:bg-brand-gold hover:border-brand-gold transition-all duration-300 shadow-sm"
            >
              Go to Dashboard
            </Link>
          ) : (
            <>
              <Link 
                href="/signin" 
                className="hidden lg:block bg-white text-brand-maroon border-2 border-brand-maroon px-6 py-2 rounded-md font-medium hover:border-brand-gold hover:text-brand-gold transition-all duration-300 shadow-sm"
              >
                Sign In
              </Link>
              <Link 
                href="/signup" 
                className="hidden lg:block bg-brand-maroon text-white border-2 border-brand-maroon px-6 py-2 rounded-md font-medium hover:bg-brand-gold hover:border-brand-gold transition-all duration-300 shadow-sm"
              >
                Sign Up Now
              </Link>
            </>
          )}

          {/* Mobile Menu Toggle Button */}
          <button 
            className="lg:hidden p-2 text-brand-maroon hover:text-brand-gold transition-colors"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-label="Toggle menu"
          >
            {isMobileMenuOpen ? <X size={28} /> : <Menu size={28} />}
          </button>
        </div>
      </div>

      {/* Mobile Slide-Down Menu */}
      <div 
        className={`absolute top-24 left-0 w-full bg-white border-b border-gray-100 shadow-xl transition-all duration-300 ease-in-out lg:hidden overflow-hidden ${
          isMobileMenuOpen ? "max-h-125 opacity-100" : "max-h-0 opacity-0"
        }`}
      >
        <div className="flex flex-col items-center py-8 gap-4">
          {navLinks.map((link) => {
            const isActive = activeSection === link.id;
            return (
              <a
                key={link.id}
                href={link.href}
                onClick={() => setIsMobileMenuOpen(false)}
                className={`text-lg font-medium transition-colors duration-300 ${
                  isActive ? "text-brand-maroon" : "text-gray-600 hover:text-brand-maroon"
                }`}
              >
                {link.name}
              </a>
            );
          })}
          
         {/* Mobile Auth Buttons Container */}
          <div className="flex flex-col w-full px-12 gap-3 mt-4">
            {isLoggedIn ? (
              <Link 
                href="/dashboard" 
                onClick={() => setIsMobileMenuOpen(false)}
                className="w-full text-center bg-brand-maroon text-white border-2 border-brand-maroon py-3 rounded-md font-medium hover:bg-brand-gold hover:border-brand-gold transition-all duration-300 shadow-sm"
              >
                Go to Dashboard
              </Link>
            ) : (
              <>
                <Link 
                  href="/signin" 
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="w-full text-center bg-white text-brand-maroon border-2 border-brand-maroon py-3 rounded-md font-medium hover:border-brand-gold hover:text-brand-gold transition-all duration-300 shadow-sm"
                >
                  Sign In
                </Link>
                <Link 
                  href="/signup" 
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="w-full text-center bg-brand-maroon text-white border-2 border-brand-maroon py-3 rounded-md font-medium hover:bg-brand-gold hover:border-brand-gold transition-all duration-300 shadow-sm"
                >
                  Sign Up Now
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}