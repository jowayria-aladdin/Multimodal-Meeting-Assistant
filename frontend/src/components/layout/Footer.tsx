import Link from "next/link";
import { Mail, MapPin } from "lucide-react";

export default function Footer() {
  return (
    <footer id="contact" className="bg-brand-maroon text-white pt-20 pb-10 border-t-4 border-brand-gold">
      <div className="container mx-auto px-6">
        
        {/* Top Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 lg:gap-8 mb-16">
          
          {/* Left Column: Brand & Slogan */}
          <div className="flex flex-col items-start">
            <Link href="/" className="flex items-center gap-3 mb-6">
              {/* Text-based logo to ensure perfect contrast on the dark background */}
              <span className="font-serif text-4xl font-bold tracking-tight text-white drop-shadow-sm">
                Lugha<span className="text-brand-gold">C</span>ap<span className="text-brand-gold">;</span>
              </span>
            </Link>
            <p className="text-white/80 font-light leading-relaxed max-w-sm">
              For The <span className="font-serif italic text-white font-medium">Words</span> You Speak,<br />
              And The <span className="font-serif italic text-brand-gold font-medium">Signs</span> You Show.
            </p>
          </div>

          {/* Middle Column: Quick Links */}
          <div className="flex flex-col items-start md:items-center">
            <div className="flex flex-col space-y-4">
              <h4 className="font-serif text-xl font-bold text-brand-gold mb-2">Quick Links</h4>
              <Link href="#features" className="text-white/70 hover:text-white hover:translate-x-1 transition-all duration-300">
                Our Features
              </Link>
              <Link href="#about" className="text-white/70 hover:text-white hover:translate-x-1 transition-all duration-300">
                How It Works
              </Link>
              {/*  Download Extension */}
              <Link href="#" className="text-white/70 hover:text-white hover:translate-x-1 transition-all duration-300">
                Download Extension
              </Link>
            </div>
          </div>

          {/* Right Column: Contact Info */}
          <div className="flex flex-col items-start md:items-end text-left md:text-right">
            <h4 className="font-serif text-xl font-bold text-brand-gold mb-6">Contact Us</h4>
            <div className="flex flex-col space-y-4">
              <a href="mailto:yugoslavia2212@gmail.com" className="group flex items-center justify-start md:justify-end gap-3 text-white/70 hover:text-white transition-colors duration-300">
                <span className="group-hover:underline">yugoslavia2212@gmail.com</span>
                <Mail className="w-5 h-5 text-brand-gold" />
              </a>
              <div className="flex items-center justify-start md:justify-end gap-3 text-white/70">
                <span>Alexandria, Egypt</span>
                <MapPin className="w-5 h-5 text-brand-gold" />
              </div>
            </div>
          </div>

        </div>

        {/* Bottom Bar: Copyright & Academic Badge */}
        <div className="border-t border-white/10 pt-8 flex flex-col md:flex-row justify-between items-center gap-4 text-sm text-white/50">
          <p>&copy; 2026 LughaCap. All rights reserved.</p>
          <p className="font-light tracking-wide text-center md:text-right">
            Computer and Communication Engineering <br className="block md:hidden"/> Graduation Project, Alexanria University.
          </p>
        </div>

      </div>
    </footer>
  );
}