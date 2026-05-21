
export default function Hero() {
  return (
    <section className="relative w-full min-h-[90vh] flex items-center bg-transparent overflow-hidden">
      {/* Ambient Background Glows for Depth */}
      <div className="absolute top-0 left-0 w-125 h-125 bg-brand-gold/10 rounded-full blur-[120px] -translate-x-1/2 -translate-y-1/2 pointer-events-none"></div>
      
      <div className="absolute bottom-0 right-0 w-150 h-150 bg-brand-maroon/5 rounded-full blur-[150px] translate-x-1/3 -translate-y-10 pointer-events-none"></div>

      <div className="container relative mx-auto px-6 py-12 md:py-24 z-10 w-full">
        <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_1fr] gap-12 lg:gap-8 items-center w-full">
          
          {/* Left Side: Text & Call to Action */}
          <div className="flex flex-col items-start text-left group/text min-w-0">
            
            <h1 className="font-serif text-6xl md:text-7xl lg:text-[5.5rem] xl:text-8xl font-bold mb-4 tracking-tight text-brand-maroon drop-shadow-sm">
              Lugha<span className="text-brand-gold">C</span>ap<span className="text-brand-gold">;</span>
            </h1>
            
            <p className="font-sans text-xl md:text-3xl lg:text-4xl text-slate-600 font-light leading-relaxed mb-10 max-w-xl">
              For The <span className="font-serif font-semibold text-brand-maroon italic">Words</span> You Speak, <br className="hidden md:block" /> 
              And The <span className="font-serif font-semibold text-brand-gold italic">Signs</span> You Show.
            </p>
            
            <div className="relative inline-block group/btn">
              
              {/* The original button styling */}
              <div className="absolute inset-0 bg-brand-gold rounded-lg blur-md opacity-20 group-hover/btn:opacity-60 transition duration-500"></div>
              
              {/* Pure Download Link */}
              <a 
                href="/LughaCap_Extension.zip" 
                download="LughaCap_Extension.zip" 
                className="relative flex items-center justify-center gap-3 bg-brand-maroon text-white px-10 py-4 rounded-lg text-lg font-medium transition-all duration-300 ease-out hover:bg-brand-gold hover:scale-[1.02] hover:-translate-y-1 border border-transparent shadow-xl w-full"
              >
                <span>Download Extension Now!</span>
                <svg className="w-5 h-5 transition-transform duration-300 group-hover/btn:translate-x-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
              </a>

            </div>
          </div>

          {/* Right Side: The Animation */}
          <div className="relative w-full flex justify-center lg:justify-end min-w-0">
            <div className="relative w-full max-w-md lg:max-w-lg xl:max-w-xl aspect-square rounded-3xl transition-all duration-500 hover:shadow-[0_20px_50px_rgba(88,24,31,0.12)] hover:-translate-y-2 border border-slate-100 bg-white/50 backdrop-blur-sm p-2 group/video cursor-default">
               <div className="absolute top-0 left-0 w-12 h-12 border-t-4 border-l-4 border-brand-gold/40 rounded-tl-3xl transition-all duration-500 group-hover/video:border-brand-gold group-hover/video:w-16 group-hover/video:h-16"></div>
               <div className="absolute bottom-0 right-0 w-12 h-12 border-b-4 border-r-4 border-brand-maroon/40 rounded-br-3xl transition-all duration-500 group-hover/video:border-brand-maroon group-hover/video:w-16 group-hover/video:h-16"></div>

               <div className="w-full h-full overflow-hidden rounded-2xl bg-white relative z-10">
                 <video 
                   src="/LughaCap_Animation.mp4" 
                   autoPlay 
                   loop 
                   muted 
                   playsInline
                   className="w-full h-full object-contain transition-transform duration-700 group-hover/video:scale-105"
                 />
               </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}