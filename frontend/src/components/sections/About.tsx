import { Mic, FileText, Sparkles } from "lucide-react";

export default function About() {
  const steps = [
    {
      id: "01",
      title: "Record Your Meeting",
      description: "Start a meeting and let LughaCap capture audio and sign language in real time.",
      icon: Mic,
    },
    {
      id: "02",
      title: "Get Instant Transcripts",
      description: "Receive accurate multilingual transcripts with speaker identification.",
      icon: FileText,
    },
    {
      id: "03",
      title: "Review AI Insights",
      description: "Access smart summaries, extracted tasks, and actionable insights instantly.",
      icon: Sparkles,
    },
  ];

  return (
    // Added scroll-mt-24 so the sticky navbar doesn't cover the title
    <section id="about" className="relative w-full py-24 md:py-32 bg-white overflow-hidden scroll-mt-24">
      <div className="container mx-auto px-6 relative z-10">
        
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-24">
          <h2 className="font-serif text-4xl md:text-5xl font-bold text-brand-maroon mb-4 tracking-tight">
            <span className="text-brand-gold">How</span> It Works
          </h2>
          <p className="text-lg md:text-xl text-slate-500 font-light leading-relaxed">
            Three simple steps to smarter meetings
          </p>
        </div>

        {/* Roadmap Grid Container */}
        <div className="relative max-w-5xl mx-auto">
          
          {/* The Connecting Line (Hidden on mobile, connects the centers of the circles on desktop) */}
          <div className="hidden md:block absolute top-12 left-[16.66%] w-[66.66%] h-0.5 bg-brand-gold/30 -z-10"></div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-16 md:gap-8">
            {steps.map((step, index) => {
              const Icon = step.icon;
              return (
                <div key={index} className="relative flex flex-col items-center text-center group cursor-default">
                  
                  {/* The Main Circle Node */}
                  <div className="relative w-24 h-24 rounded-full bg-white border-2 border-brand-gold flex items-center justify-center mb-8 transition-all duration-500 ease-out group-hover:scale-110 group-hover:shadow-[0_15px_30px_rgba(200,178,148,0.25)] z-10">
                    
                    {/* The Icon inside the circle */}
                    <Icon strokeWidth={1.5} className="w-10 h-10 text-brand-maroon transition-colors duration-500 group-hover:text-brand-gold" />

                    {/* The Number Badge (01, 02, 03) */}
                    <div className="absolute -top-1 -right-2 w-8 h-8 rounded-full bg-brand-maroon border-4 border-white flex items-center justify-center text-[11px] font-bold text-white shadow-sm transition-colors duration-500 group-hover:bg-brand-gold">
                      {step.id}
                    </div>
                  </div>

                  {/* Text Content */}
                  <h3 className="font-serif text-2xl font-bold text-brand-maroon mb-3 transition-colors duration-300">
                    {step.title}
                  </h3>
                  <p className="text-slate-600 font-light leading-relaxed px-4">
                    {step.description}
                  </p>

                </div>
              );
            })}
          </div>

        </div>
      </div>
    </section>
  );
}