import { Languages, BrainCircuit, AppWindow } from "lucide-react";

export default function Features() {
  const features = [
    {
      title: "Bilingual & Sign Language ASR",
      description:
        "Break down communication barriers instantly. LughaCap flawlessly handles Arabic-English code-switching while simultaneously translating Sign Language gestures into fully accessible text.",
      icon: Languages,
    },
    {
      title: "Smart Insights & Task Extraction",
      description:
        "Transform raw conversations into organized knowledge. Our AI utilizes speaker diarization to identify participants, generate concise meeting summaries, and automatically extract actionable tasks so no detail is missed.",
      icon: BrainCircuit,
    },
    {
      title: "Seamless Cross-Platform Capture",
      description:
        "Capture your meetings effortlessly. Operating as a secure web-based platform paired with a custom Chrome Extension, LughaCap attends your meetings directly, capturing both audio and video securely.",
      icon: AppWindow,
    },
  ];

  return (
<section id="features" className="relative w-full pb-24 md:pb-32 pt-10 bg-transparent overflow-hidden scroll-mt-24">
          <div className="container mx-auto px-6 relative z-10">
        
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-20">
          <h2 className="font-serif text-4xl md:text-5xl font-bold text-brand-maroon mb-4 tracking-tight">
            <span className="text-brand-gold">Our</span> Features
          </h2>
          <p className="text-lg md:text-xl text-slate-600 font-light leading-relaxed">
            LughaCap goes beyond simple recording by employing advanced AI models to handle diverse communication styles and ensure fully inclusive documentation.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 lg:gap-12">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div 
                key={index}
                // Changed to border-2, removed translate-y, added scale, and added the gold frame hover
                className="group relative flex flex-col items-center text-center bg-white rounded-2xl p-8 lg:p-10 border-2 border-transparent shadow-[0_8px_30px_rgb(0,0,0,0.04)] transition-all duration-300 hover:scale-[1.02] hover:shadow-[0_20px_40px_rgba(200,178,148,0.2)] hover:border-brand-gold"
              >
                {/* Icon Container with Hover Glow */}
                <div className="w-16 h-16 rounded-xl bg-brand-maroon/5 flex items-center justify-center text-brand-maroon mb-8 transition-colors duration-500 group-hover:bg-brand-gold/10 group-hover:text-brand-gold">
                  <Icon strokeWidth={1.5} className="w-8 h-8" />
                </div>

                {/* Card Text - Title is now brand-maroon */}
                <h3 className="font-serif text-2xl font-bold text-brand-maroon mb-4 transition-colors duration-300 group-hover:text-brand-gold">
                  {feature.title}
                </h3>
                <p className="text-slate-600 font-light leading-relaxed grow">
                  {feature.description}
                </p>

              </div>
            );
          })}
        </div>

      </div>
    </section>
  );
}