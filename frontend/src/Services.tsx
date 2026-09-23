import React, { useEffect, useRef, useState } from 'react';
import { Info, X } from 'lucide-react';

interface ServiceItem {
  number: string;
  title: string;
  description: string;
}

const SERVICES_DATA: ServiceItem[] = [
  {
    number: '01',
    title: 'FREIGHT FORECASTING',
    description:
      'Forecast future dry-bulk freight rates across Handysize, Supramax, Panamax and Capesize vessel classes with uncertainty around each forecast.',
  },
  {
    number: '02',
    title: 'MARKET INTELLIGENCE',
    description:
      'Read freight-market conditions through energy, commodities, foreign exchange and geopolitical signals.',
  },
  {
    number: '03',
    title: 'VESSEL INTELLIGENCE',
    description:
      'Evaluate Handysize, Supramax, Panamax and Capesize vessels for the cargo movement.',
  },
  {
    number: '04',
    title: 'PORT FEASIBILITY',
    description:
      'Evaluate draft, LOA, beam, berth limits, congestion and other destination-port constraints.',
  },
  {
    number: '05',
    title: 'RISK INTELLIGENCE',
    description:
      'Identify market, geopolitical, weather and operational risks that can affect the charter.',
  },
  {
    number: '06',
    title: 'SCENARIO ANALYSIS',
    description:
      'Compare charter-now, wait and flexible strategies using expected freight, cost and risk.',
  },
  {
    number: '07',
    title: 'CHARTER RECOMMENDATION',
    description:
      'Combine forecasts, vessel feasibility, port conditions, risk and cost into a clear chartering decision.',
  },
];

interface ServicesProps {
  onNavigate: (view: 'landing' | 'dashboard' | 'services' | 'vessel-intelligence' | 'idle-intelligence') => void;
}

export default function Services({ onNavigate }: ServicesProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Smooth real-time scroll tracking
  useEffect(() => {
    let animationFrameId: number;

    const handleScroll = () => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const containerTop = -rect.top;
      const totalScrollable = rect.height - window.innerHeight;

      if (totalScrollable <= 0) return;

      const rawProgress = Math.min(1, Math.max(0, containerTop / totalScrollable));
      setScrollProgress(rawProgress);
    };

    const onScroll = () => {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = requestAnimationFrame(handleScroll);
    };

    window.addEventListener('scroll', onScroll, { passive: true });
    handleScroll();

    return () => {
      window.removeEventListener('scroll', onScroll);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  // Map progress across 7 stages (0.00 to 6.00)
  const totalStages = SERVICES_DATA.length;
  const currentStageFloat = scrollProgress * (totalStages - 1);
  const activeIndex = Math.min(totalStages - 1, Math.max(0, Math.round(currentStageFloat)));

  return (
    <div
      ref={containerRef}
      className="relative h-[850vh] bg-[#EAEFF5] text-[#1D3045] selection:bg-[#1D3045] selection:text-white"
    >
      {/* STICKY FULL-VIEWPORT SCENE */}
      <div className="sticky top-0 w-full h-screen overflow-hidden flex flex-col justify-between select-none">
        
        {/* 1. Continuous Pale Cloudy Textured Background */}
        <div
          className="absolute inset-0 w-full h-full pointer-events-none bg-cover bg-center bg-no-repeat"
          style={{
            backgroundImage: 'url(/services-bg.png)',
            backgroundColor: '#EAEFF5',
            opacity: 0.95,
          }}
          aria-hidden="true"
        />

        {/* Soft atmospheric tonal overlay */}
        <div
          className="absolute inset-0 pointer-events-none bg-gradient-to-b from-white/30 via-transparent to-white/40"
          aria-hidden="true"
        />

        {/* 2. Top Navigation Header */}
        <header className="relative z-30 w-full px-6 sm:px-8 md:px-12 pt-8 sm:pt-10 pb-2 flex items-center justify-between text-[#1D3045]">
          
          {/* Desktop Left Cluster */}
          <nav className="hidden lg:flex items-center gap-8 xl:gap-10" aria-label="Main Navigation">
            <button
              onClick={() => onNavigate('landing')}
              className="text-xs tracking-[0.15em] uppercase font-medium hover:opacity-70 transition-opacity"
            >
              FICOS
            </button>
            <button
              onClick={() => onNavigate('landing')}
              className="text-xs tracking-[0.15em] uppercase font-medium opacity-70 hover:opacity-100 transition-opacity"
            >
              FREIGHT INTELLIGENCE
            </button>
            <button
              onClick={() => onNavigate('vessel-intelligence')}
              className="text-xs tracking-[0.15em] uppercase font-medium opacity-70 hover:opacity-100 transition-opacity"
            >
              VESSEL INTELLIGENCE
            </button>
            <button
              onClick={() => onNavigate('idle-intelligence')}
              className="text-xs tracking-[0.15em] uppercase font-medium opacity-70 hover:opacity-100 transition-opacity"
            >
              IDLE INTELLIGENCE
            </button>
            <span className="relative text-xs tracking-[0.15em] uppercase font-medium">
              HOW IT WORKS
              <span className="absolute -bottom-3 left-0 right-0 h-[2px] w-full bg-[#1D3045]" />
            </span>
            <button
              onClick={() => onNavigate('dashboard')}
              className="text-xs tracking-[0.15em] uppercase font-medium opacity-70 hover:opacity-100 transition-opacity"
            >
              DASHBOARD
            </button>
            <button
              onClick={() => onNavigate('landing')}
              className="text-xs tracking-[0.15em] uppercase font-medium opacity-70 hover:opacity-100 transition-opacity"
            >
              ABOUT
            </button>
          </nav>

          {/* Mobile brand fallback */}
          <div className="lg:hidden flex items-center gap-4">
            <button
              onClick={() => onNavigate('landing')}
              className="text-base font-bold tracking-[0.15em]"
            >
              FICOS
            </button>
            <span className="text-xs tracking-[0.15em] uppercase font-semibold border-b-2 border-[#1D3045] pb-0.5">
              SERVICES
            </span>
          </div>

          {/* Right Cluster */}
          <div className="flex items-center gap-6">
            <button
              onClick={() => onNavigate('dashboard')}
              className="text-xs tracking-[0.2em] uppercase font-medium hover:opacity-80 transition-opacity"
            >
              OPEN DASHBOARD
            </button>

            <button
              onClick={() => onNavigate('landing')}
              className="hidden sm:inline text-xs tracking-[0.2em] uppercase font-medium opacity-70 hover:opacity-100 transition-opacity"
            >
              ← BACK TO SITE
            </button>

            <div className="hidden sm:flex items-center gap-2">
              <span className="text-xs tracking-[0.2em] uppercase font-medium">
                INSIGHTS
              </span>
              <div className="w-5 h-5 rounded-full bg-[#1D3045] text-white flex items-center justify-center">
                <Info size={10} color="#FFFFFF" />
              </div>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(true)}
              className="lg:hidden p-2 focus:outline-none"
              aria-label="Open menu"
            >
              <span className="block w-5 h-[2px] bg-[#1D3045] mb-1" />
              <span className="block w-5 h-[2px] bg-[#1D3045]" />
            </button>
          </div>

        </header>

        {/* 3. Centered Main Stage: Features positioned in true vertical middle with balanced SERVICES header */}
        <main className="relative z-20 flex-1 w-full max-w-6xl mx-auto px-6 sm:px-8 md:px-12 lg:px-20 flex items-center justify-center -translate-y-4 sm:-translate-y-6 md:-translate-y-8">
          <div className="relative w-full min-h-[260px] sm:min-h-[220px] md:min-h-[240px] flex items-center">
            
            {/* SERVICES Heading - Balanced spacing directly above 01 */}
            <div className="absolute -top-20 sm:-top-24 md:-top-28 left-0 pointer-events-none">
              <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-black uppercase tracking-tight text-[#1D3045] leading-none select-none text-left">
                SERVICES
              </h1>
            </div>

            {/* Features (Service items) - Centered in true visual middle of viewport */}
            {SERVICES_DATA.map((service, index) => {
              const diff = currentStageFloat - index;
              const absDiff = Math.abs(diff);

              // 0.30 reading plateau + 0.70 transition zone with smooth cosine easing
              const plateau = 0.30;
              const transitionSpan = 0.70;

              let opacity = 0;
              let translateY = 0;

              if (absDiff <= plateau) {
                opacity = 1;
                translateY = 0;
              } else if (absDiff < plateau + transitionSpan) {
                const t = (absDiff - plateau) / transitionSpan;
                const ease = 0.5 - 0.5 * Math.cos(Math.PI * t);
                
                opacity = 1 - ease;
                translateY = (diff > 0 ? -1 : 1) * ease * 24;
              } else {
                opacity = 0;
                translateY = (diff > 0 ? -24 : 24);
              }

              if (opacity <= 0.001) return null;

              return (
                <div
                  key={service.number}
                  className="absolute inset-x-0 top-1/2 -translate-y-1/2 w-full transition-none pointer-events-auto"
                  style={{
                    opacity: opacity,
                    transform: `translate3d(0, ${translateY}px, 0)`,
                  }}
                  aria-hidden={opacity < 0.5}
                >
                  {/* Exact Layout: Big Number + Title & Description & Divider */}
                  <div className="flex flex-col md:flex-row md:items-start gap-6 md:gap-12 lg:gap-16 w-full">
                    
                    {/* Large Dominant Number (01, 02, etc.) */}
                    <div className="shrink-0 w-32 sm:w-40 md:w-48 lg:w-52">
                      <span className="text-7xl sm:text-8xl md:text-9xl font-black tracking-tighter text-[#1D3045] leading-none select-none block">
                        {service.number}
                      </span>
                    </div>

                    {/* Content Column: Title, Description, Divider */}
                    <div className="flex-1 flex flex-col justify-start max-w-2xl pt-1 md:pt-3">
                      
                      {/* Service Title */}
                      <h2 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-black tracking-tight uppercase text-[#1D3045] leading-tight mb-4">
                        {service.title}
                      </h2>

                      {/* Description */}
                      <p className="text-sm sm:text-base md:text-lg font-normal text-[#1D3045]/80 leading-relaxed max-w-xl">
                        {service.description}
                      </p>

                      {/* Thin Divider */}
                      <div className="w-full h-[1px] bg-[#1D3045]/20 mt-8 md:mt-12" />

                    </div>

                  </div>
                </div>
              );
            })}
          </div>
        </main>

        {/* 4. Footer */}
        <footer className="relative z-20 w-full px-6 sm:px-8 md:px-12 lg:px-24 pb-8 sm:pb-10 text-xs uppercase tracking-[0.2em] text-[#1D3045]/40 font-medium select-none flex justify-between items-center pointer-events-none">
          <span className="font-mono text-sm tracking-widest text-[#1D3045]">
            0{activeIndex + 1} / 07
          </span>
          <span className="hidden sm:inline">FICOS FREIGHT SERVICES</span>
        </footer>

        {/* 5. Vertical Progress Indicator (7 Dots on Right Margin) */}
        <div
          className="absolute right-4 sm:right-8 md:right-12 top-1/2 -translate-y-1/2 z-30 flex flex-col items-center gap-3.5 pointer-events-none"
          aria-label="Scroll Progress"
        >
          {SERVICES_DATA.map((_, index) => {
            const isCurrent = index === activeIndex;
            return (
              <span
                key={index}
                className="rounded-full transition-all duration-300 ease-out"
                style={{
                  width: isCurrent ? '8px' : '5px',
                  height: isCurrent ? '8px' : '5px',
                  backgroundColor: isCurrent ? '#1D3045' : '#1D304530',
                  transform: isCurrent ? 'scale(1.25)' : 'scale(1)',
                }}
              />
            );
          })}
        </div>

        {/* Mobile Menu Modal */}
        {mobileMenuOpen && (
          <div className="fixed inset-0 z-50 bg-[#EAEFF5] flex flex-col justify-between p-8">
            <div className="flex justify-end">
              <button
                onClick={() => setMobileMenuOpen(false)}
                className="w-10 h-10 rounded-full border border-[#1D3045]/30 flex items-center justify-center text-[#1D3045]"
              >
                <X size={18} />
              </button>
            </div>
            <div className="flex flex-col gap-6 text-2xl font-light tracking-wide uppercase text-[#1D3045]">
              <button
                onClick={() => {
                  setMobileMenuOpen(false);
                  onNavigate('landing');
                }}
                className="text-left"
              >
                HOME
              </button>
              <button
                onClick={() => {
                  setMobileMenuOpen(false);
                  onNavigate('services');
                }}
                className="text-left font-bold"
              >
                SERVICES (HOW IT WORKS)
              </button>
              <button
                onClick={() => {
                  setMobileMenuOpen(false);
                  onNavigate('dashboard');
                }}
                className="text-left"
              >
                DASHBOARD →
              </button>
            </div>
            <div className="text-xs tracking-widest uppercase text-[#1D3045]/60">
              FICOS FREIGHT INTELLIGENCE
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
