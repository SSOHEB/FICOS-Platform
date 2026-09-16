import React, { useEffect, useRef, useState } from 'react';
import { ArrowRight, ArrowDown, ChevronUp, Info, X } from 'lucide-react';
import { useVideoScrub } from './useVideoScrub';
import Dashboard from './Dashboard';
import Services from './Services';
import VesselIntelligence from './VesselIntelligence';
import IdleIntelligence from './IdleIntelligence';

const VIDEO_URL = 'https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260821_114821_a8ca298f-be2c-4613-a4dd-51b69e16bbde.mp4';
const DARK = '#1D3045';

interface StaggerProps {
  visible: boolean;
  delay?: number;
  children: React.ReactNode;
  className?: string;
}

const Stagger: React.FC<StaggerProps> = ({ visible, delay = 0, children, className = '' }) => {
  return (
    <div
      className={`transition-all duration-[800ms] ${className}`}
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(24px)',
        transitionTimingFunction: 'cubic-bezier(0.16, 1, 0.3, 1)',
        transitionDelay: `${delay}ms`,
      }}
    >
      {children}
    </div>
  );
};

export default function App() {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [navLoaded, setNavLoaded] = useState(false);
  const [currentView, setCurrentView] = useState<'landing' | 'dashboard' | 'services' | 'vessel-intelligence' | 'idle-intelligence'>('landing');

  // Handle URL route changes
  useEffect(() => {
    const checkRoute = () => {
      const path = window.location.pathname;
      const hash = window.location.hash;

      if (path === '/idle-intelligence' || hash === '#idle-intelligence') {
        setCurrentView('idle-intelligence');
      } else if (path === '/vessel-intelligence' || hash === '#vessel-intelligence') {
        setCurrentView('vessel-intelligence');
      } else if (path === '/dashboard' || hash === '#dashboard') {
        setCurrentView('dashboard');
      } else if (path === '/services' || hash === '#services' || hash === '#how-it-works') {
        setCurrentView('services');
      } else {
        setCurrentView('landing');
      }
    };

    checkRoute();
    window.addEventListener('popstate', checkRoute);
    window.addEventListener('hashchange', checkRoute);

    return () => {
      window.removeEventListener('popstate', checkRoute);
      window.removeEventListener('hashchange', checkRoute);
    };
  }, []);

  const navigateTo = (view: 'landing' | 'dashboard' | 'services' | 'vessel-intelligence' | 'idle-intelligence') => {
    setCurrentView(view);
    if (view === 'idle-intelligence') {
      window.history.pushState(null, '', '#idle-intelligence');
    } else if (view === 'vessel-intelligence') {
      window.history.pushState(null, '', '#vessel-intelligence');
    } else if (view === 'dashboard') {
      window.history.pushState(null, '', '#dashboard');
    } else if (view === 'services') {
      window.history.pushState(null, '', '#services');
    } else {
      window.history.pushState(null, '', '#');
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const { scrollProgress, canvasLive } = useVideoScrub({
    videoSrc: VIDEO_URL,
    containerRef,
    videoRef,
    canvasRef,
  });

  // Nav entrance delay on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      setNavLoaded(true);
    }, 200);
    return () => clearTimeout(timer);
  }, []);

  // Lock body scroll when mobile menu is open
  useEffect(() => {
    if (mobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'auto';
    }
  }, [mobileMenuOpen]);

  // Section Opacities
  const p = scrollProgress;

  const s1Opacity = p < 0.2 ? 1 : Math.max(0, 1 - (p - 0.2) / 0.08);

  const s2Opacity =
    p < 0.32
      ? 0
      : p < 0.4
      ? (p - 0.32) / 0.08
      : p < 0.55
      ? 1
      : Math.max(0, 1 - (p - 0.55) / 0.08);

  const s3Opacity = p < 0.67 ? 0 : p < 0.75 ? (p - 0.67) / 0.08 : 1;

  // Navbar color flips at p > 0.55
  const isLightNav = p > 0.55;
  const navColor = isLightNav ? '#FFFFFF' : DARK;

  const navLinks = [
    { label: 'FICOS', href: '#', active: currentView === 'landing', action: () => navigateTo('landing') },
    { label: 'FREIGHT INTELLIGENCE', href: '#', active: false, action: () => navigateTo('landing') },
    { label: 'VESSEL INTELLIGENCE', href: '#vessel-intelligence', active: currentView === 'vessel-intelligence', action: () => navigateTo('vessel-intelligence') },
    { label: 'IDLE INTELLIGENCE', href: '#idle-intelligence', active: currentView === 'idle-intelligence', action: () => navigateTo('idle-intelligence') },
    { label: 'DASHBOARD', href: '#dashboard', active: currentView === 'dashboard', action: () => navigateTo('dashboard') },
    { label: 'HOW IT WORKS', href: '#services', active: currentView === 'services', action: () => navigateTo('services') },
    { label: 'ABOUT', href: '#', active: false, action: () => navigateTo('landing') },
  ];

  if (currentView === 'idle-intelligence') {
    return <IdleIntelligence onNavigate={navigateTo} />;
  }

  if (currentView === 'vessel-intelligence') {
    return <VesselIntelligence onNavigate={navigateTo} />;
  }

  if (currentView === 'dashboard') {
    return <Dashboard onBackToLanding={() => navigateTo('landing')} onNavigate={navigateTo} />;
  }

  if (currentView === 'services') {
    return <Services onNavigate={navigateTo} />;
  }

  return (
    <div ref={containerRef} className="relative h-[500vh] bg-[#E8EEF5] text-[#1D3045]">
      
      {/* STICKY FULL VIEWPORT SCENE */}
      <div className="sticky top-0 w-full h-screen overflow-hidden">
        
        {/* 1) Video Full Cover */}
        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover"
          src={VIDEO_URL}
          muted
          playsInline
          preload="auto"
          aria-hidden="true"
        />

        {/* 2) Canvas 1920x1080 Frame Extraction Overlay */}
        <canvas
          ref={canvasRef}
          width={1920}
          height={1080}
          className="absolute inset-0 w-full h-full object-cover transition-opacity duration-300"
          style={{ opacity: canvasLive ? 1 : 0 }}
          aria-hidden="true"
        />

        {/* 3) UI Overlay Container */}
        <div className="absolute inset-0 pointer-events-none">
          
          {/* NAVBAR */}
          <header className="absolute top-0 left-0 right-0 z-50 pointer-events-auto px-6 sm:px-8 md:px-12 pt-7 sm:pt-10 pb-6 flex items-center justify-between transition-colors duration-500" style={{ color: navColor }}>
            
            {/* Desktop Left Cluster */}
            <nav className="hidden lg:flex items-center gap-6 xl:gap-8" aria-label="Main Navigation">
              {navLinks.map((link, idx) => (
                <button
                  key={link.label}
                  onClick={link.action}
                  className={`relative text-[11px] tracking-[0.16em] uppercase font-medium hover:opacity-70 transition-all duration-300 ${
                    navLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-3'
                  }`}
                  style={{
                    color: navColor,
                    transitionDuration: '600ms',
                    transitionTimingFunction: 'cubic-bezier(0.16, 1, 0.3, 1)',
                    transitionDelay: `${idx * 80 + 100}ms`,
                  }}
                >
                  {link.label}
                  {link.active && (
                    <span
                      className="absolute -bottom-3 left-0 right-0 h-[2px] w-full transition-colors duration-500"
                      style={{ backgroundColor: navColor }}
                    />
                  )}
                </button>
              ))}
            </nav>

            {/* Mobile Hamburger Button */}
            <button
              onClick={() => setMobileMenuOpen(true)}
              className="lg:hidden flex flex-col justify-center gap-[5px] p-2 focus:outline-none"
              aria-label="Open navigation menu"
            >
              <span
                className="block w-6 h-[2px] transition-colors duration-500"
                style={{ backgroundColor: navColor }}
              />
              <span
                className="block w-6 h-[2px] transition-colors duration-500"
                style={{ backgroundColor: navColor }}
              />
              <span
                className="block w-4 h-[2px] transition-colors duration-500"
                style={{ backgroundColor: navColor }}
              />
            </button>

            {/* Right Cluster */}
            <div
              className={`hidden sm:flex items-center gap-5 transition-all duration-300 ${
                navLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-3'
              }`}
              style={{
                transitionDuration: '600ms',
                transitionTimingFunction: 'cubic-bezier(0.16, 1, 0.3, 1)',
                transitionDelay: '500ms',
              }}
            >
              <button
                onClick={() => navigateTo('dashboard')}
                className="text-[11px] tracking-[0.2em] uppercase font-medium hover:opacity-80 transition-opacity"
                style={{ color: navColor }}
              >
                OPEN DASHBOARD
              </button>

              <div className="flex items-center gap-2">
                <span className="text-[11px] tracking-[0.2em] uppercase font-medium" style={{ color: navColor }}>
                  INSIGHTS
                </span>
                <div
                  className="w-5 h-5 rounded-full flex items-center justify-center transition-colors duration-500"
                  style={{
                    backgroundColor: navColor,
                    color: isLightNav ? '#1D3045' : '#FFFFFF',
                  }}
                >
                  <Info size={10} color={isLightNav ? '#1D3045' : '#FFFFFF'} />
                </div>
              </div>

              {/* Menu label */}
              <button
                onClick={() => setMobileMenuOpen(true)}
                className="text-[11px] tracking-[0.2em] uppercase font-medium focus:outline-none hover:opacity-80"
                style={{ color: navColor }}
              >
                MENU
              </button>
            </div>

          </header>

          {/* MOBILE MENU OVERLAY */}
          <div
            className={`fixed inset-0 z-[100] pointer-events-auto transition-all duration-500 ${
              mobileMenuOpen ? 'opacity-100 visible' : 'opacity-0 invisible'
            }`}
            style={{
              backgroundColor: DARK,
              transitionTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)',
            }}
          >
            {/* Inner Panel */}
            <div
              className={`w-full h-full flex flex-col justify-between transition-transform duration-500 ${
                mobileMenuOpen ? 'translate-y-0' : '-translate-y-8'
              }`}
            >
              {/* Top Bar with Close Button */}
              <div className="flex justify-end px-6 sm:px-8 pt-8 sm:pt-12">
                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="w-10 h-10 rounded-full border border-white/30 flex items-center justify-center text-white hover:border-white transition-colors"
                  aria-label="Close navigation menu"
                >
                  <X size={18} />
                </button>
              </div>

              {/* Menu Links */}
              <div className="flex flex-col justify-center px-8 sm:px-12 py-3 space-y-4">
                <button
                  onClick={() => {
                    setMobileMenuOpen(false);
                    navigateTo('dashboard');
                  }}
                  className="text-left text-2xl sm:text-3xl font-light tracking-wide uppercase text-white hover:text-white/80"
                >
                  OPEN DASHBOARD →
                </button>
                <button
                  onClick={() => {
                    setMobileMenuOpen(false);
                    navigateTo('services');
                  }}
                  className="text-left text-2xl sm:text-3xl font-light tracking-wide uppercase text-white hover:text-white/80"
                >
                  SERVICES (HOW IT WORKS) →
                </button>
                {navLinks.map((link, idx) => (
                  <button
                    key={link.label}
                    onClick={() => {
                      setMobileMenuOpen(false);
                      link.action();
                    }}
                    className={`text-left text-2xl sm:text-3xl font-light tracking-wide uppercase transition-all duration-500 ${
                      link.active ? 'text-white' : 'text-white/60 hover:text-white'
                    } ${mobileMenuOpen ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-5'}`}
                    style={{
                      transitionDelay: `${idx * 60}ms`,
                    }}
                  >
                    {link.label}
                  </button>
                ))}
              </div>

              {/* Mobile Footer */}
              <div className="flex items-center gap-8 px-8 sm:px-12 pb-10 text-xs tracking-[0.2em] uppercase text-white/60">
                <button onClick={() => { setMobileMenuOpen(false); navigateTo('dashboard'); }} className="hover:text-white uppercase">
                  DASHBOARD
                </button>
                <button onClick={() => { setMobileMenuOpen(false); navigateTo('services'); }} className="hover:text-white uppercase">
                  SERVICES
                </button>
              </div>

            </div>
          </div>

          {/* SECTION 1: HERO (Left Aligned, Vertically Centered) */}
          <div
            className="absolute inset-0 flex flex-col justify-center px-6 sm:px-8 md:px-14 lg:px-24 xl:px-28 transition-opacity duration-100 ease-out z-10 -translate-y-3 sm:-translate-y-5"
            style={{
              opacity: s1Opacity,
              pointerEvents: s1Opacity > 0.3 ? 'auto' : 'none',
            }}
          >
            <div className="max-w-[46rem]">
              <Stagger visible={s1Opacity > 0.3} delay={0}>
                <p
                  className="text-[11px] sm:text-xs tracking-[0.3em] uppercase mb-4 font-semibold text-[#1D3045]"
                >
                  FREIGHT INTELLIGENCE & CHARTER OPTIMIZATION
                </p>
                <h1
                  className="font-light uppercase leading-[1.12]"
                  style={{
                    fontSize: 'clamp(2rem, 4.15vw, 4.25rem)',
                    color: DARK,
                  }}
                >
                  SEE THE FREIGHT MARKET<br />
                  BEFORE YOU<br />
                  CHARTER.
                </h1>
              </Stagger>

              <Stagger visible={s1Opacity > 0.3} delay={150}>
                <p
                  className="mt-6 text-sm sm:text-[15px] tracking-[0.08em] max-w-xl leading-relaxed text-[#1D3045]/80 font-normal"
                >
                  Forecast freight rates, evaluate vessel and port constraints, and turn market uncertainty into a clear chartering decision.
                </p>
              </Stagger>

              <Stagger visible={s1Opacity > 0.3} delay={250}>
                <div className="mt-7 flex flex-wrap items-center gap-4">
                  <button
                    onClick={() => navigateTo('dashboard')}
                    className="px-6 py-3 rounded-full text-[11px] tracking-[0.2em] uppercase font-semibold text-white transition-opacity hover:opacity-90 shadow-sm active:translate-y-0.5"
                    style={{ backgroundColor: DARK }}
                  >
                    OPEN DASHBOARD
                  </button>
                  <button
                    onClick={() => navigateTo('services')}
                    className="px-6 py-3 rounded-full text-[11px] tracking-[0.2em] uppercase font-semibold transition-colors hover:bg-black/5"
                    style={{
                      border: `1px solid ${DARK}66`,
                      color: DARK,
                    }}
                  >
                    EXPLORE FREIGHT INTELLIGENCE
                  </button>
                </div>
              </Stagger>
            </div>

            {/* Bottom-Right Circle Button */}
            <div className="absolute bottom-12 right-6 sm:right-8 md:right-12">
              <Stagger visible={s1Opacity > 0.3} delay={300}>
                <button
                  onClick={() => navigateTo('dashboard')}
                  className="w-12 h-12 rounded-full flex items-center justify-center hover:opacity-70 transition-opacity"
                  style={{
                    border: `1px solid ${DARK}80`,
                    color: DARK,
                  }}
                  aria-label="Open Dashboard"
                >
                  <ArrowRight size={18} />
                </button>
              </Stagger>
            </div>
          </div>

          {/* SECTION 2: CENTER ALIGNED */}
          <div
            className="absolute inset-0 flex items-center justify-center px-6 sm:px-8 transition-opacity duration-100 ease-out"
            style={{
              opacity: s2Opacity,
              pointerEvents: s2Opacity > 0.3 ? 'auto' : 'none',
            }}
          >
            <div className="max-w-[960px] text-center">
              <Stagger visible={s2Opacity > 0.3} delay={0}>
                <p className="text-xs sm:text-sm tracking-[0.3em] uppercase mb-4 font-semibold text-[#1D3045]/80">
                  MARKET & RISK INTELLIGENCE
                </p>
                <h2
                  className="font-extralight tracking-wide leading-[1.3] uppercase"
                  style={{
                    fontSize: 'clamp(1.5rem, 4.2vw, 4.2rem)',
                    color: DARK,
                  }}
                >
                  FICOS forecasts freight rates with scenario precision{' '}
                  <span className="text-[#1D3045]/80">and evaluates vessel feasibility</span>{' '}
                  <span className="text-[#1D3045]/60">to recommend when and how to charter</span>
                </h2>
              </Stagger>
            </div>

            {/* Right Column Controls */}
            <div className="absolute bottom-16 right-6 sm:right-8 md:right-12 flex flex-col items-center gap-4">
              <Stagger visible={s2Opacity > 0.3} delay={200}>
                <button
                  className="w-12 h-12 rounded-full flex items-center justify-center"
                  style={{
                    border: `1px solid ${DARK}66`,
                    color: DARK,
                  }}
                  aria-label="Scroll down"
                >
                  <ArrowDown size={18} />
                </button>
              </Stagger>

              {/* Three Dots */}
              <Stagger visible={s2Opacity > 0.3} delay={350}>
                <div className="flex flex-col items-center gap-2 mt-4">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: DARK }} />
                  <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: `${DARK}66` }} />
                  <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: `${DARK}66` }} />
                </div>
              </Stagger>

              <Stagger visible={s2Opacity > 0.3} delay={500}>
                <button
                  className="w-10 h-10 rounded-full flex items-center justify-center mt-2"
                  style={{
                    border: `1px solid ${DARK}4D`,
                    color: `${DARK}CC`,
                  }}
                  aria-label="Scroll up"
                >
                  <ChevronUp size={16} />
                </button>
              </Stagger>
            </div>
          </div>

          {/* SECTION 3: RIGHT ALIGNED (White Type on Dark Landscape) */}
          <div
            className="absolute inset-0 flex items-center justify-end px-6 sm:px-8 md:px-20 lg:px-32 transition-opacity duration-100 ease-out"
            style={{
              opacity: s3Opacity,
              pointerEvents: s3Opacity > 0.3 ? 'auto' : 'none',
            }}
          >
            <div className="max-w-2xl text-left">
              <Stagger visible={s3Opacity > 0.3} delay={0}>
                <p className="text-white/60 text-sm sm:text-base tracking-[0.25em] uppercase mb-4 font-medium">
                  FICOS FREIGHT INTELLIGENCE
                </p>
              </Stagger>

              <Stagger visible={s3Opacity > 0.3} delay={150}>
                <h2
                  className="font-light text-white leading-[1.2] uppercase tracking-wide mb-8"
                  style={{
                    fontSize: 'clamp(2rem, 4vw, 4rem)',
                  }}
                >
                  SEE THE FREIGHT MARKET,<br />
                  BEFORE YOU CHARTER.
                </h2>
              </Stagger>

              <Stagger visible={s3Opacity > 0.3} delay={300}>
                <div className="flex items-center gap-4">
                  <button
                    onClick={() => navigateTo('dashboard')}
                    className="text-xs sm:text-sm tracking-[0.3em] text-white/90 uppercase font-semibold hover:text-white transition-colors"
                  >
                    OPEN DASHBOARD
                  </button>
                  <button
                    onClick={() => navigateTo('dashboard')}
                    className="w-10 h-10 rounded-full bg-white flex items-center justify-center text-gray-900 hover:scale-110 transition-transform duration-300"
                    aria-label="Open Dashboard"
                  >
                    <ArrowRight size={16} />
                  </button>
                </div>
              </Stagger>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
