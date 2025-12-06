"use client"
import React, { useState, useEffect } from 'react';
import { Search, BookOpen, Sparkles, ArrowRight, Moon, Sun } from 'lucide-react';

const MainPage = () => {
  const [isDark, setIsDark] = useState(false);
  const [selectedTextbook, setSelectedTextbook] = useState('intro_to_ml');
  const [mounted, setMounted] = useState(false);

  const featuredQueriesByTextbook = {
    'computer_networks': [
      "What is a computer network?",
      "Explain the OSI model",
      "What is IP addressing?",
    ],
    'intro_to_ml': [
      "What is machine learning?",
      "Explain neural networks",
      "Define supervised learning",
    ],
    'economics': [
      "What is supply and demand?",
      "Explain opportunity cost",
      "Define inflation"
    ]
  };

  const availableTextbooks = [
    { id: 'intro_to_ml', name: 'Machine Learning', description: 'ML algorithms' },
    { id: 'computer_networks', name: 'Computer Networks', description: 'Network protocols' }
  ];

  const getCurrentFeaturedQueries = () => {
    return featuredQueriesByTextbook[selectedTextbook] || featuredQueriesByTextbook['intro_to_ml'];
  };

  // Theme management
  useEffect(() => {
    setMounted(true);
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      setIsDark(savedTheme === 'dark');
    } else {
      setIsDark(window.matchMedia('(prefers-color-scheme: dark)').matches);
    }
  }, []);

  const toggleTheme = () => {
    const newTheme = !isDark;
    setIsDark(newTheme);
    localStorage.setItem('theme', newTheme ? 'dark' : 'light');
  };

  if (!mounted) return null;

  const themeClasses = {
    bg: isDark ? 'bg-[#0a0a0a]' : 'bg-[#fafafa]',
    text: isDark ? 'text-gray-100' : 'text-gray-900',
    textSecondary: isDark ? 'text-gray-400' : 'text-gray-500',
    cardBg: isDark ? 'bg-white/5 border-white/10' : 'bg-black/5 border-black/5',
    accent: 'bg-blue-600',
    accentHover: 'hover:bg-blue-700',
  };

  return (
    <div className={`min-h-screen flex flex-col transition-colors duration-500 ${themeClasses.bg} relative overflow-hidden`}>

      {/* Abstract Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className={`absolute top-[-20%] left-[-10%] w-[50vw] h-[50vw] rounded-full blur-[120px] opacity-30 animate-pulse ${isDark ? 'bg-blue-900/40' : 'bg-blue-200/60'}`} />
        <div className={`absolute bottom-[-20%] right-[-10%] w-[50vw] h-[50vw] rounded-full blur-[120px] opacity-30 animate-pulse delay-1000 ${isDark ? 'bg-indigo-900/40' : 'bg-indigo-200/60'}`} />
      </div>

      {/* Navigation */}
      <nav className="w-full p-6 flex justify-between items-center z-10">
        <div className="flex items-center gap-2">
          <div className={`p-2 rounded-xl ${isDark ? 'bg-white/10' : 'bg-black/5'}`}>
            <BookOpen size={20} className={isDark ? 'text-white' : 'text-black'} />
          </div>
          <span className={`font-bold text-lg tracking-tight ${themeClasses.text}`}>LearnLens</span>
        </div>
        <button
          onClick={toggleTheme}
          className={`p-2 rounded-full transition-all duration-300 ${themeClasses.cardBg} backdrop-blur-md border hover:scale-110`}
        >
          {isDark ? <Sun size={18} className="text-yellow-400" /> : <Moon size={18} className="text-slate-600" />}
        </button>
      </nav>

      {/* Main Content */}
      <main className="flex-1 flex flex-col justify-center items-center px-4 z-10 max-w-5xl mx-auto w-full">

        {/* Hero Text */}
        <div className="text-center mb-12 space-y-6 animate-fade-in-up">
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${themeClasses.cardBg} ${themeClasses.textSecondary} backdrop-blur-sm`}>
            <Sparkles size={12} />
            <span>AI-Powered Textbook Assistant</span>
          </div>

          <h1 className={`text-5xl md:text-7xl font-bold tracking-tighter ${themeClasses.text}`}>
            Master your <br />
            <span className="bg-gradient-to-r from-blue-500 to-indigo-500 bg-clip-text text-transparent">Textbook</span>
          </h1>

          <p className={`text-lg md:text-xl max-w-2xl mx-auto leading-relaxed ${themeClasses.textSecondary}`}>
            Instant, accurate answers from your study materials. <br className="hidden md:block" />
            Just ask, and let AI do the searching.
          </p>
        </div>

        {/* Interactive Area */}
        <div className="w-full max-w-2xl space-y-8 animate-fade-in-up delay-200">

          {/* Textbook Selector */}
          <div className="flex justify-center">
            <div className={`inline-flex p-1 rounded-2xl border backdrop-blur-md ${themeClasses.cardBg}`}>
              {availableTextbooks.map((textbook) => (
                <button
                  key={textbook.id}
                  onClick={() => setSelectedTextbook(textbook.id)}
                  className={`px-4 py-2 rounded-xl text-sm font-medium transition-all duration-300 ${selectedTextbook === textbook.id
                      ? 'bg-blue-600 text-white shadow-lg'
                      : `${themeClasses.textSecondary} hover:${themeClasses.text}`
                    }`}
                >
                  {textbook.name}
                </button>
              ))}
            </div>
          </div>

          {/* Search Trigger */}
          <div
            onClick={() => window.location.href = `/search?textbook=${selectedTextbook}`}
            className={`group relative w-full p-1 rounded-3xl bg-gradient-to-r from-blue-500/20 to-indigo-500/20 cursor-pointer transition-transform hover:scale-[1.01] duration-300`}
          >
            <div className={`relative flex items-center gap-4 p-4 md:p-6 rounded-[20px] ${isDark ? 'bg-gray-900' : 'bg-white'} border ${isDark ? 'border-gray-800' : 'border-gray-100'} shadow-2xl`}>
              <Search className={`w-6 h-6 ${themeClasses.textSecondary}`} />
              <div className="flex-1">
                <span className={`text-lg ${themeClasses.textSecondary} font-light`}>
                  Ask a question about {availableTextbooks.find(t => t.id === selectedTextbook)?.name}...
                </span>
              </div>
              <div className={`p-3 rounded-xl bg-blue-600 text-white transition-transform group-hover:translate-x-1`}>
                <ArrowRight size={20} />
              </div>
            </div>
          </div>

          {/* Featured Queries */}
          <div className="flex flex-wrap justify-center gap-2">
            {getCurrentFeaturedQueries().map((query, index) => (
              <button
                key={index}
                onClick={(e) => {
                  e.stopPropagation();
                  window.location.href = `/search?textbook=${selectedTextbook}&q=${encodeURIComponent(query)}`;
                }}
                className={`px-4 py-2 rounded-full text-xs md:text-sm border transition-all duration-300 hover:-translate-y-1 ${themeClasses.cardBg} ${themeClasses.textSecondary} hover:${themeClasses.text} hover:border-blue-500/30`}
              >
                {query}
              </button>
            ))}
          </div>
        </div>

      </main>

      {/* Minimal Footer */}
      <footer className="p-6 text-center z-10">
        <p className={`text-xs ${themeClasses.textSecondary}`}>
          © {new Date().getFullYear()} LearnLens. Powered by Advanced AI.
        </p>
      </footer>

      <style jsx global>{`
        @keyframes fade-in-up {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in-up {
          animation: fade-in-up 0.8s ease-out forwards;
        }
        .delay-1000 { animation-delay: 1s; }
        .delay-200 { animation-delay: 0.2s; }
      `}</style>
    </div>
  );
};

export default MainPage;