"use client"
import React, { useState, useEffect, useRef } from 'react';
import { Search, BookOpen, Sparkles, ArrowRight, Moon, Sun, Cpu, Layers, Zap } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

const MainPage = () => {
  const [isDark, setIsDark] = useState(false);
  const [selectedTextbook, setSelectedTextbook] = useState('intro_to_ml');
  const [mounted, setMounted] = useState(false);
  const [hovered, setHovered] = useState(null);

  const featuredQueriesByTextbook = {
    computer_networks: [
      "Explain the OSI model",
      "What is IP addressing?",
      "How does TCP work?",
    ],
    intro_to_ml: [
      "What is machine learning?",
      "Explain neural networks",
      "Define supervised learning",
    ],
    economics: [
      "What is supply and demand?",
      "Explain opportunity cost",
      "Define inflation",
    ],
  };

  const [availableTextbooks, setAvailableTextbooks] = useState([
    { id: 'intro_to_ml',       name: 'Machine Learning',   icon: '🤖' },
    { id: 'computer_networks', name: 'Computer Networks',  icon: '🌐' },
    { id: 'economics',         name: 'Economics',           icon: '📊' },
  ]);

  const getCurrentQueries = () =>
    featuredQueriesByTextbook[selectedTextbook] || featuredQueriesByTextbook['intro_to_ml'];

  useEffect(() => {
    setMounted(true);
    const saved = localStorage.getItem('theme');
    if (saved) {
      setIsDark(saved === 'dark');
    } else {
      setIsDark(window.matchMedia('(prefers-color-scheme: dark)').matches);
    }

    fetch(`${API_URL}/textbooks`)
      .then(r => r.json())
      .then(data => {
        if (data.textbooks?.length) {
          const iconMap = { intro_to_ml: '🤖', computer_networks: '🌐', economics: '📊' };
          setAvailableTextbooks(
            data.textbooks.map(t => ({ id: t.id, name: t.name, icon: iconMap[t.id] || '📚' }))
          );
        }
      })
      .catch(() => {});
  }, []);

  const toggleTheme = () => {
    const next = !isDark;
    setIsDark(next);
    localStorage.setItem('theme', next ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark', next);
  };

  // Sync <html> class with isDark
  useEffect(() => {
    if (mounted) document.documentElement.classList.toggle('dark', isDark);
  }, [isDark, mounted]);

  if (!mounted) return null;

  const pillStyle = (id) => ({
    padding: '0.5rem 1.25rem',
    borderRadius: '100px',
    fontSize: '0.85rem',
    fontWeight: 500,
    cursor: 'pointer',
    border: '1px solid',
    transition: 'all 0.2s',
    background: selectedTextbook === id ? 'var(--accent-1)' : 'transparent',
    color: selectedTextbook === id ? '#fff' : 'var(--text-secondary)',
    borderColor: selectedTextbook === id ? 'var(--accent-1)' : 'var(--border-base)',
    transform: hovered === id && selectedTextbook !== id ? 'translateY(-1px)' : 'none',
  });

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-base)', position: 'relative', overflow: 'hidden' }}>

      {/* Ambient background blobs */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0 }}>
        <div className="ll-blob-a animate-pulse-soft" style={{ position: 'absolute', top: '-15%', left: '-10%', width: '45vw', height: '45vw', opacity: 0.8 }} />
        <div className="ll-blob-b animate-pulse-soft delay-1000" style={{ position: 'absolute', bottom: '-15%', right: '-8%', width: '40vw', height: '40vw', opacity: 0.7 }} />
      </div>

      {/* Nav */}
      <nav className="ll-nav" style={{ position: 'sticky', top: 0, zIndex: 50, padding: '0 2rem', height: '64px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ background: 'var(--accent-glow)', border: '1px solid rgba(82,183,136,0.25)', borderRadius: '10px', padding: '8px' }}>
            <BookOpen size={18} style={{ color: 'var(--accent-1)' }} />
          </div>
          <span style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
            LearnLens
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', padding: '4px 10px', background: 'var(--bg-elevated)', borderRadius: '100px', border: '1px solid var(--border-subtle)' }}>
            RAG · Hybrid Retrieval · Groq LLM
          </span>
          <button
            onClick={toggleTheme}
            style={{ padding: '8px', borderRadius: '10px', border: '1px solid var(--border-base)', background: 'var(--bg-elevated)', cursor: 'pointer', display: 'flex', alignItems: 'center', transition: 'all 0.2s' }}
          >
            {isDark
              ? <Sun size={16} style={{ color: '#f59e0b' }} />
              : <Moon size={16} style={{ color: 'var(--text-secondary)' }} />
            }
          </button>
        </div>
      </nav>

      {/* Hero */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '3rem 1.5rem 4rem', position: 'relative', zIndex: 10 }}>

        {/* Eyebrow pill */}
        <div className="animate-fade-in" style={{ display: 'inline-flex', alignItems: 'center', gap: '7px', padding: '5px 14px', borderRadius: '100px', border: '1px solid var(--border-base)', background: 'var(--bg-surface)', marginBottom: '2rem', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
          <Sparkles size={11} style={{ color: 'var(--warm-1)' }} />
          <span>Production-Grade RAG · Hybrid BM25+FAISS+RRF</span>
        </div>

        {/* Headline */}
        <div className="animate-fade-in-up" style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <h1 style={{ fontSize: 'clamp(2.8rem, 6vw, 5.5rem)', fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1.08, color: 'var(--text-primary)', margin: 0 }}>
            Master your<br />
            <span style={{
              background: 'linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 50%, var(--warm-1) 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}>Textbook</span>
          </h1>
        </div>

        {/* Subheadline */}
        <p className="animate-fade-in-up delay-100" style={{ fontSize: 'clamp(1rem, 2vw, 1.15rem)', color: 'var(--text-secondary)', textAlign: 'center', maxWidth: '520px', lineHeight: 1.65, marginBottom: '3rem' }}>
          Instant, cited answers from your study materials. Ask anything — AI retrieves, reranks, and synthesizes.
        </p>

        {/* Textbook selector */}
        <div className="animate-fade-in-up delay-200" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center', marginBottom: '2rem' }}>
          {availableTextbooks.map(tb => (
            <button
              key={tb.id}
              style={pillStyle(tb.id)}
              onClick={() => setSelectedTextbook(tb.id)}
              onMouseEnter={() => setHovered(tb.id)}
              onMouseLeave={() => setHovered(null)}
            >
              {tb.icon} {tb.name}
            </button>
          ))}
        </div>

        {/* Search bar */}
        <div className="animate-fade-in-up delay-300" style={{ width: '100%', maxWidth: '620px', marginBottom: '1.5rem' }}>
          <div
            onClick={() => window.location.href = `/search/answer?textbook=${selectedTextbook}`}
            style={{
              display: 'flex', alignItems: 'center', gap: '14px',
              padding: '14px 14px 14px 20px',
              borderRadius: '16px',
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-base)',
              boxShadow: 'var(--shadow-md)',
              cursor: 'pointer',
              transition: 'box-shadow 0.25s, transform 0.2s',
            }}
            onMouseEnter={e => { e.currentTarget.style.boxShadow = 'var(--shadow-lg)'; e.currentTarget.style.transform = 'translateY(-2px)'; }}
            onMouseLeave={e => { e.currentTarget.style.boxShadow = 'var(--shadow-md)'; e.currentTarget.style.transform = 'translateY(0)'; }}
          >
            <Search size={20} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
            <span style={{ flex: 1, color: 'var(--text-muted)', fontSize: '1rem' }}>
              Ask anything about {availableTextbooks.find(t => t.id === selectedTextbook)?.name || 'your textbook'}...
            </span>
            <div className="ll-btn-accent" style={{ padding: '8px 16px', borderRadius: '10px', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.875rem', whiteSpace: 'nowrap' }}>
              <Sparkles size={14} /> Ask AI
            </div>
          </div>
        </div>

        {/* Quick chips */}
        <div className="animate-fade-in-up delay-500" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center', maxWidth: '600px' }}>
          {getCurrentQueries().map((q, i) => (
            <button
              key={i}
              onClick={() => window.location.href = `/search/answer?textbook=${selectedTextbook}&q=${encodeURIComponent(q)}`}
              className="ll-btn-ghost"
              style={{ fontSize: '0.8rem', padding: '5px 13px' }}
            >
              {q}
            </button>
          ))}
        </div>

        {/* Feature strip */}
        <div className="animate-fade-in-up delay-500" style={{ display: 'flex', gap: '20px', marginTop: '4rem', flexWrap: 'wrap', justifyContent: 'center' }}>
          {[
            { icon: <Layers size={15} />, label: 'Hybrid Retrieval', sub: 'BM25 + FAISS + RRF' },
            { icon: <Cpu size={15} />,    label: 'Cross-Encoder Reranking', sub: 'ms-marco MiniLM L-12' },
            { icon: <Zap size={15} />,    label: 'Groq LLM', sub: '~500 tok/s · Llama 3.3 70B' },
          ].map((f, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', borderRadius: '12px', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
              <span style={{ color: 'var(--accent-1)' }}>{f.icon}</span>
              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{f.label}</span>
              <span style={{ color: 'var(--text-muted)' }}>· {f.sub}</span>
            </div>
          ))}
        </div>
      </main>

      {/* Footer */}
      <footer style={{ padding: '1.5rem', textAlign: 'center', position: 'relative', zIndex: 10 }}>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          © {new Date().getFullYear()} LearnLens — Production RAG demo · Built for interview showcase
        </p>
      </footer>
    </div>
  );
};

export default MainPage;