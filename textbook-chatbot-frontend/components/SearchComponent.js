"use client"
import React, { useState, useEffect } from 'react';
import { Search, BookOpen, AlertCircle, RefreshCw, Lightbulb, Moon, Sun, Sparkles } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

const SearchComponent = () => {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchAttempted, setSearchAttempted] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedTextbook, setSelectedTextbook] = useState('intro_to_ml');
  const [availableTextbooks, setAvailableTextbooks] = useState([
    { id: 'intro_to_ml',       name: 'Introduction to Machine Learning' },
    { id: 'computer_networks', name: 'Computer Networks' },
  ]);

  const suggestionsMap = {
    computer_networks: ["What is a computer network?", "Explain the OSI model", "What is IP addressing?", "Define packet switching", "How does DNS work?"],
    economics:         ["What is demand?", "Explain law of supply and demand", "What is GDP?", "Define inflation", "What is opportunity cost?"],
    intro_to_ml:       ["What is machine learning?", "Explain supervised learning", "How does classification work?", "What is regression?", "Explain feature selection"],
  };

  // ── Init ──────────────────────────────────────────────────
  useEffect(() => {
    setMounted(true);
    const saved = localStorage.getItem('theme');
    const dark = saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDark(dark);
    document.documentElement.classList.toggle('dark', dark);

    fetch(`${API_URL}/textbooks`)
      .then(r => r.json())
      .then(d => { if (d.textbooks?.length) setAvailableTextbooks(d.textbooks.map(t => ({ id: t.id, name: t.name }))); })
      .catch(() => {});
  }, []);

  // URL params
  useEffect(() => {
    const p = new URLSearchParams(window.location.search);
    const q = p.get('q');
    const tb = p.get('textbook');
    if (tb) setSelectedTextbook(tb);
    if (q) { setQuery(q); setTimeout(() => performSearch(q, tb || selectedTextbook), 100); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleTheme = () => {
    const next = !isDark;
    setIsDark(next);
    localStorage.setItem('theme', next ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark', next);
  };

  // ── Search ────────────────────────────────────────────────
  const performSearch = async (q, tb = selectedTextbook) => {
    if (!q?.trim()) return;
    setLoading(true); setError(''); setResults([]); setSearchAttempted(true);
    try {
      const r = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q.trim(), top_k: 5, textbook: tb }),
      });
      if (!r.ok) { const d = await r.json(); throw new Error(d.message || `HTTP ${r.status}`); }
      const data = await r.json();
      setResults(Array.isArray(data.results) ? data.results : []);
    } catch (e) {
      setError(e.message || 'Unexpected error');
    } finally { setLoading(false); }
  };

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    await performSearch(query, selectedTextbook);
    if (query.trim()) window.history.pushState({}, '', `/search?textbook=${selectedTextbook}&q=${encodeURIComponent(query.trim())}`);
  };

  const handleSuggestion = (s) => {
    setQuery(s);
    performSearch(s, selectedTextbook);
    window.history.pushState({}, '', `/search?textbook=${selectedTextbook}&q=${encodeURIComponent(s)}`);
  };

  if (!mounted) return null;
  const currentSuggestions = suggestionsMap[selectedTextbook] || suggestionsMap['intro_to_ml'];

  // Shared style helpers
  const S = {
    nav: { position: 'sticky', top: 0, zIndex: 50, padding: '0 2rem', height: '64px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
    card: { background: 'var(--bg-surface)', border: '1px solid var(--border-base)', borderRadius: '16px', padding: '1.25rem', boxShadow: 'var(--shadow-sm)' },
    label: { fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' },
    input: { padding: '10px 14px 10px 38px', width: '100%', fontSize: '0.9rem', borderRadius: '10px', border: '1px solid var(--border-base)', background: 'var(--bg-elevated)', color: 'var(--text-primary)', outline: 'none' },
    select: { padding: '9px 14px', width: '100%', fontSize: '0.875rem', borderRadius: '10px', border: '1px solid var(--border-base)', background: 'var(--bg-elevated)', color: 'var(--text-primary)', outline: 'none', cursor: 'pointer' },
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-base)', position: 'relative', overflow: 'hidden' }}>

      {/* Blobs */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0 }}>
        <div className="ll-blob-a animate-pulse-soft" style={{ position: 'absolute', top: '-20%', left: '-10%', width: '45vw', height: '45vw', opacity: 0.5 }} />
        <div className="ll-blob-b animate-pulse-soft delay-1000" style={{ position: 'absolute', bottom: '-20%', right: '-8%', width: '40vw', height: '40vw', opacity: 0.4 }} />
      </div>

      {/* Nav */}
      <nav className="ll-nav" style={S.nav}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }} onClick={() => router.push('/')}>
          <div style={{ background: 'var(--accent-glow)', border: '1px solid rgba(82,183,136,0.25)', borderRadius: '10px', padding: '8px' }}>
            <BookOpen size={18} style={{ color: 'var(--accent-1)' }} />
          </div>
          <span style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>LearnLens</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Link
            href={`/search/answer?textbook=${selectedTextbook}${query ? `&q=${encodeURIComponent(query)}` : ''}`}
            style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '7px 14px', borderRadius: '10px', background: 'var(--accent-1)', color: '#fff', fontWeight: 600, fontSize: '0.82rem', textDecoration: 'none', transition: 'background 0.2s' }}
          >
            <Sparkles size={13} /> Get AI Answer
          </Link>
          <button onClick={toggleTheme} style={{ padding: '8px', borderRadius: '10px', border: '1px solid var(--border-base)', background: 'var(--bg-elevated)', cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
            {isDark ? <Sun size={16} style={{ color: '#f59e0b' }} /> : <Moon size={16} style={{ color: 'var(--text-secondary)' }} />}
          </button>
        </div>
      </nav>

      {/* Main */}
      <main style={{ flex: 1, maxWidth: '1200px', margin: '0 auto', width: '100%', padding: '2rem 1.5rem', position: 'relative', zIndex: 10 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,320px) 1fr', gap: '1.5rem' }}>

          {/* ── Sidebar ── */}
          <aside style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

            {/* Controls card */}
            <div style={S.card}>
              <h2 style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', marginBottom: '1rem' }}>Search Textbook</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div>
                  <label style={S.label}>Textbook</label>
                  <select value={selectedTextbook} onChange={e => setSelectedTextbook(e.target.value)} style={S.select}>
                    {availableTextbooks.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
                  </select>
                </div>

                <div style={{ position: 'relative' }}>
                  <label style={S.label}>Query</label>
                  <Search size={16} style={{ position: 'absolute', left: '11px', top: 'calc(1.2rem + 10px)', color: 'var(--text-muted)' }} />
                  <input
                    type="text"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleSearch(e)}
                    placeholder="Search topics…"
                    style={S.input}
                  />
                </div>

                <button
                  onClick={handleSearch}
                  disabled={loading || !query.trim()}
                  className="ll-btn-accent"
                  style={{ width: '100%', padding: '10px', fontSize: '0.875rem', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '7px' }}
                >
                  {loading ? <RefreshCw size={15} style={{ animation: 'spin 1s linear infinite' }} /> : <Search size={15} />}
                  {loading ? 'Searching…' : 'Search'}
                </button>
              </div>
            </div>

            {/* Suggestions */}
            {!searchAttempted && (
              <div style={S.card}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '10px' }}>
                  <Lightbulb size={14} style={{ color: 'var(--warm-1)' }} />
                  <h3 style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-muted)', margin: 0 }}>Try asking…</h3>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  {currentSuggestions.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => handleSuggestion(s)}
                      style={{ textAlign: 'left', padding: '7px 10px', borderRadius: '8px', border: 'none', background: 'transparent', color: 'var(--text-secondary)', fontSize: '0.85rem', cursor: 'pointer', transition: 'background 0.15s, color 0.15s', lineHeight: 1.4 }}
                      onMouseEnter={e => { e.currentTarget.style.background = 'var(--accent-glow)'; e.currentTarget.style.color = 'var(--accent-1)'; }}
                      onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-secondary)'; }}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </aside>

          {/* ── Results ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

            {/* Status bar */}
            {searchAttempted && !loading && !error && (
              <div style={{ padding: '10px 4px', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Found <strong style={{ color: 'var(--text-primary)' }}>{results.length}</strong> results for <em>"{query}"</em>
              </div>
            )}

            {/* Error */}
            {error && (
              <div style={{ padding: '1.25rem', borderRadius: '14px', border: '1px solid rgba(220,38,38,0.2)', background: 'rgba(220,38,38,0.05)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#dc2626', marginBottom: '4px' }}>
                  <AlertCircle size={16} />
                  <strong style={{ fontSize: '0.9rem' }}>Search Error</strong>
                </div>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>{error}</p>
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '4rem 0' }}>
                <div style={{ position: 'relative', width: '48px', height: '48px', marginBottom: '1rem' }}>
                  <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '3px solid var(--border-base)' }} />
                  <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '3px solid var(--accent-1)', borderTopColor: 'transparent', animation: 'spin 0.8s linear infinite' }} />
                </div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Searching your textbook…</p>
              </div>
            )}

            {/* Result cards */}
            {results.map((r, i) => (
              <div
                key={r.chunk_id || i}
                className="animate-fade-in-up ll-card"
                style={{ padding: '1.25rem', animationDelay: `${i * 60}ms` }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                  <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'var(--accent-glow)', border: '1px solid rgba(82,183,136,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-1)', flexShrink: 0 }}>
                    {i + 1}
                  </div>

                  {r.rerank_score > 0 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div className="score-bar" style={{ width: '80px' }}>
                        <div className="score-bar-fill" style={{ width: `${Math.min(100, r.rerank_score * 100)}%` }} />
                      </div>
                      <span style={{ fontSize: '0.72rem', color: 'var(--accent-1)', fontWeight: 600 }}>
                        {Math.round(r.rerank_score * 100)}% match
                      </span>
                    </div>
                  )}
                </div>

                <p style={{ fontSize: '0.9rem', lineHeight: 1.7, color: 'var(--text-primary)', margin: '0 0 0.75rem' }}>
                  {r.content?.length > 600 ? r.content.slice(0, 600) + '…' : r.content}
                </p>

                <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '0.6rem', display: 'flex', gap: '1rem', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  <span>Chunk #{r.chunk_id}</span>
                  <span>{r.word_count || '—'} words</span>
                </div>
              </div>
            ))}

            {/* Empty state */}
            {!searchAttempted && !loading && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '5rem 0', opacity: 0.5, textAlign: 'center' }}>
                <Search size={40} style={{ color: 'var(--text-muted)', marginBottom: '1rem' }} />
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>Type a question or pick a suggestion on the left.</p>
              </div>
            )}

            {/* No results */}
            {searchAttempted && !loading && results.length === 0 && !error && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '4rem 0', textAlign: 'center' }}>
                <AlertCircle size={36} style={{ color: 'var(--warm-1)', marginBottom: '1rem' }} />
                <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>No results found</h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>Try rephrasing or switching textbooks.</p>
              </div>
            )}
          </div>
        </div>
      </main>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
};

export default SearchComponent;
