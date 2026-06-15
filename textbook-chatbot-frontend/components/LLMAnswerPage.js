"use client"
import React, { useState, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Search, BookOpen, AlertCircle, RefreshCw, Lightbulb, Clock, Moon, Sun, Bot, ChevronDown, ChevronUp, ArrowLeft, Copy, CheckCircle, Sparkles, Database, Brain, Zap, Layers } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

// ── Badges ────────────────────────────────────────────────────
const ConfidenceBadge = ({ confidence }) => {
  if (!confidence || confidence === 'N/A') return null;
  const cfg = {
    High:   { bg: 'rgba(82,183,136,0.12)', color: 'var(--accent-1)',  border: 'rgba(82,183,136,0.25)',  dot: 'var(--accent-1)',  label: 'High Confidence' },
    Medium: { bg: 'rgba(244,162,97,0.12)',  color: 'var(--warm-1)',   border: 'rgba(244,162,97,0.3)',   dot: 'var(--warm-1)',   label: 'Medium Confidence' },
    Low:    { bg: 'rgba(220,38,38,0.10)',   color: '#dc2626',          border: 'rgba(220,38,38,0.2)',   dot: '#dc2626',          label: 'Low Confidence' },
  };
  const c = cfg[confidence] || cfg.Medium;
  return (
    <span style={{ display:'inline-flex', alignItems:'center', gap:'6px', padding:'3px 10px', borderRadius:'100px', fontSize:'0.72rem', fontWeight:600, background:c.bg, color:c.color, border:`1px solid ${c.border}` }}>
      <span style={{ width:'6px', height:'6px', borderRadius:'50%', background:c.dot, flexShrink:0 }} />
      {c.label}
    </span>
  );
};

const ModelBadge = ({ apiUsed, model }) => {
  if (!apiUsed || apiUsed === 'raw') return null;
  const map = {
    groq:       { bg:'rgba(82,183,136,0.10)', color:'var(--accent-1)', border:'rgba(82,183,136,0.25)', label:'Groq · Llama 3.3 70B', icon:'⚡' },
    gemini:     { bg:'rgba(59,130,246,0.10)', color:'#3b82f6',          border:'rgba(59,130,246,0.25)', label:'Gemini 2.0 Flash',     icon:'✦' },
    'together.ai':{ bg:'rgba(168,85,247,0.10)', color:'#a855f7',        border:'rgba(168,85,247,0.25)', label:'Llama 3.3 70B',        icon:'🦙' },
    openrouter: { bg:'rgba(244,162,97,0.10)', color:'var(--warm-1)',   border:'rgba(244,162,97,0.3)',  label:'Mistral 7B',           icon:'🔶' },
  };
  const key = Object.keys(map).find(k => apiUsed?.toLowerCase().includes(k)) || 'gemini';
  const b = map[key];
  return (
    <span style={{ display:'inline-flex', alignItems:'center', gap:'5px', padding:'3px 10px', borderRadius:'100px', fontSize:'0.72rem', fontWeight:600, background:b.bg, color:b.color, border:`1px solid ${b.border}` }}>
      {b.icon} {b.label}
    </span>
  );
};

const RetrievalBadge = ({ mode }) => {
  if (!mode) return null;
  const hybrid = mode === 'hybrid';
  return (
    <span style={{ display:'inline-flex', alignItems:'center', gap:'5px', padding:'3px 10px', borderRadius:'100px', fontSize:'0.72rem', fontWeight:600, background: hybrid ? 'rgba(82,183,136,0.10)' : 'rgba(156,154,148,0.10)', color: hybrid ? 'var(--accent-1)' : 'var(--text-muted)', border:`1px solid ${hybrid ? 'rgba(82,183,136,0.25)' : 'var(--border-base)'}` }}>
      <Database size={10} />
      {hybrid ? 'Hybrid BM25+FAISS+RRF' : 'Dense FAISS'}
    </span>
  );
};

const Skeleton = () => (
  <div style={{ display:'flex', flexDirection:'column', gap:'12px' }}>
    <div style={{ background:'var(--bg-surface)', border:'1px solid var(--border-base)', borderRadius:'16px', padding:'1.5rem' }}>
      <div style={{ display:'flex', alignItems:'center', gap:'12px', marginBottom:'1.25rem' }}>
        <div className="skeleton" style={{ width:'44px', height:'44px', borderRadius:'12px' }} />
        <div style={{ flex:1, display:'flex', flexDirection:'column', gap:'8px' }}>
          <div className="skeleton" style={{ height:'14px', width:'120px', borderRadius:'6px' }} />
          <div className="skeleton" style={{ height:'11px', width:'180px', borderRadius:'6px' }} />
        </div>
      </div>
      {[1,0.85,1,0.7,0.9].map((w,i) => (
        <div key={i} className="skeleton" style={{ height:'12px', width:`${w*100}%`, borderRadius:'6px', marginBottom:'10px' }} />
      ))}
    </div>
  </div>
);

// ── Main ──────────────────────────────────────────────────────
const LLMAnswerPage = () => {
  const router = useRouter();
  const [query, setQuery]     = useState('');
  const [answer, setAnswer]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState('');
  const [searched, setSearched] = useState(false);
  const [isDark, setIsDark]   = useState(false);
  const [showSrc, setShowSrc] = useState(true);
  const [copied, setCopied]   = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedTextbook, setSelectedTextbook] = useState('intro_to_ml');
  const [availableTextbooks, setAvailableTextbooks] = useState([
    { id:'intro_to_ml', name:'Introduction to Machine Learning' },
    { id:'computer_networks', name:'Computer Networks' },
  ]);
  const [history, setHistory] = useState([]);

  const suggestions = {
    computer_networks: ["What is a computer network and how does it work?","Explain the OSI model and its seven layers","How does IP addressing work?","What is packet switching vs circuit switching?","Explain the TCP/IP protocol suite"],
    intro_to_ml:       ["What is machine learning and how does it work?","Explain supervised vs unsupervised learning","How do neural networks process information?","What is gradient descent optimization?","Explain the concept of overfitting"],
    economics:         ["Explain supply and demand with examples","What is GDP and how is it calculated?","Define inflation and its causes","What is opportunity cost?","Explain market equilibrium"],
  };

  useEffect(() => {
    setMounted(true);
    const saved = localStorage.getItem('theme');
    const dark = saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDark(dark);
    document.documentElement.classList.toggle('dark', dark);

    try { setHistory(JSON.parse(sessionStorage.getItem('learnlens_history') || '[]')); } catch {}

    fetch(`${API_URL}/textbooks`)
      .then(r => r.json())
      .then(d => { if (d.textbooks?.length) setAvailableTextbooks(d.textbooks.map(t => ({ id:t.id, name:t.name }))); })
      .catch(() => {});

    const p = new URLSearchParams(window.location.search);
    const q = p.get('q'), tb = p.get('textbook');
    if (tb) setSelectedTextbook(tb);
    if (q) { setQuery(q); setTimeout(() => performSearch(q, tb || 'intro_to_ml'), 150); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleTheme = () => {
    const next = !isDark;
    setIsDark(next);
    localStorage.setItem('theme', next ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark', next);
  };

  const performSearch = useCallback(async (q, tb = selectedTextbook) => {
    if (!q?.trim()) return;
    setLoading(true); setError(''); setAnswer(null); setSearched(true); setShowSrc(true);
    try {
      const r = await fetch(`${API_URL}/search/answer`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ query: q.trim(), textbook: tb }),
      });
      if (!r.ok) { const e = await r.json(); throw new Error(e.message || e.error || `HTTP ${r.status}`); }
      const data = await r.json();
      setAnswer(data);
      const entry = { query:q.trim(), textbook:tb, ts:Date.now() };
      setHistory(prev => {
        const next = [entry, ...prev.filter(h => h.query !== entry.query)].slice(0, 8);
        sessionStorage.setItem('learnlens_history', JSON.stringify(next));
        return next;
      });
    } catch (e) { setError(e.message || 'Unexpected error'); }
    finally { setLoading(false); }
  }, [selectedTextbook]);

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;
    await performSearch(query, selectedTextbook);
    window.history.pushState({}, '', `/search/answer?textbook=${selectedTextbook}&q=${encodeURIComponent(query.trim())}`);
  };

  const handleSuggestion = (s) => {
    setQuery(s);
    performSearch(s, selectedTextbook);
    window.history.pushState({}, '', `/search/answer?textbook=${selectedTextbook}&q=${encodeURIComponent(s)}`);
  };

  const handleCopy = async () => {
    if (!answer?.answer) return;
    await navigator.clipboard.writeText(answer.answer).catch(() => {});
    setCopied(true); setTimeout(() => setCopied(false), 2000);
  };

  if (!mounted) return null;

  const currentSuggestions = suggestions[selectedTextbook] || suggestions['intro_to_ml'];

  const S = {
    nav: { position:'sticky', top:0, zIndex:50, padding:'0 2rem', height:'64px', display:'flex', alignItems:'center', justifyContent:'space-between' },
    card: { background:'var(--bg-surface)', border:'1px solid var(--border-base)', borderRadius:'16px', padding:'1.25rem', boxShadow:'var(--shadow-sm)' },
    label: { fontSize:'0.72rem', fontWeight:600, textTransform:'uppercase', letterSpacing:'0.06em', color:'var(--text-muted)', display:'block', marginBottom:'6px' },
    textarea: { padding:'10px 14px', width:'100%', fontSize:'0.875rem', borderRadius:'10px', border:'1px solid var(--border-base)', background:'var(--bg-elevated)', color:'var(--text-primary)', outline:'none', resize:'none', fontFamily:'inherit', lineHeight:1.6 },
    select: { padding:'9px 14px', width:'100%', fontSize:'0.875rem', borderRadius:'10px', border:'1px solid var(--border-base)', background:'var(--bg-elevated)', color:'var(--text-primary)', outline:'none', cursor:'pointer' },
  };

  return (
    <div style={{ minHeight:'100vh', display:'flex', flexDirection:'column', background:'var(--bg-base)', position:'relative', overflow:'hidden' }}>

      {/* Blobs */}
      <div style={{ position:'fixed', inset:0, pointerEvents:'none', zIndex:0 }}>
        <div className="ll-blob-a animate-pulse-soft" style={{ position:'absolute', top:'-20%', left:'-10%', width:'45vw', height:'45vw', opacity:0.45 }} />
        <div className="ll-blob-b animate-pulse-soft delay-1000" style={{ position:'absolute', bottom:'-20%', right:'-8%', width:'40vw', height:'40vw', opacity:0.35 }} />
      </div>

      {/* Nav */}
      <nav className="ll-nav" style={S.nav}>
        <div style={{ display:'flex', alignItems:'center', gap:'10px', cursor:'pointer' }} onClick={() => router.push('/')}>
          <div style={{ background:'var(--accent-glow)', border:'1px solid rgba(82,183,136,0.25)', borderRadius:'10px', padding:'8px' }}>
            <BookOpen size={18} style={{ color:'var(--accent-1)' }} />
          </div>
          <span style={{ fontWeight:700, fontSize:'1.05rem', color:'var(--text-primary)', letterSpacing:'-0.02em' }}>LearnLens</span>
        </div>

        <div style={{ display:'flex', alignItems:'center', gap:'10px' }}>
          <Link href={`/search?textbook=${selectedTextbook}`}
            style={{ display:'flex', alignItems:'center', gap:'6px', padding:'7px 14px', borderRadius:'10px', border:'1px solid var(--border-base)', background:'var(--bg-elevated)', color:'var(--text-secondary)', fontSize:'0.82rem', fontWeight:500, textDecoration:'none' }}>
            <Search size={13} /> Raw Search
          </Link>
          <button onClick={toggleTheme} style={{ padding:'8px', borderRadius:'10px', border:'1px solid var(--border-base)', background:'var(--bg-elevated)', cursor:'pointer', display:'flex', alignItems:'center' }}>
            {isDark ? <Sun size={16} style={{ color:'#f59e0b' }} /> : <Moon size={16} style={{ color:'var(--text-secondary)' }} />}
          </button>
        </div>
      </nav>

      {/* Main */}
      <main style={{ flex:1, maxWidth:'1200px', margin:'0 auto', width:'100%', padding:'2rem 1.5rem', position:'relative', zIndex:10 }}>
        <div style={{ display:'grid', gridTemplateColumns:'minmax(0,320px) 1fr', gap:'1.5rem' }}>

          {/* ── Sidebar ── */}
          <aside style={{ display:'flex', flexDirection:'column', gap:'1rem' }}>

            {/* Controls */}
            <div style={S.card}>
              <h2 style={{ fontSize:'0.72rem', fontWeight:700, textTransform:'uppercase', letterSpacing:'0.07em', color:'var(--text-muted)', marginBottom:'1rem', margin:'0 0 1rem' }}>Ask Your Textbook</h2>
              <div style={{ display:'flex', flexDirection:'column', gap:'10px' }}>
                <div>
                  <label style={S.label}>Textbook</label>
                  <select value={selectedTextbook} onChange={e => setSelectedTextbook(e.target.value)} style={S.select}>
                    {availableTextbooks.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
                  </select>
                </div>

                <div>
                  <label style={S.label}>Question</label>
                  <textarea
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSearch(e)}
                    placeholder="Ask anything… (Enter to send)"
                    rows={3}
                    style={S.textarea}
                  />
                </div>

                <button onClick={handleSearch} disabled={loading || !query.trim()} className="ll-btn-accent"
                  style={{ width:'100%', padding:'10px', fontSize:'0.875rem', borderRadius:'10px', display:'flex', alignItems:'center', justifyContent:'center', gap:'7px' }}>
                  {loading ? <RefreshCw size={15} style={{ animation:'spin 0.8s linear infinite' }} /> : <Sparkles size={15} />}
                  {loading ? 'Analyzing…' : 'Generate Answer'}
                </button>
              </div>
            </div>

            {/* History */}
            {history.length > 0 && (
              <div style={S.card}>
                <div style={{ display:'flex', alignItems:'center', gap:'7px', marginBottom:'10px' }}>
                  <Clock size={13} style={{ color:'var(--accent-1)' }} />
                  <h3 style={{ fontSize:'0.72rem', fontWeight:700, textTransform:'uppercase', letterSpacing:'0.07em', color:'var(--text-muted)', margin:0 }}>Recent</h3>
                </div>
                {history.map((h, i) => (
                  <button key={i} onClick={() => handleSuggestion(h.query)}
                    style={{ width:'100%', textAlign:'left', padding:'6px 10px', borderRadius:'8px', border:'none', background:'transparent', color:'var(--text-secondary)', fontSize:'0.8rem', cursor:'pointer', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap', transition:'background 0.15s' }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-elevated)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                    {h.query}
                  </button>
                ))}
              </div>
            )}

            {/* Suggestions */}
            {(!searched || answer) && (
              <div style={S.card}>
                <div style={{ display:'flex', alignItems:'center', gap:'7px', marginBottom:'10px' }}>
                  <Lightbulb size={13} style={{ color:'var(--warm-1)' }} />
                  <h3 style={{ fontSize:'0.72rem', fontWeight:700, textTransform:'uppercase', letterSpacing:'0.07em', color:'var(--text-muted)', margin:0 }}>Try asking…</h3>
                </div>
                {currentSuggestions.map((s, i) => (
                  <button key={i} onClick={() => handleSuggestion(s)}
                    style={{ width:'100%', textAlign:'left', padding:'7px 10px', borderRadius:'8px', border:'none', background:'transparent', color:'var(--text-secondary)', fontSize:'0.83rem', cursor:'pointer', lineHeight:1.4, transition:'background 0.15s, color 0.15s' }}
                    onMouseEnter={e => { e.currentTarget.style.background='var(--accent-glow)'; e.currentTarget.style.color='var(--accent-1)'; }}
                    onMouseLeave={e => { e.currentTarget.style.background='transparent'; e.currentTarget.style.color='var(--text-secondary)'; }}>
                    {s}
                  </button>
                ))}
              </div>
            )}
          </aside>

          {/* ── Answer Area ── */}
          <div style={{ display:'flex', flexDirection:'column', gap:'1rem' }}>

            {/* Error */}
            {error && (
              <div className="animate-fade-in-up" style={{ padding:'1.25rem', borderRadius:'14px', border:'1px solid rgba(220,38,38,0.2)', background:'rgba(220,38,38,0.05)' }}>
                <div style={{ display:'flex', alignItems:'center', gap:'8px', color:'#dc2626', marginBottom:'4px' }}>
                  <AlertCircle size={16} /><strong style={{ fontSize:'0.9rem' }}>Generation Failed</strong>
                </div>
                <p style={{ fontSize:'0.85rem', color:'var(--text-secondary)', margin:0 }}>{error}</p>
              </div>
            )}

            {/* Skeleton */}
            {loading && <Skeleton />}

            {/* Answer card */}
            {!loading && answer?.answer && (
              <div className="animate-fade-in-up" style={{ background:'var(--bg-surface)', border:'1px solid var(--border-base)', borderRadius:'18px', overflow:'hidden', boxShadow:'var(--shadow-md)' }}>

                {/* Card header */}
                <div style={{ padding:'1.25rem 1.5rem', borderBottom:'1px solid var(--border-subtle)', background: isDark ? 'linear-gradient(135deg, rgba(64,145,108,0.08) 0%, rgba(82,183,136,0.04) 100%)' : 'linear-gradient(135deg, rgba(64,145,108,0.06) 0%, rgba(82,183,136,0.02) 100%)' }}>
                  <div style={{ display:'flex', alignItems:'flex-start', justifyContent:'space-between', gap:'1rem' }}>
                    <div style={{ display:'flex', alignItems:'center', gap:'12px' }}>
                      <div style={{ padding:'10px', background:'var(--accent-1)', borderRadius:'12px', boxShadow:'0 4px 16px var(--accent-glow)', flexShrink:0 }}>
                        <Brain size={18} color="#fff" />
                      </div>
                      <div>
                        <h2 style={{ fontWeight:700, fontSize:'1rem', color:'var(--text-primary)', margin:0 }}>AI Analysis</h2>
                        <p style={{ fontSize:'0.75rem', color:'var(--text-muted)', margin:'3px 0 0' }}>
                          {answer.chunks_processed || 0} source{answer.chunks_processed !== 1 ? 's' : ''} · {answer.duration || '—'}
                        </p>
                      </div>
                    </div>
                    <button onClick={handleCopy} title="Copy answer"
                      style={{ padding:'7px', borderRadius:'9px', border:'1px solid var(--border-base)', background:'var(--bg-elevated)', cursor:'pointer', display:'flex', alignItems:'center', flexShrink:0 }}>
                      {copied ? <CheckCircle size={16} style={{ color:'var(--accent-1)' }} /> : <Copy size={16} style={{ color:'var(--text-muted)' }} />}
                    </button>
                  </div>

                  {/* Badges */}
                  <div style={{ display:'flex', flexWrap:'wrap', gap:'6px', marginTop:'12px' }}>
                    <ConfidenceBadge confidence={answer.confidence} />
                    <ModelBadge apiUsed={answer.api_used} model={answer.model} />
                    <RetrievalBadge mode={answer.retrieval_mode} />
                  </div>
                </div>

                {/* Markdown body */}
                <div style={{ padding:'1.5rem 1.75rem' }}>
                  <div className="prose-answer">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer.answer}</ReactMarkdown>
                  </div>
                </div>

                {/* Sources — always visible below the answer */}
                {answer.search_results?.length > 0 && (
                  <div style={{ marginTop:'1.5rem' }}>
                    {/* Divider + header */}
                    <div style={{ display:'flex', alignItems:'center', gap:'10px', marginBottom:'1rem', paddingTop:'1rem', borderTop:'1px solid var(--border-subtle)' }}>
                      <div style={{ padding:'6px', background:'var(--accent-glow)', border:'1px solid rgba(82,183,136,0.25)', borderRadius:'8px' }}>
                        <BookOpen size={14} style={{ color:'var(--accent-1)' }} />
                      </div>
                      <div>
                        <h3 style={{ fontSize:'0.85rem', fontWeight:700, color:'var(--text-primary)', margin:0 }}>Retrieved Source Chunks</h3>
                        <p style={{ fontSize:'0.72rem', color:'var(--text-muted)', margin:0 }}>{answer.search_results.length} chunks · reranked by ms-marco cross-encoder · hybrid BM25+FAISS retrieval</p>
                      </div>
                    </div>

                    <div style={{ display:'flex', flexDirection:'column', gap:'10px' }}>
                      {answer.search_results.map((chunk, i) => (
                        <div key={i} style={{ padding:'1rem', borderRadius:'12px', background:'var(--bg-elevated)', border:'1px solid var(--border-base)' }}>
                          {/* Chunk header */}
                          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'10px' }}>
                            <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
                              <span style={{ width:'24px', height:'24px', borderRadius:'50%', background:'var(--accent-glow)', border:'1px solid rgba(82,183,136,0.3)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'0.72rem', fontWeight:700, color:'var(--accent-1)', flexShrink:0 }}>
                                {i + 1}
                              </span>
                              <span style={{ fontSize:'0.72rem', fontFamily:'monospace', color:'var(--text-muted)', background:'var(--bg-surface)', padding:'2px 7px', borderRadius:'5px', border:'1px solid var(--border-subtle)' }}>chunk #{chunk.chunk_id}</span>
                              <span style={{ fontSize:'0.72rem', color:'var(--text-muted)' }}>{chunk.word_count || '—'} words</span>
                            </div>
                            {chunk.score > 0 && (
                              <div style={{ display:'flex', alignItems:'center', gap:'8px' }}>
                                <div className="score-bar" style={{ width:'64px' }}>
                                  <div className="score-bar-fill" style={{ width:`${Math.min(100, chunk.score * 100)}%` }} />
                                </div>
                                <span style={{ fontSize:'0.72rem', color:'var(--accent-1)', fontWeight:600, minWidth:'36px', textAlign:'right' }}>
                                  {Math.round(chunk.score * 100)}%
                                </span>
                              </div>
                            )}
                          </div>
                          {/* Chunk text */}
                          <p style={{ fontSize:'0.84rem', lineHeight:1.7, color:'var(--text-secondary)', margin:0, fontStyle:'normal' }}>
                            {chunk.preview}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Empty state */}
            {!searched && !loading && (
              <div className="animate-fade-in-up" style={{ display:'flex', flexDirection:'column', alignItems:'center', padding:'5rem 1rem', textAlign:'center', opacity:0.65 }}>
                <div style={{ padding:'1.5rem', borderRadius:'20px', background:'var(--bg-surface)', border:'1px solid var(--border-base)', marginBottom:'1.25rem' }}>
                  <Bot size={44} style={{ color:'var(--text-muted)' }} />
                </div>
                <h3 style={{ fontSize:'1.1rem', fontWeight:700, color:'var(--text-primary)', marginBottom:'0.5rem' }}>Ready to Analyze</h3>
                <p style={{ fontSize:'0.875rem', color:'var(--text-secondary)', maxWidth:'380px', lineHeight:1.65, marginBottom:'2rem' }}>
                  Ask any question — the AI retrieves the most relevant textbook sections and synthesizes a comprehensive answer.
                </p>

                {/* Pipeline feature cards */}
                <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:'12px', width:'100%', maxWidth:'520px' }}>
                  {[
                    { icon:<Layers size={16}/>, label:'Hybrid Retrieval', desc:'BM25 + FAISS + RRF' },
                    { icon:<Brain size={16}/>, label:'Reranking', desc:'ms-marco MiniLM L-12' },
                    { icon:<Zap size={16}/>, label:'Groq LLM', desc:'Llama 3.3 70B · ~500 tok/s' },
                  ].map((f, i) => (
                    <div key={i} style={{ padding:'0.9rem', borderRadius:'12px', background:'var(--bg-surface)', border:'1px solid var(--border-base)', textAlign:'center' }}>
                      <div style={{ color:'var(--accent-1)', marginBottom:'6px', display:'flex', justifyContent:'center' }}>{f.icon}</div>
                      <p style={{ fontSize:'0.72rem', fontWeight:700, color:'var(--text-primary)', margin:'0 0 2px' }}>{f.label}</p>
                      <p style={{ fontSize:'0.68rem', color:'var(--text-muted)', margin:0 }}>{f.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
};

export default LLMAnswerPage;