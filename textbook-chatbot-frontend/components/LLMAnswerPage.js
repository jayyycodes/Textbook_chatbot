"use client"
import React, { useState, useEffect } from 'react';
import { Search, BookOpen, AlertCircle, RefreshCw, Lightbulb, Zap, Clock, Hash, Moon, Sun, Bot, ChevronDown, ChevronUp, ExternalLink, ArrowLeft, Copy, CheckCircle, Sparkles } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

const LLMAnswerPage = () => {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchAttempted, setSearchAttempted] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const [copied, setCopied] = useState(false);
  const [mounted, setMounted] = useState(false);

  const [selectedTextbook, setSelectedTextbook] = useState('intro_to_ml');// Default textbook
  const [availableTextbooks] = useState([
    { id: 'intro_to_ml', name: 'Introduction to Machine Learning', description: 'ML algorithms and concepts' },
    { id: 'computer_networks', name: 'Computer Networks', description: 'Network protocols and systems' }
  ]);

  const answerSuggestionsByTextbook = {
    'computer_networks': [
      "What is a computer network and how does it work?",
      "Explain the OSI model and its seven layers in detail",
      "How does IP addressing work and what are the different types?",
      "What is packet switching and how does it differ from circuit switching?",
      "Explain the TCP/IP protocol suite and its importance",
      "How does DNS work and what is its role in networking?",
      "What are network protocols and why are they essential?",
      "Compare different network topologies and their advantages"
    ],
    'economics': [
      "What is demand and how does it affect market prices?",
      "Explain the law of supply and demand with real-world examples",
      "What is GDP and how is it calculated and interpreted?",
      "Define inflation and explain its causes and effects on the economy",
      "What are market structures and how do they influence competition?",
      "What is opportunity cost and how does it guide economic decisions?",
      "Explain economic equilibrium and how markets reach balance",
      "What is monetary policy and how do central banks use it?"
    ],

    // Default fallback for any unmapped textbooks
    'intro_to_ml': [
      "What is machine learning and how does it work?",
      "Explain the difference between supervised and unsupervised learning",
      "How do neural networks process information?",
      "What are the main types of machine learning algorithms?",
      "Explain the concept of overfitting in machine learning",
      "What is feature engineering and why is it important?",
      "How does gradient descent optimization work?",
      "What are the applications of deep learning?"
    ]
  };

  const getCurrentAnswerSuggestions = () => {
    return answerSuggestionsByTextbook[selectedTextbook] || answerSuggestionsByTextbook['intro_to_ml'];
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

  // Extract query from URL and auto-search on component mount
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const queryParam = urlParams.get('q');
    const textbookParam = urlParams.get('textbook');

    if (textbookParam && availableTextbooks.find(t => t.id === textbookParam)) {
      setSelectedTextbook(textbookParam);
    }

    if (queryParam) {
      setQuery(queryParam);
      // Wait for textbook to be set before searching
      setTimeout(() => performSearch(queryParam, textbookParam || selectedTextbook), 100);
    }
  }, []);

  const performSearch = async (searchQuery, textbook = selectedTextbook) => {
    if (!searchQuery || !searchQuery.trim()) return;

    setLoading(true);
    setError('');
    setAnswer(null);
    setSearchAttempted(true);
    setShowSources(false);

    try {
      const response = await fetch('http://localhost:5000/search/answer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery.trim(),
          textbook: textbook  // Add textbook parameter
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setAnswer(data);

    } catch (err) {
      console.error('Answer generation error:', err);
      setError(err.message || 'An unexpected error occurred while generating the answer');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    await performSearch(query, selectedTextbook);

    // Update URL with query and textbook parameters
    if (query.trim()) {
      const newUrl = `/search/answer?textbook=${selectedTextbook}&q=${encodeURIComponent(query.trim())}`;
      window.history.pushState({}, '', newUrl);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setQuery(suggestion);
    performSearch(suggestion, selectedTextbook);

    // Update URL
    const newUrl = `/search/answer?textbook=${selectedTextbook}&q=${encodeURIComponent(suggestion)}`;
    window.history.pushState({}, '', newUrl);
  };

  const handleCopyAnswer = async () => {
    if (answer && answer.answer) {
      try {
        await navigator.clipboard.writeText(answer.answer);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (err) {
        console.error('Failed to copy:', err);
      }
    }
  };

  const formatDuration = (ms) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  if (!mounted) return null;

  const themeClasses = {
    bg: isDark ? 'bg-[#0a0a0a]' : 'bg-[#fafafa]',
    text: isDark ? 'text-gray-100' : 'text-gray-900',
    textSecondary: isDark ? 'text-gray-400' : 'text-gray-500',
    textMuted: isDark ? 'text-gray-500' : 'text-gray-400',
    cardBg: isDark ? 'bg-white/5 border-white/10' : 'bg-white border-gray-100',
    inputBg: isDark ? 'bg-white/5 border-white/10 focus:border-blue-500' : 'bg-white border-gray-200 focus:border-blue-500',
    accent: 'bg-blue-600',
    accentHover: 'hover:bg-blue-700',
    border: isDark ? 'border-white/10' : 'border-gray-100',
    answerBg: isDark ? 'bg-gradient-to-r from-blue-900/10 to-purple-900/10' : 'bg-gradient-to-r from-blue-50/50 to-purple-50/50',
  };

  return (
    <div className={`min-h-screen flex flex-col transition-colors duration-500 ${themeClasses.bg} relative overflow-hidden`}>

      {/* Abstract Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none fixed">
        <div className={`absolute top-[-20%] left-[-10%] w-[50vw] h-[50vw] rounded-full blur-[120px] opacity-20 animate-pulse ${isDark ? 'bg-blue-900/40' : 'bg-blue-200/60'}`} />
        <div className={`absolute bottom-[-20%] right-[-10%] w-[50vw] h-[50vw] rounded-full blur-[120px] opacity-20 animate-pulse delay-1000 ${isDark ? 'bg-indigo-900/40' : 'bg-indigo-200/60'}`} />
      </div>

      {/* Navigation */}
      <nav className="w-full p-6 flex justify-between items-center z-10 sticky top-0 backdrop-blur-md border-b border-transparent">
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => router.push('/')}>
          <div className={`p-2 rounded-xl ${isDark ? 'bg-white/10' : 'bg-black/5'}`}>
            <BookOpen size={20} className={isDark ? 'text-white' : 'text-black'} />
          </div>
          <span className={`font-bold text-lg tracking-tight ${themeClasses.text}`}>LearnLens</span>
        </div>

        <div className="flex items-center gap-4">
          <Link
            href={`/search?textbook=${selectedTextbook}`}
            className={`hidden md:flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${isDark ? 'bg-white/10 hover:bg-white/20 text-white' : 'bg-black/5 hover:bg-black/10 text-black'}`}
          >
            <Search size={16} />
            <span>Raw Search</span>
          </Link>
          <button
            onClick={toggleTheme}
            className={`p-2 rounded-full transition-all duration-300 ${themeClasses.cardBg} backdrop-blur-md border hover:scale-110`}
          >
            {isDark ? <Sun size={18} className="text-yellow-400" /> : <Moon size={18} className="text-slate-600" />}
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8 z-10">
        <div className="grid lg:grid-cols-12 gap-8">

          {/* Sidebar */}
          <div className="lg:col-span-4 xl:col-span-3 space-y-6">

            {/* Search Controls */}
            <div className={`p-6 rounded-2xl backdrop-blur-xl border ${themeClasses.cardBg} shadow-sm`}>
              <div className="space-y-4">
                {/* Textbook Select */}
                <div>
                  <label className={`text-xs font-semibold uppercase tracking-wider ${themeClasses.textMuted} mb-2 block`}>Textbook</label>
                  <select
                    value={selectedTextbook}
                    onChange={(e) => setSelectedTextbook(e.target.value)}
                    className={`w-full px-3 py-2 rounded-lg text-sm outline-none transition-all ${themeClasses.inputBg} ${themeClasses.text}`}
                  >
                    {availableTextbooks.map((textbook) => (
                      <option key={textbook.id} value={textbook.id} className="text-black">
                        {textbook.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Search Input */}
                <div className="relative">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSearch(e)}
                    placeholder="Ask a question..."
                    className={`w-full pl-10 pr-4 py-3 rounded-xl outline-none transition-all ${themeClasses.inputBg} ${themeClasses.text}`}
                  />
                  <Bot className={`absolute left-3 top-1/2 -translate-y-1/2 ${themeClasses.textMuted}`} size={18} />
                </div>

                <button
                  onClick={handleSearch}
                  disabled={loading || !query.trim()}
                  className={`w-full py-3 rounded-xl font-medium text-white transition-all active:scale-95 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2`}
                >
                  {loading ? <RefreshCw className="animate-spin" size={18} /> : <Sparkles size={18} />}
                  <span>Generate Answer</span>
                </button>
              </div>
            </div>

            {/* Suggestions */}
            {(!searchAttempted || (answer && !loading)) && (
              <div className={`p-6 rounded-2xl backdrop-blur-xl border ${themeClasses.cardBg} shadow-sm`}>
                <div className="flex items-center gap-2 mb-4">
                  <Lightbulb className="text-yellow-500" size={18} />
                  <h3 className={`font-semibold ${themeClasses.text}`}>Try asking...</h3>
                </div>
                <div className="space-y-2">
                  {getCurrentAnswerSuggestions().map((suggestion, index) => (
                    <button
                      key={index}
                      onClick={() => handleSuggestionClick(suggestion)}
                      className={`w-full text-left px-3 py-2 text-sm rounded-lg transition-colors hover:bg-blue-500/10 ${themeClasses.textSecondary} hover:${themeClasses.text}`}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Main Answer Area */}
          <div className="lg:col-span-8 xl:col-span-9">

            {/* Search Status */}
            {searchAttempted && !loading && !error && answer && (
              <div className="flex items-center justify-between mb-6 px-2">
                <div className="flex items-center gap-2">
                  <Sparkles className="text-purple-500" size={16} />
                  <span className={`text-sm ${themeClasses.textSecondary}`}>
                    AI Answer for "<span className={`font-bold ${themeClasses.text}`}>{query}</span>"
                  </span>
                </div>
                {answer.timing && (
                  <div className={`flex items-center gap-1 text-xs ${themeClasses.textMuted}`}>
                    <Clock size={12} />
                    <span>{formatDuration(answer.timing.total_duration)}</span>
                  </div>
                )}
              </div>
            )}

            {/* Error */}
            {error && (
              <div className={`p-6 rounded-2xl border border-red-500/20 bg-red-500/5 mb-6`}>
                <div className="flex items-center gap-3 text-red-500 mb-2">
                  <AlertCircle size={20} />
                  <h3 className="font-semibold">Generation Failed</h3>
                </div>
                <p className={`text-sm ${themeClasses.textSecondary}`}>{error}</p>
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div className="flex flex-col items-center justify-center py-20">
                <div className="relative w-16 h-16 mb-6">
                  <div className="absolute inset-0 rounded-full border-4 border-purple-500/20"></div>
                  <div className="absolute inset-0 rounded-full border-4 border-purple-500 border-t-transparent animate-spin"></div>
                </div>
                <p className={`${themeClasses.textSecondary} animate-pulse`}>Analyzing textbook content...</p>
              </div>
            )}

            {/* Answer Display */}
            {answer && answer.answer && (
              <div className={`rounded-2xl backdrop-blur-sm border overflow-hidden mb-6 ${themeClasses.cardBg} ${themeClasses.border}`}>
                {/* Header */}
                <div className={`p-6 border-b ${themeClasses.border} ${themeClasses.answerBg}`}>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-4">
                      <div className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl shadow-lg shadow-purple-500/20">
                        <Bot className="text-white" size={24} />
                      </div>
                      <div>
                        <h2 className={`text-xl font-bold ${themeClasses.text}`}>AI Analysis</h2>
                        <div className={`flex items-center gap-2 text-sm ${themeClasses.textMuted}`}>
                          <span>Based on {answer.chunks_processed || 0} sources</span>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={handleCopyAnswer}
                      className={`p-2 rounded-lg transition-all ${isDark ? 'hover:bg-white/10' : 'hover:bg-black/5'}`}
                      title="Copy answer"
                    >
                      {copied ? <CheckCircle className="text-green-500" size={20} /> : <Copy className={themeClasses.textMuted} size={20} />}
                    </button>
                  </div>
                </div>

                {/* Content */}
                <div className="p-8">
                  <div className={`prose max-w-none ${isDark ? 'prose-invert' : ''}`}>
                    <div className={`whitespace-pre-wrap leading-relaxed text-base lg:text-lg ${themeClasses.text}`}>
                      {answer.answer}
                    </div>
                  </div>
                </div>

                {/* Sources Toggle */}
                {answer.search_results && answer.search_results.length > 0 && (
                  <div className={`border-t ${themeClasses.border}`}>
                    <button
                      onClick={() => setShowSources(!showSources)}
                      className={`w-full p-4 flex items-center justify-between transition-colors ${isDark ? 'hover:bg-white/5' : 'hover:bg-black/5'}`}
                    >
                      <div className={`flex items-center gap-2 text-sm font-medium ${themeClasses.textSecondary}`}>
                        <BookOpen size={16} />
                        <span>View Source Material</span>
                      </div>
                      {showSources ? <ChevronUp size={16} className={themeClasses.textMuted} /> : <ChevronDown size={16} className={themeClasses.textMuted} />}
                    </button>

                    {showSources && (
                      <div className={`p-4 space-y-3 ${isDark ? 'bg-black/20' : 'bg-gray-50/50'}`}>
                        {answer.search_results.map((chunk, index) => (
                          <div key={index} className={`p-4 rounded-xl border ${themeClasses.cardBg} ${themeClasses.border}`}>
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <span className={`flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${isDark ? 'bg-blue-900/50 text-blue-200' : 'bg-blue-100 text-blue-700'}`}>
                                  {index + 1}
                                </span>
                                <span className={`text-xs font-mono ${themeClasses.textMuted}`}>ID: {chunk.chunk_id}</span>
                              </div>
                              {chunk.score && (
                                <span className={`text-xs px-2 py-1 rounded-full ${isDark ? 'bg-green-900/30 text-green-400' : 'bg-green-100 text-green-700'}`}>
                                  {Math.round(chunk.score * 100)}% match
                                </span>
                              )}
                            </div>
                            <p className={`text-sm leading-relaxed ${themeClasses.textSecondary}`}>
                              {chunk.preview}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Empty State */}
            {!searchAttempted && !loading && (
              <div className="flex flex-col items-center justify-center py-20 text-center opacity-50">
                <Bot size={64} className={`mb-6 ${themeClasses.textMuted}`} />
                <h3 className={`text-xl font-semibold mb-2 ${themeClasses.text}`}>Ready to Analyze</h3>
                <p className={`max-w-md ${themeClasses.textSecondary}`}>
                  Ask any question about your textbook. AI will analyze the content and provide a comprehensive answer with citations.
                </p>
              </div>
            )}

          </div>
        </div>
      </main>
    </div>
  );
};

export default LLMAnswerPage;