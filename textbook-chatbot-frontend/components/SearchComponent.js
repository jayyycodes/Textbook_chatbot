"use client"
import React, { useState, useEffect } from 'react';
import { Search, BookOpen, AlertCircle, RefreshCw, Lightbulb, Zap, Clock, Hash, Moon, Sun, Bot, ArrowRight, Sparkles } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

const SearchComponent = () => {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchAttempted, setSearchAttempted] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Replace the hardcoded searchSuggestions array with a dynamic object
  const searchSuggestionsByTextbook = {
    'computer_networks': [
      "What is a computer network?",
      "Explain the OSI model",
      "What is IP addressing?",
      "Define packet switching",
      "What is TCP/IP?",
      "How does DNS work?",
      "What are network protocols?",
      "Explain network topologies"
    ],
    'economics': [
      "What is demand?",
      "Explain the law of supply and demand",
      "What is GDP?",
      "Define inflation",
      "What are market structures?",
      "What is opportunity cost?",
      "Explain economic equilibrium",
      "What is monetary policy?"
    ],
    'intro_to_ml': [
      "What is machine learning?",
      "Explain supervised learning",
      "Introduction to data science",
      "How does classification work?",
      "What is regression?",
      "Explain feature selection"
    ]
  };

  const [selectedTextbook, setSelectedTextbook] = useState('intro_to_ml'); // Default textbook
  const [availableTextbooks] = useState([
    { id: 'intro_to_ml', name: 'Introduction to Machine Learning', description: 'ML algorithms and concepts' },
    { id: 'computer_networks', name: 'Computer Networks', description: 'Network protocols and systems' }
  ]);

  const getCurrentSuggestions = () => {
    return searchSuggestionsByTextbook[selectedTextbook] || searchSuggestionsByTextbook['intro_to_ml'];
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
    setResults([]);
    setSearchAttempted(true);

    try {
      const endpoint = 'http://localhost:5000/search';

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery.trim(),
          top_k: 5,
          textbook: textbook
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // Handle Raw Search response
      if (data && data.results && Array.isArray(data.results)) {
        setResults(data.results);
      } else if (data && data.error) {
        throw new Error(data.error);
      } else {
        setResults([]);
      }

    } catch (err) {
      console.error('Search error:', err);
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    await performSearch(query, selectedTextbook);

    // Update URL with query and textbook parameters
    if (query.trim()) {
      const newUrl = `/search?textbook=${selectedTextbook}&q=${encodeURIComponent(query.trim())}`;
      window.history.pushState({}, '', newUrl);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setQuery(suggestion);
    performSearch(suggestion, selectedTextbook);

    // Update URL
    const newUrl = `/search?textbook=${selectedTextbook}&q=${encodeURIComponent(suggestion)}`;
    window.history.pushState({}, '', newUrl);
  };

  const truncateText = (text, maxLength = 500) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
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
            href={`/search/answer?textbook=${selectedTextbook}&q=${encodeURIComponent(query)}`}
            className={`hidden md:flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${isDark ? 'bg-white/10 hover:bg-white/20 text-white' : 'bg-black/5 hover:bg-black/10 text-black'}`}
          >
            <Sparkles size={16} />
            <span>Get AI Answer</span>
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
                    placeholder="Search topics..."
                    className={`w-full pl-10 pr-4 py-3 rounded-xl outline-none transition-all ${themeClasses.inputBg} ${themeClasses.text}`}
                  />
                  <Search className={`absolute left-3 top-1/2 -translate-y-1/2 ${themeClasses.textMuted}`} size={18} />
                </div>

                <button
                  onClick={handleSearch}
                  disabled={loading || !query.trim()}
                  className={`w-full py-3 rounded-xl font-medium text-white transition-all active:scale-95 ${themeClasses.accent} ${themeClasses.accentHover} disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2`}
                >
                  {loading ? <RefreshCw className="animate-spin" size={18} /> : <Search size={18} />}
                  <span>Search</span>
                </button>
              </div>
            </div>

            {/* Suggestions */}
            {!searchAttempted && (
              <div className={`p-6 rounded-2xl backdrop-blur-xl border ${themeClasses.cardBg} shadow-sm`}>
                <div className="flex items-center gap-2 mb-4">
                  <Lightbulb className="text-yellow-500" size={18} />
                  <h3 className={`font-semibold ${themeClasses.text}`}>Try asking...</h3>
                </div>
                <div className="space-y-2">
                  {getCurrentSuggestions().map((suggestion, index) => (
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

          {/* Results Area */}
          <div className="lg:col-span-8 xl:col-span-9">

            {/* Status Bar */}
            {searchAttempted && !loading && !error && (
              <div className="flex items-center justify-between mb-6 px-2">
                <span className={`text-sm ${themeClasses.textSecondary}`}>
                  Found <span className={`font-bold ${themeClasses.text}`}>{results.length}</span> results for "<span className="italic">{query}</span>"
                </span>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className={`p-6 rounded-2xl border border-red-500/20 bg-red-500/5 mb-6`}>
                <div className="flex items-center gap-3 text-red-500 mb-2">
                  <AlertCircle size={20} />
                  <h3 className="font-semibold">Search Error</h3>
                </div>
                <p className={`text-sm ${themeClasses.textSecondary}`}>{error}</p>
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div className="flex flex-col items-center justify-center py-20">
                <div className="relative w-16 h-16 mb-6">
                  <div className="absolute inset-0 rounded-full border-4 border-blue-500/20"></div>
                  <div className="absolute inset-0 rounded-full border-4 border-blue-500 border-t-transparent animate-spin"></div>
                </div>
                <p className={`${themeClasses.textSecondary} animate-pulse`}>Searching your textbook...</p>
              </div>
            )}

            {/* Results List */}
            <div className="space-y-4">
              {results.map((result, index) => (
                <div
                  key={result.chunk_id || index}
                  className={`group p-6 rounded-2xl backdrop-blur-sm border transition-all duration-300 hover:shadow-lg hover:-translate-y-1 ${themeClasses.cardBg} ${themeClasses.border}`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className={`flex items-center justify-center w-8 h-8 rounded-full text-xs font-bold ${isDark ? 'bg-blue-900/50 text-blue-200' : 'bg-blue-100 text-blue-700'}`}>
                      {index + 1}
                    </div>
                    <div className="flex items-center gap-2">
                      {result.score && (
                        <span className={`text-xs px-2 py-1 rounded-full ${isDark ? 'bg-green-900/30 text-green-400' : 'bg-green-100 text-green-700'}`}>
                          {Math.round(result.score * 100)}% match
                        </span>
                      )}
                    </div>
                  </div>

                  <p className={`text-sm md:text-base leading-relaxed mb-4 ${themeClasses.text}`}>
                    {truncateText(result.content, 600)}
                  </p>

                  <div className={`pt-4 border-t flex items-center gap-4 text-xs ${isDark ? 'border-white/5' : 'border-black/5'} ${themeClasses.textMuted}`}>
                    <div className="flex items-center gap-1">
                      <Hash size={12} />
                      <span>ID: {result.chunk_id}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Clock size={12} />
                      <span>{result.word_count || 'N/A'} words</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Empty State */}
            {!searchAttempted && !loading && (
              <div className="flex flex-col items-center justify-center py-20 text-center opacity-50">
                <Search size={48} className={`mb-4 ${themeClasses.textMuted}`} />
                <p className={themeClasses.textSecondary}>Start by typing a question or selecting a topic.</p>
              </div>
            )}

            {/* No Results */}
            {searchAttempted && !loading && results.length === 0 && !error && (
              <div className="flex flex-col items-center justify-center py-20 text-center">
                <div className={`p-4 rounded-full mb-4 ${isDark ? 'bg-yellow-900/20 text-yellow-500' : 'bg-yellow-100 text-yellow-600'}`} />
                <AlertCircle size={32} className={isDark ? 'text-yellow-500' : 'text-yellow-600'} />
                <h3 className={`text-lg font-semibold mb-2 ${themeClasses.text}`}>No results found</h3>
                <p className={`max-w-md ${themeClasses.textSecondary}`}>
                  Try adjusting your search terms or selecting a different textbook.
                </p>
              </div>
            )}

          </div>
        </div>
      </main>
    </div>
  );
};

export default SearchComponent;
