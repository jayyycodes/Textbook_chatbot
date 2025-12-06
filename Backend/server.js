const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();

// Textbook configuration
const TEXTBOOK_CONFIG = {
    'intro_ml': {
        name: 'Introduction to Machine Learning',
        description: 'ML algorithms and concepts'
    },
    'Computer_Networks': {
        name: 'Computer Networks',
        description: 'Computer networking fundamentals'
    },
    'economics': {
        name: 'Economics',
        description: 'Economic principles and theories'
    }
};

// CORS configuration
app.use(cors({
    origin: [
        'http://localhost:3000',
        'http://127.0.0.1:3000',
        'http://localhost:3001',
        'http://127.0.0.1:3001'
    ],
    credentials: true,
    optionsSuccessStatus: 200
}));

app.use(express.json());

// Enhanced request logging
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
    next();
});

// Cache for Python command
let workingPythonCommand = null;

/**
 * Find working Python command
 */
async function findWorkingPythonCommand() {
    if (workingPythonCommand) {
        return workingPythonCommand;
    }

    // Prioritize virtual environment
    const venvPython = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
    const pythonCommands = [venvPython, 'python', 'python3', 'py'];

    for (const command of pythonCommands) {
        try {
            // Skip if file path doesn't exist (for absolute paths)
            if (path.isAbsolute(command) && !fs.existsSync(command)) {
                continue;
            }

            const result = await new Promise((resolve) => {
                const process = spawn(command, ['--version'], { timeout: 3000 });
                process.on('close', (code) => resolve(code === 0));
                process.on('error', () => resolve(false));
            });

            if (result) {
                workingPythonCommand = command;
                console.log(`[INFO] Using Python command: ${command}`);
                return command;
            }
        } catch (error) {
            continue;
        }
    }

    console.error(`[ERROR] No working Python found`);
    return null;
}

/**
 * Find search_faiss.py script path
 */
function findScriptPath() {
    const possiblePaths = [
        path.join(__dirname, 'embeddings', 'search_faiss.py'),
        path.join(__dirname, 'search_faiss.py')
    ];

    for (const scriptPath of possiblePaths) {
        if (fs.existsSync(scriptPath)) {
            console.log(`[INFO] Found script at: ${scriptPath}`);
            return scriptPath;
        }
    }

    console.error(`[ERROR] search_faiss.py not found`);
    return null;
}

/**
 * Execute search_faiss.py with given mode
 */
async function executeSearch(pythonCommand, scriptPath, textbook, query, mode = 'raw') {
    const args = [
        scriptPath,
        'query',
        '--id', textbook,
        '--question', query.trim(),
        '--mode', mode
    ];

    console.log(`[DEBUG] Executing: ${pythonCommand} ${args.join(' ')}`);

    const pythonProcess = spawn(pythonCommand, args, {
        cwd: path.dirname(scriptPath),
        env: {
            ...process.env,
            PYTHONUNBUFFERED: '1',
            PYTHONIOENCODING: 'utf-8'
        },
        timeout: 180000  // 3 minutes timeout
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
    });

    return new Promise((resolve, reject) => {
        pythonProcess.on('close', (code) => {
            console.log(`[DEBUG] Python exit code: ${code}`);
            console.log(`[DEBUG] Python stdout length: ${stdout.length}`);

            if (code === 0) {
                resolve({ success: true, output: stdout, stderr });
            } else {
                reject({ success: false, code, stderr: stderr.trim(), stdout: stdout.trim() });
            }
        });

        pythonProcess.on('error', (error) => {
            reject({ success: false, error: error.message });
        });
    });
}

/**
 * Parse the search_faiss.py output format
 */
function parseSearchOutput(output, textbook, query) {
    try {
        const lines = output.split('\n');
        let bookRelevance = null;
        let confidence = null;
        let chunksUsed = 0;
        let chunkIds = [];
        let bestRerankScore = null;
        let answerStarted = false;
        let answerLines = [];

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            if (line.includes('BOOK RELEVANCE:')) {
                bookRelevance = parseFloat(line.split(':')[1].trim());
            } else if (line.includes('CONFIDENCE:')) {
                confidence = line.split(':')[1].trim();
            } else if (line.includes('CHUNKS USED:')) {
                chunksUsed = parseInt(line.split(':')[1].trim());
            } else if (line.includes('CHUNK_IDS:')) {
                try {
                    const idsStr = line.split('CHUNK_IDS:')[1].trim();
                    chunkIds = JSON.parse(idsStr);
                } catch (e) {
                    console.log('[WARN] Could not parse chunk IDs:', line);
                    chunkIds = [];
                }
            } else if (line.includes('BEST RERANK SCORE:')) {
                bestRerankScore = parseFloat(line.split(':')[1].trim());
            } else if (line.includes('ANSWER:')) {
                answerStarted = true;
            } else if (answerStarted && line.includes('----------------------------------------------------------------------')) {
                if (answerLines.length === 0) continue; // Skip first separator
                else break; // Stop at second separator
            } else if (answerStarted) {
                answerLines.push(line);
            }
        }

        const answer = answerLines.join('\n').trim();

        // Check if it's an error message
        if (answer.includes('The textbook does not contain relevant information')) {
            return {
                error: 'Not Relevant',
                message: answer,
                book_relevance: bookRelevance,
                textbook: textbook,
                query: query,
                results: []
            };
        }

        // Parse evidence chunks from answer (for RAW mode)
        const evidencePattern = /\[Evidence (\d+) - Chunk ID: ([^\]]+)\]\s+([\s\S]*?)(?=\[Evidence|\n\n$|$)/g;
        const results = [];
        let match;

        while ((match = evidencePattern.exec(answer)) !== null) {
            results.push({
                rank: parseInt(match[1]),
                chunk_id: match[2],
                content: match[3].trim(),
                word_count: match[3].trim().split(' ').length
            });
        }

        return {
            query: query,
            textbook: textbook,
            textbook_name: TEXTBOOK_CONFIG[textbook]?.name || textbook,
            book_relevance: bookRelevance,
            confidence: confidence,
            best_rerank_score: bestRerankScore,
            chunks_used: chunksUsed,
            chunk_ids: chunkIds,
            total_results: results.length,
            results: results,
            answer: answer  // Full answer (structured for RAW, rewritten for LLM)
        };
    } catch (error) {
        console.error('[ERROR] Failed to parse output:', error.message);
        console.error('[DEBUG] Output was:', output.substring(0, 500));
        throw new Error(`Parse error: ${error.message}`);
    }
}

/**
 * POST /search - Semantic search using search_faiss.py (RAW mode)
 * Returns evidence chunks without LLM rewriting
 */
app.post('/search', async (req, res) => {
    const startTime = Date.now();

    try {
        const { query, top_k, textbook } = req.body;

        // Input validation
        if (!query || typeof query !== 'string' || query.trim().length === 0) {
            return res.status(400).json({
                error: 'Bad Request',
                message: 'Query is required and must be a non-empty string'
            });
        }

        // Map textbook values
        const textbookMapping = {
            'computer_networks': 'Computer_Networks',
            'ml': 'intro_ml',
            'machine_learning': 'intro_ml',
            'intro_to_ml': 'intro_ml',
            'economics': 'economics'
        };

        const selectedTextbook = textbookMapping[textbook?.toLowerCase()] || textbook || 'intro_ml';

        console.log(`[${new Date().toISOString()}] Search (RAW): "${query.substring(0, 50)}..." in ${selectedTextbook}`);

        const scriptPath = findScriptPath();
        if (!scriptPath) {
            return res.status(500).json({
                error: 'Configuration Error',
                message: 'Search script not found'
            });
        }

        const pythonCommand = await findWorkingPythonCommand();
        if (!pythonCommand) {
            return res.status(500).json({
                error: 'Python Not Available',
                message: 'No working Python installation found'
            });
        }

        // Execute search in RAW mode
        const result = await executeSearch(pythonCommand, scriptPath, selectedTextbook, query, 'raw');

        // Parse output
        const parsedResult = parseSearchOutput(result.output, selectedTextbook, query);
        parsedResult.duration = `${Date.now() - startTime}ms`;

        console.log(`[${new Date().toISOString()}] Search completed in ${parsedResult.duration}`);
        res.status(200).json(parsedResult);

    } catch (error) {
        const duration = Date.now() - startTime;
        console.error(`[${new Date().toISOString()}] Search failed after ${duration}ms:`, error);

        res.status(500).json({
            error: 'Search Failed',
            message: error.message || 'An unexpected error occurred',
            textbook: req.body.textbook,
            duration: `${duration}ms`
        });
    }
});

/**
 * POST /search/answer - Generate LLM answer using search_faiss.py (LLM mode)
 * Uses built-in LLM generation from search_faiss.py
 */
app.post('/search/answer', async (req, res) => {
    const startTime = Date.now();

    try {
        const { query, textbook } = req.body;

        // Input validation
        if (!query || typeof query !== 'string' || query.trim().length === 0) {
            return res.status(400).json({
                error: 'Bad Request',
                message: 'Query is required'
            });
        }

        // Map textbook
        const textbookMapping = {
            'computer_networks': 'Computer_Networks',
            'ml': 'intro_ml',
            'machine_learning': 'intro_ml',
            'intro_to_ml': 'intro_ml',
            'economics': 'economics'
        };

        const selectedTextbook = textbookMapping[textbook?.toLowerCase()] || textbook || 'intro_ml';

        console.log(`[${new Date().toISOString()}] LLM Answer: "${query.substring(0, 50)}..." in ${selectedTextbook}`);

        const scriptPath = findScriptPath();
        if (!scriptPath) {
            return res.status(500).json({
                error: 'Configuration Error',
                message: 'Search script not found'
            });
        }

        const pythonCommand = await findWorkingPythonCommand();
        if (!pythonCommand) {
            return res.status(500).json({
                error: 'Python Not Available'
            });
        }

        // Execute search in LLM mode - this calls llm_answer.py internally
        const result = await executeSearch(pythonCommand, scriptPath, selectedTextbook, query, 'llm');

        // Parse output
        const parsedResult = parseSearchOutput(result.output, selectedTextbook, query);
        parsedResult.duration = `${Date.now() - startTime}ms`;
        parsedResult.mode = 'llm';

        console.log(`[${new Date().toISOString()}] LLM Answer completed in ${parsedResult.duration}`);
        res.status(200).json(parsedResult);

    } catch (error) {
        const duration = Date.now() - startTime;
        console.error(`[${new Date().toISOString()}] LLM Answer failed:`, error);

        if (error.stderr) {
            console.error('[DEBUG] Script Stderr:', error.stderr);
        }

        res.status(500).json({
            error: 'LLM Answer Failed',
            message: error.message || 'Failed to generate answer',
            query: req.body.query,
            duration: `${duration}ms`
        });
    }
});

/**
 * GET /health - Health check
 */
app.get('/health', async (req, res) => {
    const scriptPath = findScriptPath();
    const pythonCommand = await findWorkingPythonCommand();

    res.status(200).json({
        status: scriptPath && pythonCommand ? 'healthy' : 'unhealthy',
        script_found: !!scriptPath,
        python_available: !!pythonCommand,
        timestamp: new Date().toISOString()
    });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(`[${new Date().toISOString()}] Error:`, err);
    res.status(500).json({
        error: 'Internal Server Error',
        timestamp: new Date().toISOString()
    });
});

// Start server
const PORT = process.env.PORT || 5000;
const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 Textbook Chatbot API running on port ${PORT}`);
    console.log(`📍 Search (RAW): POST http://localhost:${PORT}/search`);
    console.log(`🤖 LLM Answer: POST http://localhost:${PORT}/search/answer`);
    console.log(`❤️ Health: GET http://localhost:${PORT}/health`);

    // Initial validation
    setTimeout(async () => {
        const scriptPath = findScriptPath();
        const pythonCommand = await findWorkingPythonCommand();
        console.log(scriptPath && pythonCommand ? '✅ System ready' : '❌ System validation failed');
    }, 1000);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('Shutting down gracefully...');
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});

module.exports = app;