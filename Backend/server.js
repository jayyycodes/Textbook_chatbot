const express = require('express');
const cors = require('cors');
const http = require('http');

const app = express();

// =========================================================================
// CONFIGURATION
// =========================================================================

// FastAPI backend URL (persistent Python RAG server)
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8000';

// =========================================================================
// MIDDLEWARE
// =========================================================================

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

// =========================================================================
// HELPER: Proxy to FastAPI
// =========================================================================

/**
 * Forward a request to the FastAPI backend and return the response.
 * @param {string} method - HTTP method ('GET' or 'POST')
 * @param {string} path - Path on the FastAPI server (e.g., '/search')
 * @param {object|null} body - Request body for POST requests
 * @returns {Promise<{statusCode: number, data: object}>}
 */
function proxyToFastAPI(method, path, body = null) {
    return new Promise((resolve, reject) => {
        const url = new URL(path, FASTAPI_URL);
        
        const options = {
            hostname: url.hostname,
            port: url.port,
            path: url.pathname,
            method: method,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            timeout: 120000, // 2 minute timeout
        };

        const req = http.request(options, (res) => {
            let data = '';
            res.on('data', (chunk) => data += chunk);
            res.on('end', () => {
                try {
                    const parsed = JSON.parse(data);
                    resolve({ statusCode: res.statusCode, data: parsed });
                } catch (e) {
                    resolve({ statusCode: res.statusCode, data: { raw: data } });
                }
            });
        });

        req.on('error', (error) => {
            reject(error);
        });

        req.on('timeout', () => {
            req.destroy();
            reject(new Error('Request to FastAPI timed out'));
        });

        if (body) {
            req.write(JSON.stringify(body));
        }

        req.end();
    });
}

// =========================================================================
// ROUTES
// =========================================================================

/**
 * GET /health — Health check (proxied to FastAPI)
 */
app.get('/health', async (req, res) => {
    try {
        const result = await proxyToFastAPI('GET', '/health');
        res.status(result.statusCode).json({
            ...result.data,
            proxy: 'node.js',
            fastapi_url: FASTAPI_URL,
        });
    } catch (error) {
        res.status(503).json({
            status: 'unhealthy',
            error: 'FastAPI backend is not reachable',
            message: error.message,
            fastapi_url: FASTAPI_URL,
            hint: 'Start the FastAPI server: cd embeddings && python api_server.py',
        });
    }
});

/**
 * GET /textbooks — List available textbooks (proxied to FastAPI)
 */
app.get('/textbooks', async (req, res) => {
    try {
        const result = await proxyToFastAPI('GET', '/textbooks');
        res.status(result.statusCode).json(result.data);
    } catch (error) {
        console.error(`[${new Date().toISOString()}] /textbooks failed:`, error.message);
        res.status(503).json({
            error: 'FastAPI backend not available',
            message: error.message,
        });
    }
});

/**
 * POST /search — Semantic search (RAW mode, proxied to FastAPI)
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

        console.log(`[${new Date().toISOString()}] Search: "${query.substring(0, 50)}..." in ${textbook || 'default'}`);

        const result = await proxyToFastAPI('POST', '/search', {
            query: query.trim(),
            textbook: textbook || 'intro_ml',
            top_k: top_k || 5,
        });

        const duration = Date.now() - startTime;
        console.log(`[${new Date().toISOString()}] Search completed in ${duration}ms`);

        res.status(result.statusCode).json(result.data);

    } catch (error) {
        const duration = Date.now() - startTime;
        console.error(`[${new Date().toISOString()}] Search failed after ${duration}ms:`, error.message);

        res.status(500).json({
            error: 'Search Failed',
            message: error.message || 'FastAPI backend not available',
            duration: `${duration}ms`,
            hint: 'Ensure FastAPI is running: cd embeddings && python api_server.py',
        });
    }
});

/**
 * POST /search/answer — LLM-enhanced answer (proxied to FastAPI)
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

        console.log(`[${new Date().toISOString()}] LLM Answer: "${query.substring(0, 50)}..." in ${textbook || 'default'}`);

        const result = await proxyToFastAPI('POST', '/search/answer', {
            query: query.trim(),
            textbook: textbook || 'intro_ml',
        });

        const duration = Date.now() - startTime;
        console.log(`[${new Date().toISOString()}] LLM Answer completed in ${duration}ms`);

        res.status(result.statusCode).json(result.data);

    } catch (error) {
        const duration = Date.now() - startTime;
        console.error(`[${new Date().toISOString()}] LLM Answer failed:`, error.message);

        res.status(500).json({
            error: 'LLM Answer Failed',
            message: error.message || 'FastAPI backend not available',
            duration: `${duration}ms`,
            hint: 'Ensure FastAPI is running: cd embeddings && python api_server.py',
        });
    }
});

// =========================================================================
// ERROR HANDLING
// =========================================================================

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(`[${new Date().toISOString()}] Error:`, err);
    res.status(500).json({
        error: 'Internal Server Error',
        timestamp: new Date().toISOString()
    });
});

// =========================================================================
// START SERVER
// =========================================================================

const PORT = process.env.PORT || 5000;
const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 LearnLens API Gateway running on port ${PORT}`);
    console.log(`🔗 Proxying to FastAPI: ${FASTAPI_URL}`);
    console.log(`📍 Search (RAW): POST http://localhost:${PORT}/search`);
    console.log(`🤖 LLM Answer:   POST http://localhost:${PORT}/search/answer`);
    console.log(`📚 Textbooks:     GET  http://localhost:${PORT}/textbooks`);
    console.log(`❤️  Health:        GET  http://localhost:${PORT}/health`);

    // Check FastAPI connectivity on startup
    setTimeout(async () => {
        try {
            const result = await proxyToFastAPI('GET', '/health');
            if (result.data.status === 'healthy') {
                console.log(`✅ FastAPI backend is healthy (${result.data.textbooks_loaded} textbooks loaded)`);
            } else {
                console.log(`⚠️  FastAPI backend status: ${result.data.status}`);
            }
        } catch (error) {
            console.log(`❌ FastAPI backend not reachable at ${FASTAPI_URL}`);
            console.log(`   Start it with: cd embeddings && python api_server.py`);
        }
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