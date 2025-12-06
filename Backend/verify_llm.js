const http = require('http');

const data = JSON.stringify({
    query: "What is machine learning?",
    textbook: "intro_to_ml"
});

const options = {
    hostname: 'localhost',
    port: 5000,
    path: '/search/answer',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length
    }
};

console.log('Sending request to /search/answer...');
const req = http.request(options, (res) => {
    let body = '';
    res.on('data', (chunk) => body += chunk);
    res.on('end', () => {
        console.log('Status Code:', res.statusCode);
        try {
            const json = JSON.parse(body);
            if (json.answer) {
                console.log('✅ Answer received!');
                console.log('Answer preview:', json.answer.substring(0, 100) + '...');
                console.log('Metadata:', JSON.stringify(json.metadata, null, 2));
                console.log('Chunks processed:', json.chunks_processed);
            } else {
                console.log('❌ No answer in response');
                console.log('Full response:', json);
            }
        } catch (e) {
            console.log('Failed to parse JSON:', e.message);
            console.log('Raw body:', body);
        }
    });
});

req.on('error', (error) => {
    console.error('Error:', error.message);
});

req.write(data);
req.end();
