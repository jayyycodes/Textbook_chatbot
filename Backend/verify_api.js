const http = require('http');

const data = JSON.stringify({
    query: "What is machine learning?",
    textbook: "intro_to_ml"
});

const options = {
    hostname: 'localhost',
    port: 5000,
    path: '/search',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length
    }
};

const req = http.request(options, (res) => {
    let body = '';
    res.on('data', (chunk) => body += chunk);
    res.on('end', () => {
        console.log('Status Code:', res.statusCode);
        try {
            const json = JSON.parse(body);
            console.log('Results count:', json.results ? json.results.length : 'No results field');
            if (json.results && json.results.length > 0) {
                console.log('First result chunk_id:', json.results[0].chunk_id);
                console.log('First result content length:', json.results[0].content.length);
            } else {
                console.log('Full response keys:', Object.keys(json));
                if (json.raw_answer) {
                    console.log('RAW ANSWER PREVIEW:');
                    console.log(JSON.stringify(json.raw_answer.substring(0, 200))); // Stringify to see escape chars
                }
                if (json.error) console.log('Error:', json.error);
                if (json.message) console.log('Message:', json.message);
            }
        } catch (e) {
            console.log('Failed to parse JSON:', e.message);
            console.log('Raw body preview:', body.substring(0, 200));
        }
    });
});

req.on('error', (error) => {
    console.error('Error:', error);
});

req.write(data);
req.end();
