 const express = require('express');
 const path = require('path');
 const { spawn } = require('child_process');

 const app = express();
 const PORT = process.env.PORT || 3000;

 app.use(express.json());
 app.use(express.static(path.join(__dirname, 'public')));

 let pyProc = null;
 const sseClients = new Set();

 app.post('/start', (req, res) => {
   if (pyProc) {
     return res.status(400).json({ message: 'Process already running' });
   }

   const pythonCmd = process.env.PYTHON || 'python';
   pyProc = spawn(pythonCmd, ['detect.py'], { cwd: __dirname });

   pyProc.on('spawn', () => {
     console.log('detect.py started');
   });
   pyProc.stdout.on('data', (data) => {
     const text = data.toString();
     // Broadcast raw lines to SSE consumers
     for (const res of sseClients) {
       res.write(`data: ${text}\n\n`);
     }
     console.log('[detect.py]', text.trim());
   });
   pyProc.stderr.on('data', (data) => {
     const text = data.toString();
     console.error('[detect.py][err]', text.trim());
   });
   pyProc.on('exit', (code, signal) => {
     console.log('detect.py exited', { code, signal });
     for (const res of sseClients) {
       try { res.write('event: end\n'); res.write('data: {"ended":true}\n\n'); } catch {}
     }
     pyProc = null;
   });

   return res.json({ message: 'started' });
 });

 // Server-Sent Events endpoint to stream stdout lines from detect.py
 app.get('/events', (req, res) => {
   res.writeHead(200, {
     'Content-Type': 'text/event-stream',
     'Cache-Control': 'no-cache',
     Connection: 'keep-alive'
   });
   res.write('\n');
   sseClients.add(res);
   req.on('close', () => {
     sseClients.delete(res);
   });
 });

 app.post('/stop', (req, res) => {
  if (!pyProc) {
    return res.json({ message: 'not running' });
  }
  const p = pyProc;
  pyProc = null;
  try {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', String(p.pid), '/f', '/t']);
    } else {
      p.kill('SIGTERM');
    }
  } catch (e) {
    console.error('Failed to stop detect.py', e);
  }
  return res.json({ message: 'stopping' });
});

app.get('/health', (req, res) => {
  res.json({ ok: true, running: Boolean(pyProc) });
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});


