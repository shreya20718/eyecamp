const express = require('express');
const path = require('path');
const { spawn } = require('child_process');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Enhanced CORS configuration for mobile deployment
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

// Middleware with increased limits for image data
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// MongoDB Atlas Connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb+srv://shreyagaikwad107_db_user:bKPS4n1iO6BTJYoN@cluster0.aortiot.mongodb.net/optifocus?retryWrites=true&w=majority';

mongoose.connect(MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  serverSelectionTimeoutMS: 10000,
  socketTimeoutMS: 45000,
})
.then(() => console.log('✓ MongoDB Atlas connected successfully'))
.catch(err => console.error('✗ MongoDB connection error:', err));

// MongoDB Models
const pupilMeasurementSchema = new mongoose.Schema({
  pd: { type: Number, required: true },
  leftNose: { type: Number, required: true },
  rightNose: { type: Number, required: true },
  tilt: { type: Number, required: true },
  distance: { type: Number, default: null },
  accuracy: { type: String, enum: ['High', 'Medium'], required: true },
  filename: { type: String, required: true },
  image: { type: String, required: true },
  timestamp: { type: Date, required: true }
}, { timestamps: true });

const frameMeasurementSchema = new mongoose.Schema({
  leftWidth: { type: Number, required: true },
  leftHeight: { type: Number, required: true },
  rightWidth: { type: Number, required: true },
  rightHeight: { type: Number, required: true },
  bridge: { type: Number, required: true },
  filename: { type: String, required: true },
  image: { type: String, required: true },
  timestamp: { type: Date, required: true }
}, { timestamps: true });

const PupilMeasurement = mongoose.model('PupilMeasurement', pupilMeasurementSchema);
const FrameMeasurement = mongoose.model('FrameMeasurement', frameMeasurementSchema);

// Python process and SSE setup
let pyProc = null;
const sseClients = new Set();

// ==================== MEASUREMENT ROUTES ====================

// Save pupil measurement with enhanced error handling
app.post('/api/measurements/pupil', async (req, res) => {
  try {
    // Check MongoDB connection
    if (mongoose.connection.readyState !== 1) {
      return res.status(503).json({
        success: false,
        error: 'Database connection unavailable'
      });
    }

    const { pd, leftNose, rightNose, tilt, distance, accuracy, filename, image, timestamp } = req.body;
    
    // Validate required fields
    if (!pd || !leftNose || !rightNose || !tilt || !accuracy || !filename || !image) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields'
      });
    }
    
    const measurement = new PupilMeasurement({
      pd,
      leftNose,
      rightNose,
      tilt,
      distance,
      accuracy,
      filename,
      image,
      timestamp: new Date(timestamp)
    });
    
    await measurement.save();
    
    console.log('\n=== PUPIL MEASUREMENT SAVED ===');
    console.log(JSON.stringify({
      id: measurement._id,
      pd,
      leftNose,
      rightNose,
      tilt,
      distance,
      accuracy,
      filename,
      timestamp
    }, null, 2));
    console.log('===============================\n');
    
    res.status(201).json({
      success: true,
      message: 'Pupil measurement saved successfully',
      id: measurement._id
    });
  } catch (error) {
    console.error('Error saving pupil measurement:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to save measurement'
    });
  }
});

// Save frame measurement with enhanced error handling
app.post('/api/measurements/frame', async (req, res) => {
  try {
    // Check MongoDB connection
    if (mongoose.connection.readyState !== 1) {
      return res.status(503).json({
        success: false,
        error: 'Database connection unavailable'
      });
    }

    const { leftWidth, leftHeight, rightWidth, rightHeight, bridge, filename, image, timestamp } = req.body;
    
    // Validate required fields
    if (!leftWidth || !leftHeight || !rightWidth || !rightHeight || !bridge || !filename || !image) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields'
      });
    }
    
    const measurement = new FrameMeasurement({
      leftWidth,
      leftHeight,
      rightWidth,
      rightHeight,
      bridge,
      filename,
      image,
      timestamp: new Date(timestamp)
    });
    
    await measurement.save();
    
    console.log('\n=== FRAME MEASUREMENT SAVED ===');
    console.log(JSON.stringify({
      id: measurement._id,
      leftWidth,
      leftHeight,
      rightWidth,
      rightHeight,
      bridge,
      filename,
      timestamp
    }, null, 2));
    console.log('===============================\n');
    
    res.status(201).json({
      success: true,
      message: 'Frame measurement saved successfully',
      id: measurement._id
    });
  } catch (error) {
    console.error('Error saving frame measurement:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to save measurement'
    });
  }
});

// Get all pupil measurements
app.get('/api/measurements/pupil', async (req, res) => {
  try {
    const measurements = await PupilMeasurement.find()
      .sort({ timestamp: -1 })
      .select('-image')
      .limit(100);
    
    res.json({ 
      success: true, 
      count: measurements.length,
      data: measurements 
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get all frame measurements
app.get('/api/measurements/frame', async (req, res) => {
  try {
    const measurements = await FrameMeasurement.find()
      .sort({ timestamp: -1 })
      .select('-image')
      .limit(100);
    
    res.json({ 
      success: true, 
      count: measurements.length,
      data: measurements 
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get single pupil measurement by ID (with image)
app.get('/api/measurements/pupil/:id', async (req, res) => {
  try {
    const measurement = await PupilMeasurement.findById(req.params.id);
    if (!measurement) {
      return res.status(404).json({ success: false, error: 'Measurement not found' });
    }
    res.json({ success: true, data: measurement });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get single frame measurement by ID (with image)
app.get('/api/measurements/frame/:id', async (req, res) => {
  try {
    const measurement = await FrameMeasurement.findById(req.params.id);
    if (!measurement) {
      return res.status(404).json({ success: false, error: 'Measurement not found' });
    }
    res.json({ success: true, data: measurement });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// ==================== PYTHON PROCESS ROUTES ====================

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
      try { 
        res.write('event: end\n'); 
        res.write('data: {"ended":true}\n\n'); 
      } catch {}
    }
    pyProc = null;
  });
  
  return res.json({ message: 'started' });
});

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
  res.json({ 
    ok: true, 
    running: Boolean(pyProc),
    mongodb: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
    dbState: mongoose.connection.readyState
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({
    success: false,
    error: err.message || 'Internal server error'
  });
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
  console.log(`MongoDB: ${mongoose.connection.readyState === 1 ? 'Connected' : 'Connecting...'}`);
  console.log(`CORS enabled for all origins`);
});