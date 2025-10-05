# PD Detector Web App

A web-based Pupillary Distance (PD) measurement tool using MediaPipe Face Mesh for real-time eye detection and measurement.

## Features

- **Real-time PD measurement** with camera access
- **Manual pupil adjustment** by clicking and dragging
- **Capture and save** images with measurements
- **Mobile responsive** design
- **Touch support** for mobile devices
- **Keyboard shortcuts** for quick access

## Live Demo

[Deployed on Render](https://your-app-name.onrender.com)

## Local Development

### Prerequisites

- Node.js (v14 or higher)
- Modern web browser with camera support
- HTTPS (required for camera access on mobile)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd webapp
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

4. Open your browser and navigate to `http://localhost:3000`

### Usage

1. Click **Start** to begin camera detection
2. Allow camera permission when prompted
3. Position your face within the oval guide
4. Use **Capture** to save measurements
5. **Reset Auto** to clear manual adjustments
6. **Reset Initial** to restore original detection

### Keyboard Shortcuts

- **C** - Capture current frame
- **R** - Reset to latest auto detection
- **I** - Reset to initial auto detection
- **Esc** - Stop camera

## Deployment

### Deploy to Render

1. Push your code to GitHub
2. Connect your GitHub repository to Render
3. Render will automatically detect the Node.js app
4. The app will be available at `https://your-app-name.onrender.com`

### Environment Variables

- `NODE_ENV=production`
- `PORT=10000` (Render sets this automatically)

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari (iOS 11+)
- Edge

## Mobile Support

- Android Chrome/Firefox
- iOS Safari
- Requires HTTPS for camera access

## Technical Details

- **Frontend**: HTML5, JavaScript, MediaPipe Face Mesh
- **Backend**: Node.js, Express
- **Camera**: getUserMedia API
- **Detection**: MediaPipe Face Mesh landmarks
- **Measurements**: Real-time PD, head tilt, distance estimation

## License

MIT License
