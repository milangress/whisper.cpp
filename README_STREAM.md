
# Integrating whisper.cpp with Electron Applications

This guide explains how to integrate the whisper-stream tool with an Electron application, treating whisper.cpp as a Git submodule and handling the build process through npm scripts.

## Adding whisper.cpp as a Git Submodule

1. **Add the submodule to your Electron project:**

```bash
# From your Electron project root
git submodule add -b milan/stream git@github.com:milangress/whisper.cpp.git vendor/whisper.cpp
git submodule update --init --recursive
```

2. **Create a custom build script directory:**

```bash
mkdir -p scripts/whisper-build
```

## Building the Binary

### Prerequisites

Install the required dependencies:

- **macOS:**
  ```bash
  brew install cmake sdl2
  ```

- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get install cmake libsdl2-dev
  ```

- **Windows:**
  ```bash
  # Install CMake from https://cmake.org/download/
  # Install SDL2 from https://www.libsdl.org/download-2.0.php
  # Or use vcpkg:
  vcpkg install sdl2:x64-windows
  ```

### Build Script Setup

Create a build script for each platform:

**scripts/whisper-build/build.js:**

```javascript
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Paths
const ROOT_DIR = path.resolve(__dirname, '../..');
const WHISPER_DIR = path.join(ROOT_DIR, 'vendor', 'whisper.cpp');
const LIB_DIR = path.join(ROOT_DIR, 'resources', 'lib');

// Create resources/lib directory if it doesn't exist
if (!fs.existsSync(LIB_DIR)) {
  fs.mkdirSync(LIB_DIR, { recursive: true });
}

// Platform-specific configurations
const platform = os.platform();
console.log(`Building whisper-stream for ${platform}...`);

try {
  // Update git submodules if needed
  console.log('Updating git submodules...');
  execSync('git submodule update --init --recursive', { 
    stdio: 'inherit',
    cwd: ROOT_DIR
  });

  // Change to whisper.cpp directory
  process.chdir(WHISPER_DIR);

  // Build based on platform
  if (platform === 'darwin') {  // macOS
    // macOS special handling for C++ include path if needed
    const cplusplusPath = '/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1';
    if (fs.existsSync(cplusplusPath)) {
      process.env.CPLUS_INCLUDE_PATH = cplusplusPath;
    }
    
    execSync('cmake -B build -DWHISPER_SDL2=ON -DWHISPER_COREML=1', { stdio: 'inherit' });
    execSync('cmake --build build --config Release', { stdio: 'inherit' });
    
    // Copy binary to the resources/lib directory
    fs.copyFileSync(
      path.join(WHISPER_DIR, 'build', 'bin', 'whisper-stream'),
      path.join(LIB_DIR, 'whisper-stream')
    );
    
    // Make executable
    execSync(`chmod +x "${path.join(LIB_DIR, 'whisper-stream')}"`, { stdio: 'inherit' });
  } 
  else if (platform === 'linux') {  // Linux
    execSync('cmake -B build -DWHISPER_SDL2=ON', { stdio: 'inherit' });
    execSync('cmake --build build --config Release', { stdio: 'inherit' });
    
    // Copy binary to the resources/lib directory
    fs.copyFileSync(
      path.join(WHISPER_DIR, 'build', 'bin', 'whisper-stream'),
      path.join(LIB_DIR, 'whisper-stream')
    );
    
    // Make executable
    execSync(`chmod +x "${path.join(LIB_DIR, 'whisper-stream')}"`, { stdio: 'inherit' });
  } 
  else if (platform === 'win32') {  // Windows
    execSync('cmake -B build -DWHISPER_SDL2=ON', { stdio: 'inherit' });
    execSync('cmake --build build --config Release', { stdio: 'inherit' });
    
    // Copy binary to the resources/lib directory
    fs.copyFileSync(
      path.join(WHISPER_DIR, 'build', 'bin', 'Release', 'whisper-stream.exe'),
      path.join(LIB_DIR, 'whisper-stream.exe')
    );
    
    // Copy SDL2.dll if needed
    try {
      const sdlDllPath = path.join(WHISPER_DIR, 'build', 'bin', 'Release', 'SDL2.dll');
      if (fs.existsSync(sdlDllPath)) {
        fs.copyFileSync(sdlDllPath, path.join(LIB_DIR, 'SDL2.dll'));
      }
    } catch (err) {
      console.warn('SDL2.dll not found. You may need to copy it manually.');
    }
  }

  console.log('Build completed successfully!');
  console.log(`Binary located at: ${path.join(LIB_DIR, platform === 'win32' ? 'whisper-stream.exe' : 'whisper-stream')}`);
} catch (error) {
  console.error('Build failed:', error);
  process.exit(1);
}```

### NPM Scripts Integration

Add the following to your Electron app's `package.json`:

```json
{
  "scripts": {
    "build:whisper": "node scripts/whisper-build/build.js",
    "download:models": "node scripts/whisper-build/download-models.js",
    "postinstall": "npm run build:whisper && npm run download:models"
  }
}
```

## Downloading Models

Create a script to download the whisper models:

**scripts/whisper-build/download-models.js:**

```javascript
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths
const ROOT_DIR = path.resolve(__dirname, '../..');
const WHISPER_DIR = path.join(ROOT_DIR, 'vendor', 'whisper.cpp');
const MODELS_DIR = path.join(ROOT_DIR, 'resources', 'models');

// Create models directory if it doesn't exist
if (!fs.existsSync(MODELS_DIR)) {
  fs.mkdirSync(MODELS_DIR, { recursive: true });
}

// Models to download (add more as needed)
const models = [
  'small.en-q5_1',
  // Add other models as needed
];

console.log('Downloading whisper models...');

try {
  // Update git submodules if needed
  console.log('Updating git submodules...');
  execSync('git submodule update --init --recursive', {
    stdio: 'inherit',
    cwd: ROOT_DIR,
  });

  // Change to whisper.cpp directory to access the download script
  process.chdir(WHISPER_DIR);

  for (const model of models) {
    console.log(`Downloading ${model}...`);

    // Download directly to our models directory
    // The download script will add the ggml- prefix and .bin extension
    execSync(`bash ./models/download-ggml-model.sh ${model} "${MODELS_DIR}"`, {
      stdio: 'inherit',
    });

    // Verify the file was downloaded
    const expectedPath = path.join(MODELS_DIR, `ggml-${model}.bin`);
    if (fs.existsSync(expectedPath)) {
      console.log(`Model successfully downloaded to ${expectedPath}`);
    } else {
      console.error(`Error: Model file not found at ${expectedPath} after download attempt`);
      throw new Error('Model download failed');
    }
  }

  console.log('All models downloaded successfully!');
} catch (error) {
  console.error('Model download failed:', error);
  process.exit(1);
}
```

Add to package.json:

```json
{
  "scripts": {
    "download:models": "node scripts/whisper-build/download-models.js"
  }
}
```

## Using whisper-stream in Your Electron App

Add this to your main process file:

```javascript
const { spawn } = require('child_process');
const path = require('path');

function startWhisperStream() {
  // Path to the whisper-stream binary relative to the app
  const whisperPath = path.join(__dirname, 'resources', 'lib', process.platform === 'win32' ? 'whisper-stream.exe' : 'whisper-stream');
  const modelPath = path.join(__dirname, 'resources', 'models', 'ggml-small.en-q5_1.bin');

  // Make sure the file is executable on Unix systems
  if (process.platform !== 'win32') {
    require('fs').chmodSync(whisperPath, 0o755);
  }

  // Spawn the process with JSON output enabled
  const whisperProcess = spawn(whisperPath, [
    '-m', modelPath,
    '-j', // Enable JSON output
    '--step', '500',
    '--length', '5000'
  ]);

  // Handle standard output (JSON data)
  whisperProcess.stdout.on('data', (data) => {
    try {
      // Parse each line as JSON
      const lines = data.toString().trim().split('\n');
      lines.forEach(line => {
        if (line.trim()) {
          const jsonData = JSON.parse(line);
          console.log('Received transcription:', jsonData);
          // Send to renderer or process the data
        }
      });
    } catch (e) {
      console.error('Error parsing JSON:', e);
    }
  });

  // Handle stderr
  whisperProcess.stderr.on('data', (data) => {
    console.error(`whisper-stream error: ${data}`);
  });

  // Handle process close
  whisperProcess.on('close', (code) => {
    console.log(`whisper-stream process exited with code ${code}`);
  });

  return whisperProcess;
}

// Example usage in your app
app.whenReady().then(() => {
  // Start whisper-stream when app is ready
  const whisperProcess = startWhisperStream();
  
  // Clean up on app quit
  app.on('will-quit', () => {
    whisperProcess.kill();
  });
});
```

## Electron Builder Configuration

To include the whisper-stream binaries and models in your packaged Electron app, add to `package.json`:

```json
{
  "build": {
    "extraFiles": [
      {
        "from": "resources/lib",
        "to": "resources/lib",
        "filter": ["**/*"]
      },
      {
        "from": "resources/models",
        "to": "resources/models",
        "filter": ["**/*"]
      }
    ]
  }
}
```

## Development Workflow

1. **Initial setup:**
   ```bash
   git clone --recursive YOUR_ELECTRON_APP_REPO
   cd YOUR_ELECTRON_APP
   npm install
   npm run download:models
   ```

2. **Rebuilding whisper-stream (if needed):**
   ```bash
   npm run build:whisper
   ```

3. **Running your Electron app:**
   ```bash
   npm start
   ```

## Build Commands Reference

For manual building of whisper.cpp:

(pyenv activate py311-whisper)
generate coreml model with: `./models/generate-coreml-model.sh base.en`

```bash
# macOS
cd vendor/whisper.cpp
cmake -B build -DWHISPER_SDL2=ON -DWHISPER_COREML=1
export CPLUS_INCLUDE_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
cmake --build build --config Release

# Linux
cd vendor/whisper.cpp
cmake -B build -DWHISPER_SDL2=ON
cmake --build build --config Release

# Windows
cd vendor/whisper.cpp
cmake -B build -DWHISPER_SDL2=ON
cmake --build build --config Release
```

## Troubleshooting

- **SDL2 not found**: Ensure SDL2 is installed and properly linked. On macOS/Linux, use package managers. On Windows, you may need to set SDL2_DIR environment variable.

- **Binary not executable**: On macOS/Linux, ensure the binary is executable: `chmod +x bin/whisper-stream`

- **Missing dependencies**: Run `ldd bin/whisper-stream` (Linux) or `otool -L bin/whisper-stream` (macOS) to check dynamic library dependencies.