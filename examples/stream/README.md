# whisper.cpp/examples/stream

This is a naive example of performing real-time inference on audio from your microphone.
The `whisper-stream` tool samples the audio every half a second and runs the transcription continously.
More info is available in [issue #10](https://github.com/ggerganov/whisper.cpp/issues/10).

```bash
./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 8 --step 500 --length 5000
```


https://user-images.githubusercontent.com/1991296/194935793-76afede7-cfa8-48d8-a80f-28ba83be7d09.mp4

## Sliding window mode with VAD

Setting the `--step` argument to `0` enables the sliding window mode:

```bash
 ./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 6 --step 0 --length 30000 -vth 0.6
```

In this mode, the tool will transcribe only after some speech activity is detected. A very
basic VAD detector is used, but in theory a more sophisticated approach can be added. The
`-vth` argument determines the VAD threshold - higher values will make it detect silence more often.
It's best to tune it to the specific use case, but a value around `0.6` should be OK in general.
When silence is detected, it will transcribe the last `--length` milliseconds of audio and output
a transcription block that is suitable for parsing.

## Building

The `whisper-stream` tool depends on SDL2 library to capture audio from the microphone. You can build it like this:

```bash
# Install SDL2
# On Debian based linux distributions:
sudo apt-get install libsdl2-dev

# On Fedora Linux:
sudo dnf install SDL2 SDL2-devel

# Install SDL2 on Mac OS
brew install sdl2

cmake -B build -DWHISPER_SDL2=ON
cmake --build build --config Release

./build/bin/whisper-stream
```

## Web version

This tool can also run in the browser: [examples/stream.wasm](/examples/stream.wasm)

## JSON Output Mode

The stream tool supports JSON output format, which is useful for programmatic processing of transcriptions:

```bash
./build/bin/whisper-stream -m ./models/ggml-base.en.bin -j
```

When JSON output is enabled with the `-j/--json` flag, each transcription is output as a structured JSON object including:
- Segment information with timestamps
- Transcription text
- Speaker turns (when using a model that supports diarization)
- Optional token-level probabilities (with `-pt/--print-tokens` flag)

The JSON output is compatible with log analyzers and provides a machine-readable way to process transcriptions in real-time applications.

Example JSON output format:
```json
{
  "type": "transcription",
  "iter": 5,
  "iter_start_ms": 15243,
  "segments": [
    {
      "id": 0,
      "text": "Hello world, this is a test.",
      "start_ms": 15243,
      "end_ms": 16784,
      "speaker_turn": false,
      "confidence": 0.873
    }
  ],
  "text": "Hello world, this is a test."
}
```

## Replay Functionality

The stream tool can replay previously recorded transcriptions from a JSONL file:

```bash
./build/bin/whisper-stream -r recordings.jsonl
```

This feature is useful for:
- Testing speech recognition output without needing audio
- Demonstrating the transcription process
- Debugging issues with transcription timing

To use the replay feature:
1. Record a session with JSON output enabled: `./build/bin/whisper-stream -m ./models/ggml-base.en.bin -j -f recordings.jsonl`
2. Replay the recorded session: `./build/bin/whisper-stream -r recordings.jsonl`

During replay, the timing between transcriptions is preserved, reproducing the original transcription experience.
```