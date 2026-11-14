# Visual Localization MVP

## Engineering review

Computer vision for frame localization on a pre-indexed route. Indication of the reference and runtime localization are divided; retrieval restricts search, matching and geometry verify coincidence, confidence and temporary processing allow to reject doubtful results.

### Code navigation

- [app/indexing_service.py](app/indexing_service.py)
- [app/runtime.py](app/runtime.py)
- [app/vloc/localizer.py](app/vloc/localizer.py)
- [app/vloc/matching.py](app/vloc/matching.py)
- [app/vloc/confidence.py](app/vloc/confidence.py)

### Reproduction and evaluation

For honest verification, divide the query and reference records. Measure the p50/p95 geographical error by independent ground truth, false localization frequency, rejection and delay. The distance between routes does not replace the error of the predicted position. The project did not identify the service providers &apos; built-in keys during the ongoing baseline verification.


## General description

`VLOC MVP' is a visual localization system for aircraft.

The project was implemented as a MVP version of the system that defines the position of the aircraft
concerning the pre-prepared reference route.

The system receives a video stream or a video file, compares current images with
Core personnel reference base, computes geographical coordinates and
transmits the result to the web interface where it is displayed simultaneously:

- A map with current status;
- Video stream of runtime;
- Localization status;
- Service telemetry.

The project consists of a backend part on Python/FastAPI and a Frontend part on
TML/CSS/JavaScript with map interface on Leaflet.

## Technology

### Backend

- `Python 3.11+`
- `FastAPI` – HTTP API, SSE and WebSocket endpoints
- `Uvicorn` – ASGI server
- `SQLite' is a local storage of keyframes, embeddings and features
- `NumPy` — Working with embedding and vector data
- `OpenCV' is the decoded image, human resources and geometry
- `PyTorch + CUDA` — GPU Calculations
- `LightGlue` — Comparison of Local Signs
- `FFmpeg / FFprobe' is a video decoded and stream generated

### Frontend

- `HTML5`
- `CSS3`
- `Vanilla JavaScript`
- `Leaflet` interactive map
- `MJPEG preview' is a display of the current video stream in the browser
- `SSE' is the real-time localization results

## Decision architecture

The system is divided into two main contours:

### 1. Standard indexing connections

Used for the preparation of the reference route, which is then carried out
Runtime-localization.

Steps:

1. The reference video is being downloaded.
2. The `points.txt` coordinates file shall be downloaded.
3. The keyframe with the specified frequency.
4. For each keyframe-a, the following are listed:
   - Global embedding;
   - Local features;
   - Geographical reference.
5. The results are stored in the local SQLite database.
6. Retrieval index is collected from embedding.

### 2. Runtime-localization conturbation

Used to compute current position via video input.

Steps:

1. Runtime flow comes from:
   - Video files;
   - RTSP;
   - WebSocket bridge.
2. FFmpeg decodes the flow into the sequence of the frame.
3. Backend selects a fresh frame for localization.
4. Embedding is calculated for the frame.
5. Retrieval of the nearest reference keyframes is being implemented.
6. A local comparison of the topics is being carried out.
7. Compute confidence localization.
8. Applicable:
   - temporal lock;
   - speed gate;
   - smoothing.
9. The final coordinates are published in the SSE stream.
10. Frontend receives the coordinates and displays the position on the map.

## Scheme of work

### Indexing scenario

`Keyframe -> embedding + local features -> SQLite -> retrieval index`

### Runtime script

`Incoming flow -> FFmpeg -> frame -> retrieval -> matching -> conference -> coordinates -> SSE -> map`

## Project structure

- `app/` – Backend logic of the project
- `web/` - Frentend interface
- `inputs/videos/` - Reference and ringtime video
- `inputs/points/` - Reference route coordinates files
- `Data/' is a database, index and prepared keyframe frame
- `INSTALL_GPU_ONLY_WINDOWS.txt' is a separate instruction for GPU installation
- `requirements.txt` – Python Dependence of the Project

## Main backend moduli

### `app/main.py`

Entry point of the application. Creates FastAPI application, connects Rows, initialises
Runtime, localizer and retrieval-state.

### `app/api.py`

HTTP API Project. Contains:

- health/version endpoints;
- runtime endpoints;
- Endpoints indexation;
- SSE-flow of localization;
- WebSocket ingest endpoint.

### `app/runtime.py`

Runtime-localization:

- FFmpeg;
- Staff buffers;
- Localization challenge;
- Publication of the result.

### `app/indexing_service.py`

Background indexation of the reference route:

- Accepting the task parameters;
- Preparation of a reference video;
- Cleaning of the old index;
- Indexer launch;
- Retrieval index.

### `app/vloc/localizer.py`

Key localization module. Contains:

- retrieval;
- Local features;
- Geometric verification;
- temporal lock;
- smoothing the coordinates.

## Web interface

The interface contains:

- Standard indexing unit;
- Runtime launch unit;
- Video stream panel;
- Map;
- Telemetry of localization;
- Status of processing and number of personnel processed.

Frontend obtains results through:

- `GET /api/v1/localization/stream`
- `GET /api/v1/runtime/preview`

## Input format

### Reference video

By default:

- `inputs/videos/ref_01.mp4`

### Runtime video

By default:

- `inputs/videos/ref_02.mp4`

### Coordinate File

By default:

- `inputs/points/ref_01.txt`

Line format:

```text
MM:SS - LAT, LNG
```

Example:

```text
00:00 - 48.402872, 37.342720
00:05 - 48.402910, 37.342940
00:10 - 48.402950, 37.343180
```

## Project data

In operating mode, the project uses external data:

- `inputs/videos/` is a video file for indexing and runtime.
- `inputs/points/` - Reference route coordinates files
- `Data/' is the base of SQLite, retrieval index and the production of keyframe frames

When launched through Docker, these catalogs are not included in the image and are connected
It's like a volume outside.

## API

### Service endpoints

- `GET /api/v1/health` is the server state and assembly version
- `GET /api/v1/version` = assembly version

### Runtime

- `GET /api/v1/runtime/status` state
- `GET /api/v1/runtime/sources' are accessible source
- `POST /api/v1/runtime/start' is a run-off of Runtime-localization
- `POST /api/v1/runtime/stop` — Stopping Runtime
- `GET /api/v1/runtime/preview` — MJPEG preview
- `GET /api/v1/runtime/frame.mjpg' is an alternative preview endpoint
- `GET /api/v1/runtime/preview.mjpg' is an alternative preview endpoint
- `WS /api/v1/runtime/ws/{token}` — WebSocket ingest

### Localization

- `GET /api/v1/localization/stream' is a co-ordinate flow across SSE
- `POST /api/v1/localize/frame` - One frame localization

### Indexation

- `GET /api/v1/index/status` — State of Indexation
- `GET /api/v1/index/sources` source of reference video
- `GET /api/v1/index/routes` — List of Routes and Points
- `POST /api/v1/index/upload-and-build' is the download and launch of UI indexation

## Docker

### What's in the Docker view?

The Docker forms include:

- Backend source code;
- Frontend static files;
- Python dependence;
- FFmpeg and system libraries required to launch the application.

### What doesn't fit in the Docker view.

The Docker species deliberately does not include:

- `data/`
- `inputs/`

This is done to:

- Do not inflate the image with working data;
- Keep the index separate from the code;
- Allow your files to be displayed via frantend without re-assembly.

### Containerization files

- `Dockerfile` - Application image description
- `.dockerignore` — Exemptions from Assembly Context
- `docker-compose.yml` Local container launch

### Docker image collection

```powershell
docker compose build
```

### Start container

```powershell
docker compose up --build
```

Once launched, the container prints a short step instruction to open an interface in a browser.

### Stopping container

```powershell
docker compose down
```

### External catalogues connected to the container

`docker-compose.yml` is connected to:

- named volume `vloc-data -> /app/data`
- `./inputs -> /app/inputs`
- Windows user home directory `${USERPROFILE} -> /host-home` in `read-only mode`

Thus:

- The code is inside the image;
- The index is stored in the Docker volume;
- standard project logic via `inputs/videos` folders, `inputs/points`, `inputs/live` retained;
- user videos and `points.txt` read directly from host without manual copying to project.

### Start order

1. Run `docker compose up --build`.
2. Open a link to the frantend from the container's lair.
3. In the indexing unit, state:
4. Either the path from standard project folders e.g. `/app/inputs/videos/ref_01.mp4` and `/app/inputs/points/ref_01.txt`;
5. or arbitrary routes, e.g. `/host-home/Downloads/ref_01.mp4` and `/host-home/Downloads/ref_01.txt`.
6. Start indexing.
7. In the runtime block, state:
8. Or a runtime video from `/app/inputs/videos`, `/app/inputs/live` or `/host-home/...`;
9. Either `rtsp://...`;
10. or `ws://...` for WebSocket source.
11. Press launch and run through Frontend.

### Docker's changed environment

Basic container parameters are set via environment:

- `VLOC_DEVICE`
- `VLOC_SHARED_HINT`
- `VLOC_DB_PATH`
- `VLOC_FAISS_INDEX_PATH`
- `VLOC_FRAMES_DIR`
- `VLOC_VIDEOS_DIR`
- `VLOC_POINTS_DIR`
- `VLOC_LIVE_DIR`

By default, the container is set to:

- `VLOC_DEVICE=cuda`

If necessary, the value can be re-defined before starting:

```powershell
$env:VLOC_DEVICE="cuda"
docker compose up --build
```

## Baseline use scenario

1. Run `docker compose up --build`.
2. Open the web interface by reference from the container's lair.
3. Set the path to the reference video and the path to `points.txt`.
4. Do indexation.
5. Specify ringtime source.
6. Run runtime-localization.
7. Control:
   - Video;
   - Map;
   - confidence;
   - Runtime status.

## Diagnostics

### server check

Check local endpoint `GET /api/v1/health`.

### Runtime Check

Check Local Endpoint `GET /api/v1/runtime/status`.

### If localization does not appear

Check:

1. The server is running;
2. The reference index has been collected;
3. CUDA is available;
4. Runtime video is available;
5. `runtime/status` does not contain `last_error`.

## Composition of delivery

It is recommended that:

- Project reference code;
- `requirements.txt`;
- `INSTALL_GPU_ONLY_WINDOWS.txt`;
- `Dockerfile`;
- `.dockerignore`;
- `docker-compose.yml`;
- Current `README.md`.

`data/` and `inputs/` catalogues are not included and transferred to the Docker configuration
or are connected separately if work with an index already prepared is required
and inputs.

## Private configuration

Provider keys, tokens, passwords, proxy and personal parameters are set through variable environments. `.env.example` contains names without values. `.env` filled and local settings deleted from Git. Do not transmit secrets by command arguments and do not publish real magazines, downloads, certificates or user documents.

Cloud keys are not required for the offline scenario of the project. The data path, weights, and network settings are checked against the launch manual below.

Checking the source and configuration formats does not replace the launch of the infrastructure and model evaluation. Privacy changes do not confirm the quality of AI, the performance of external APIs, or the willingness of the entire system to operate.
