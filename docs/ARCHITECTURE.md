# System architecture

## Processing stages

| Stage | Input | Output | Implementation |
| --- | --- | --- | --- |
| Detection | Video frame | Person bounding boxes | Ultralytics YOLO checkpoint |
| Tracking | Bounding boxes | Persistent track identifiers | SORT with a Kalman filter |
| Identification | Cropped person images | Target identity or unknown | Haar face detection and face embeddings |
| Evidence capture | Frames for a matched track | Saved video and coordinates | OpenCV video writer |
| Expression analysis | Target evidence clip | Percentage of observed expression labels | DeepFace |
| Connection analysis | Evidence clip with target region masked | Other detected faces | YOLO, SORT, and face embeddings |
| Persistence | Identities, clips, and summaries | Investigation records | MySQL |

## Current boundaries

The current implementation is a single-machine prototype. Detection, recording, analysis, and database access are tightly coupled and several settings still reflect the original experiment. A production redesign should separate capture, inference, storage, and review into independently tested components.

## Recommended next refactor

1. Move database and path settings into one configuration module.
2. Add a command-line interface for camera, checkpoint, and output selection.
3. Replace the shared `data/coordinates.txt` file with an in-memory record or structured JSON file per clip.
4. Add database migrations and repository interfaces.
5. Add unit tests for tracking, coordinate parsing, and evidence generation.
6. Add an integration test using a short, consented sample video.
