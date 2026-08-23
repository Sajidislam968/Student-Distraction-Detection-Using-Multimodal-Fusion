# Student Distraction Detection Using Multimodal Fusion

## Abstract

Maintaining concentration during study sessions is difficult when distractions arise from multiple sources, including unsuitable applications, background audio, keyboard and mouse activity, and changes in facial or head orientation. This project presents a real-time student focus monitoring system that combines computer vision, audio analysis, application activity, and input-device signals. The system uses face and eye tracking, head-pose estimation, object detection, speech and noise analysis, active-window classification, and a fusion engine to estimate the learner's focus state. A dashboard and session logger provide live feedback and summarize focus-related events over time. The modular design allows each sensing component to operate independently while enabling multimodal fusion for a more robust distraction estimate. The repository contains the source code, training and evaluation scripts, and test modules. Datasets, generated logs, personal activity records, model weights, and virtual environments are intentionally excluded from version control and can be added locally when needed.

> This abstract is a working draft and can be updated later.

## Features

- Face, eye, and head-pose monitoring
- Focus-object detection with YOLO-based scripts
- Active application classification
- Audio, speech, and noise monitoring
- Keyboard and mouse activity monitoring
- Multimodal focus prediction through a fusion engine
- Live dashboard and study-session logging
- Personalization and session-level analysis modules

## Project Structure

| Path | Description |
| --- | --- |
| `integrated_monitor.py` | Main multimodal monitoring entry point |
| `run_focusmonitor.py` | Focus monitor runner |
| `dashboard/` | Dashboard interface and supporting scripts |
| `audio/` | Audio, speech, and noise analysis modules |
| `app_classifier/` | Active-window and application classification |
| `fusion_engine/` | Multimodal prediction and training scripts |
| `personalization/` | User-specific activity components |
| `*_monitor.py` | Individual input and computer-vision monitors |
| `test_*.py` | Focused test and evaluation scripts |

## Setup

Use Python 3.12 or a compatible Python 3 environment, then install the dependencies required by the modules you plan to run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some modules may require system-level access to a camera, microphone, active-window information, or input-device events. Local model weights and datasets are not included in this repository.

## Running

```bash
python integrated_monitor.py
```

Individual modules and tests can be run directly with Python when their local data and model requirements are available.

## Data and Privacy

Datasets, CSV files, logs, runtime status files, personal activity records, generated reports, model weights, and virtual environments are excluded through `.gitignore`. Keep any locally collected data outside public version control and review the ignore rules before adding new artifacts.

## License

No license has been selected yet.
