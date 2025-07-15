# CrossroadFusion

## Overview

**CrossroadFusion** is a project that implements multi-source trajectory fusion for roadside sensing devices deployed at intersections. The system collects, associates, and fuses trajectory data from multiple roadside units (RSUs) to provide accurate and consistent object tracking in crossroad scenarios.

## Features

- Ingestion of raw trajectory data streams from multiple RSUs

- Data association and track matching across sensors

- Trajectory fusion algorithms (e.g., Kalman Filter-based fusion)

- Visualization tools for fused tracks

- Modular design for easy integration with simulation or real-world RSU data

[![Watch the video](https://img.youtube.com/vi/dRQLWckiGzk/hqdefault.jpg)](https://www.youtube.com/watch?v=dRQLWckiGzk)


## Directory Structure

`
main.py              - Core fusion algorithms
README.md         - Project overview
`

## Requirements
- Python 3.8+
- NumPy 
- SciPy 
- Sklearn-learn
- Pyproj
- Websockets



## Installation

`git clone https://github.com/philtell/CrossroadFusion.git`

`cd CrossroadFusion
`

`pip install -r requirements.txt
`

## Quick Start
Run a sample fusion pipeline with provided demo data:

`python main.py
`

## License
MIT License

## Contact
For questions or contributions, please open an issue or contact [your email].