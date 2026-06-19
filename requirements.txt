# 📋 System Requirements Specification (SRS)

This document outlines the functional benchmarks, operational constraints, hardware dependencies, and performance metrics required for the Autonomous UAV System for Real-Time Human Distress Detection & Health Vitals Monitoring.

---

## 1. Functional Requirements (FR)
The core software architecture must execute the following distinct operations:
* **FR-1 (Autonomous Flight):** The system must accept autonomous waypoint commands and handle flight telemetry streams via the MAVLink protocol.
* **FR-2 (Computer Vision Pose Tracking):** The vision pipeline must run an edge-optimized inference model to detect human skeletal landmarks and classify distress states (e.g., frantic waving, falling, immobility).
* **FR-3 (Contactless rPPG Processing):** The system must dynamically isolate facial Regions of Interest (ROI) and extract a continuous Blood Volume Pulse (BVP) signal to compute Heart Rate (BPM) and Respiratory Rate (RPM) without physical sensors.
* **FR-4 (Telemetry Broadcaster):** The system must serve as a local network server to broadcast real-time multi-variable telemetry and biometric arrays via WebSockets.
* **FR-5 (Automated Incident Alerting):** Upon identifying a high-confidence threat state, the system must autonomously generate an encrypted payload containing a captured image, calculated health vitals, and precise GPS coordinates, routing it instantly via an IoT gateway.

---

## 2. Non-Functional Requirements (NFR)
To perform reliably in time-critical emergency scenarios, the system must adhere to the following performance bounds:
* **NFR-1 (Latency):** The end-to-end telemetry and video matrix broadcast delay to the mobile dashboard must remain below 200 milliseconds over a local network.
* **NFR-2 (Compute Thresholds):** Real-time computer vision inference and rPPG digital signal processing loops must run concurrently on edge hardware without dropping the camera capture rate below 15 Frames Per Second (FPS).
* **NFR-3 (Network Independence):** The core AI detection, digital signal processing, and telemetry logic must execute completely at the edge on the local hardware layer, without relying on external cloud compute or internet dependencies.

---

## 3. Hardware Requirements
* **Companion Computer:** Raspberry Pi 5 (8GB RAM recommended for matrix processing overhead).
* **Flight Controller:** Pixhawk 2.4.8 (or Pixhawk-based platform running ArduPilot firmware).
* **Sensor Payload:** High-definition global-shutter camera module connected via CSI/USB interface to mitigate rolling-shutter artifacts during flight vibrations.
* **Communication Link:** High-bandwidth Wi-Fi telemetry module or RF transceiver for edge network routing.

---

## 4. Software Environment
* **Operating System:** Raspberry Pi OS (64-bit, Bookworm or newer) optimized for hardware-accelerated computer vision layers.
* **Core Libraries:** Python 3.10+, OpenCV (Open Source Computer Vision Library), MediaPipe, PyMAVLink, and `asyncio`/`websockets` networking stacks.
