# Project Goals – Stud Sense

## Purpose
To create a smartphone application that uses the camera to recognize LEGO pieces in real time and generate a digital 3D model of the creation as it is built. The app should allow users to start building from scratch and automatically track progress without manual input.

## Core Features
- Detect and classify individual LEGO bricks through the phone camera
- Track the order, position, and orientation of each placed piece
- Construct a real-time 3D model that updates as new bricks are added
- Generate step-by-step build instructions from the user’s actual build
- Maintain a build history (timeline or log of part placement)

## Stretch Features
- Export builds to formats like Stud.io or LDraw
- Real-time collaboration (multiple users building together)
- Instruction sharing/community showcase features
- Offline functionality and camera-based build tracking without AR dependencies

## Technical Considerations
- Object Detection: YOLOv8 or TensorFlow-based custom model
- Dataset: Synthetic images using BrickLink Studio / LDraw parts
- Model Training: Roboflow for early no-code experimentation
- 3D Modeling: Unity (AR Foundation) or WebGL (Three.js)
- Platform: iOS and Android

## Milestones (Draft)
- [ ] Capture or generate initial dataset of 5–10 bricks
- [ ] Train basic detection model using Roboflow
- [ ] Test model on real camera input
- [ ] Create simple structure to represent digital build
- [ ] Begin live tracking of part placements

## Notes
This project is a technical and educational exercise. The emphasis is on building a functional prototype, not commercial release. Branding, UI polish, and advanced AR features will come later if the core functionality proves feasible.

## Token
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
