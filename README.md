# Rubik’s Cube Solver

A cross-platform mobile application built with Flutter and Dart that captures and analyzes images of a Rubik’s Cube to compute and display an optimal solution using Kociemba’s two-phase algorithm.

---

## Features

* **Real-time Cube Capture**: Guides the user through capturing all six faces of the cube using the device camera plugin.
* **Image Processing**: Applies OpenCV-inspired techniques for color detection, edge detection, and perspective correction directly in Dart.
* **Fast Solving Algorithm**: Implements Kociemba’s two-phase algorithm to generate optimal move sequences in under a few seconds.
* **Smooth UI Performance**: Utilizes Dart isolates and futures for asynchronous image analysis without janking the UI.
* **Error Handling**: Detects and recovers from camera initialization issues (e.g., `LateInitializationError`) and invalid captures.
* **Automated Testing**: Includes unit tests for processing routines and widget tests for UI components to maintain high code quality.

---

## Installation

1. **Prerequisites**: Ensure you have Flutter installed (>= 3.0) and set up on your system.
2. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/rubiks-cube-solver.git
   cd rubiks-cube-solver
   ```
3. **Install dependencies**:

   ```bash
   flutter pub get
   ```
4. **Run on a device or emulator**:

   ```bash
   flutter run
   ```

---

## Usage

1. Launch the app on your mobile device or emulator.
2. Follow the on-screen prompts to capture each face of the cube: rotate to the next face when prompted.
3. Once all faces are captured and validated, the solver will display a sequence of moves (e.g., `R U R' U'`).
4. Follow the move list to solve your cube.

---

## Roadmap (Work in Progress)

The project is still under active development. Planned enhancements include:

* **Manual Color Correction**: Allow users to adjust detected colors if automatic detection misclassifies stickers.
* **Support for Other Cubes**: Extend solving capabilities to 2×2, 4×4, and 5×5 cubes.
* **Export & Share Solutions**: Provide options to export move sequences as text or animated GIFs and share with friends.
* **Animated Move Playback**: Integrate 3D cube animations to visually demonstrate each step.
* **UI/UX Polish**: Improve layout, add theming options, and refine capture guidance overlays.
* **Performance Optimizations**: Profile and optimize image-processing routines for lower-end devices.
* **Flutter Web/Desktop**: Port the application to web and desktop platforms.


