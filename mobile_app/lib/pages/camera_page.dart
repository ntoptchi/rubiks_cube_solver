import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import '../services/api.dart'; // uploadOneFace, solveFromGrids
import 'package:mobile_app/pages/solve_coach.dart';

class CameraPage extends StatefulWidget {
  const CameraPage({Key? key}) : super(key: key);

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  List<CameraDescription>? _cameras;
  CameraController? _controller;

  bool _isCameraReady = false;
  bool _isUploading = false;

  // Torch toggle
  bool _torchOn = false;

  // Per-face flow (must match backend)
  final List<String> _faceOrder = ['U', 'R', 'F', 'D', 'L', 'B'];
  int _currentFace = 0;
  final Map<String, List<List<String>>> _grids = {};

  // Colors for the 3×3 preview
  final Map<String, Color> _colorMap = const {
    'W': Colors.white,
    'Y': Colors.yellow,
    'R': Colors.red,
    'O': Colors.orange,
    'G': Colors.green,
    'B': Colors.blue,
  };

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      _cameras = await availableCameras();
      _controller = CameraController(
        _cameras!.first,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await _controller!.initialize();
      try {
        await _controller!.setFlashMode(FlashMode.off); // default off
      } catch (_) {}
      if (mounted) setState(() => _isCameraReady = true);
    } catch (e) {
      if (!mounted) return;
      _showError('Camera init failed: $e');
    }
  }

  Future<void> _setTorch(bool on) async {
    if (_controller == null) return;
    try {
      await _controller!.setFlashMode(on ? FlashMode.torch : FlashMode.off);
      if (mounted) setState(() => _torchOn = on);
    } catch (e) {
      debugPrint('Torch not available: $e');
    }
  }

  void _showError(String msg) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Oops'),
        content: Text(msg),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK')),
        ],
      ),
    );
  }

  Future<bool> _askRetry(String message) async {
    if (!mounted) return false;
    return await showDialog<bool>(
          context: context,
          barrierDismissible: false,
          builder: (_) => AlertDialog(
            title: const Text('Scan issue'),
            content: Text(message),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context, false), // Retake
                child: const Text('Retake'),
              ),
              ElevatedButton(
                onPressed: () => Navigator.pop(context, true), // Retry
                child: const Text('Retry'),
              ),
            ],
          ),
        ) ??
        false;
  }

  Future<bool> _confirmUseFace(String label, List<List<String>> grid) async {
    return await showDialog<bool>(
          context: context,
          barrierDismissible: false,
          builder: (_) {
            return AlertDialog(
              title: Text('Use this $label face?'),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  SizedBox(
                    width: 180,
                    height: 180,
                    child: GridView.builder(
                      itemCount: 9,
                      physics: const NeverScrollableScrollPhysics(),
                      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 3,
                        mainAxisSpacing: 4,
                        crossAxisSpacing: 4,
                      ),
                      itemBuilder: (ctx, i) {
                        final r = i ~/ 3;
                        final c = i % 3;
                        final ch = grid[r][c];
                        final col = _colorMap[ch] ?? Colors.grey;
                        return Container(
                          decoration: BoxDecoration(
                            color: col,
                            border: Border.all(color: Colors.black54, width: 2),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          alignment: Alignment.center,
                          child: Text(
                            ch,
                            style: TextStyle(
                              color: (ch == 'Y' || ch == 'W') ? Colors.black : Colors.white,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text('If the colors look wrong, tap Retake.'),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context, false),
                  child: const Text('Retake'),
                ),
                ElevatedButton(
                  onPressed: () => Navigator.pop(context, true),
                  child: const Text('Looks good'),
                ),
              ],
            );
          },
        ) ??
        false;
  }

  Future<void> _capture() async {
    if (!_isCameraReady || _controller == null || _isUploading) return;

    try {
      setState(() => _isUploading = true);

      final shot = await _controller!.takePicture();
      final capturedPath = shot.path;
      final String currentLabel = _faceOrder[_currentFace];

      // Upload ONE face → get 3×3 grid
      final res = await uploadOneFace(capturedPath);
      final grid = (res['grid'] as List)
          .map((row) => List<String>.from(row as List))
          .toList();

      // log the grid (pretty)
      for (final row in grid) {
        // ignore: avoid_print
        print(row.join(' '));
      }

      // Stop spinner before asking
      if (mounted) setState(() => _isUploading = false);

      // Confirm / Retake
      final ok = await _confirmUseFace(currentLabel, grid);
      if (!ok) {
        // User wants to retake this same face; do nothing else.
        return;
      }

      // Commit this face
      _grids[currentLabel] = grid;
      setState(() => _currentFace++);

      // If all 6 faces captured → solve
      if (_currentFace == _faceOrder.length) {
        await _solveAndNavigate();
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _isUploading = false);

      // Ask to retry upload or retake
      final retry = await _askRetry('Scan failed: $e');
      if (retry) {
        // try again
        await _capture();
      } else {
        // stay on same face; user will take another photo
      }
    }
  }

  Future<void> _solveAndNavigate() async {
    try {
      setState(() => _isUploading = true);

      // Attempt solve (with one optional retry via dialog)
      while (true) {
        try {
          final solved = await solveFromGrids(_grids);
          final moves = List<String>.from(solved['solution'] ?? const <String>[]);

          if (!mounted) return;
          setState(() => _isUploading = false);

          // Go to SolveCoachPage
          await Navigator.of(context).push(
            MaterialPageRoute(
              builder: (_) => SolveCoachPage(
                faceGrids: _grids,
                moves: moves,
              ),
            ),
          );

          // (Optional) reset when coming back
          // setState(() { _grids.clear(); _currentFace = 0; });
          break;
        } catch (e) {
          if (!mounted) return;
          setState(() => _isUploading = false);
          final retry = await _askRetry('Solve failed: $e');
          if (retry) {
            setState(() => _isUploading = true);
            continue; // retry loop
          } else {
            // Let user retake the last face, if you want:
            setState(() {
              _currentFace = _faceOrder.length - 1;
              _grids.remove(_faceOrder.last);
            });
            break;
          }
        }
      }
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    // simple spinners while busy/uninitialized
    if (!_isCameraReady || _controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    if (_isUploading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    // pretty label
    final pretty = {'U': 'Up', 'R': 'Right', 'F': 'Front', 'D': 'Down', 'L': 'Left', 'B': 'Back'};
    final faceKey = _currentFace < _faceOrder.length ? _faceOrder[_currentFace] : _faceOrder.last;
    final faceLabel = pretty[faceKey] ?? faceKey;

    return Scaffold(
      backgroundColor: Colors.black, // avoids any white bars
      appBar: AppBar(
        title: Text('Capture $faceLabel Face'),
        actions: [
          IconButton(
            tooltip: _torchOn ? 'Turn torch off' : 'Turn torch on',
            icon: Icon(_torchOn ? Icons.flash_on : Icons.flash_off),
            onPressed: () => _setTorch(!_torchOn),
          ),
        ],
      ),
      body: Stack(
        children: [
          // simple preview (kept)
          Positioned.fill(child: CameraPreview(_controller!)),

          // Center overlay guide (semi-transparent square like a face)
          IgnorePointer(
            child: Center(
              child: FractionallySizedBox(
                widthFactor: 0.8, // tweak size (0.75..0.9)
                child: AspectRatio(
                  aspectRatio: 1,
                  child: Container(
                    margin: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.06), // subtle translucent fill
                      border: Border.all(color: Colors.white70, width: 3), // box outline
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _isUploading ? null : _capture,
        child: const Icon(Icons.camera_alt),
      ),
    );
  }

  @override
  void dispose() {
    _controller?.setFlashMode(FlashMode.off);
    _controller?.dispose();
    super.dispose();
  }
}
