import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import '../services/api.dart'; // uploadOneFace, solveFromGrids

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

  // Torch toggle (kept; shown in AppBar)
  bool _torchOn = false;

  // Per-face flow state (order must match your backend expectation)
  final List<String> _faceOrder = ['U', 'R', 'F', 'D', 'L', 'B'];
  int _currentFace = 0;
  final Map<String, List<List<String>>> _grids = {};

  // For 3×3 preview dialog
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

      // Log the grid in console (handy during tuning)
      for (final row in grid) {
        debugPrint(row.join(' '));
      }

      // Ask user to accept or retake this face
      final ok = await _confirmUseFace(currentLabel, grid);
      if (!mounted) return;

      if (!ok) {
        // User chose Retake → do not store or advance
        setState(() => _isUploading = false);
        return;
      }

      _grids[currentLabel] = grid;

      // Next face
      setState(() => _currentFace++);

      // When all six faces captured → solve
      if (_currentFace == _faceOrder.length) {
        final solved = await solveFromGrids(_grids);
        final textures = Map<String, dynamic>.from(solved['textures'] as Map);

        // Backend already returns absolute URLs (Option A)
        final urls = <String, String>{
          'U': textures['U'] as String,
          'R': textures['R'] as String,
          'F': textures['F'] as String,
          'D': textures['D'] as String,
          'L': textures['L'] as String,
          'B': textures['B'] as String,
        };

        if (!mounted) return;
        setState(() => _isUploading = false);

        // Navigate to your 3D viewer (expects Map<String,String> argument)
        Navigator.pushNamed(context, '/viewer', arguments: urls);
      } else {
        if (mounted) setState(() => _isUploading = false);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _isUploading = false);
      _showError('Scan failed: $e');
      // Do not advance; user can tap again to retake this face.
    }
  }

  // Kept (not exposed in the old UI)
  void _retakePrevious() {
    if (_isUploading || _currentFace == 0) return;
    final prev = _faceOrder[_currentFace - 1];
    setState(() {
      _grids.remove(prev);
      _currentFace -= 1;
    });
  }

  // Kept (not exposed in the old UI)
  void _resetAll() {
    if (_isUploading) return;
    setState(() {
      _grids.clear();
      _currentFace = 0;
    });
  }

  @override
  Widget build(BuildContext context) {
    // old simple spinners while busy/uninitialized
    if (!_isCameraReady || _controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    if (_isUploading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    // pretty label like before
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
          // classic simple preview (no extra scaling logic)
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
                      border: Border.all(color: Colors.white70, width: 3),
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
