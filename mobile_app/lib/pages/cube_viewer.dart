import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:vector_math/vector_math_64.dart' as vm;

class CubeViewer extends StatefulWidget {
  /// Absolute URLs for each face: keys must be U, R, F, D, L, B
  final Map<String, String> textures;
  const CubeViewer({Key? key, required this.textures}) : super(key: key);

  @override
  State<CubeViewer> createState() => _CubeViewerState();
}

class _CubeViewerState extends State<CubeViewer> {
  // --- Interaction state (single scale recognizer) ---
  Offset _lastFocal = Offset.zero;
  double _baseZoom = 1.0;

  double _zoom = 0.9;  // start slightly zoomed out
  double _yaw = 0.2;   // a bit rotated so you can see 3 faces on open
  double _pitch = -0.15;

  static const double _persp = 0.0015; // perspective strength

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Cube • 3D Preview')),
      backgroundColor: Colors.black,
      body: LayoutBuilder(
        builder: (ctx, constraints) {
          final size = constraints.biggest;
          final s = math.min(size.width, size.height) * 0.70; // face size
          final depth = s / 2;

          // Build face specs with per-face orientation and position sign (±depth)
          final faces = <_FaceSpec>[
            _FaceSpec('F', widget.textures['F'], rx: 0,            ry: 0,             tzSign: 1),
            _FaceSpec('B', widget.textures['B'], rx: 0,            ry: math.pi,       tzSign: -1),
            _FaceSpec('U', widget.textures['U'], rx: -math.pi / 2, ry: 0,             tzSign: 1),
            _FaceSpec('D', widget.textures['D'], rx:  math.pi / 2, ry: 0,             tzSign: 1),
            _FaceSpec('R', widget.textures['R'], rx: 0,            ry:  math.pi / 2,  tzSign: 1),
            _FaceSpec('L', widget.textures['L'], rx: 0,            ry: -math.pi / 2,  tzSign: 1),
          ];

          // Depth sort so farther faces are painted first (basic painter’s algorithm)
          final sorted = _sortByDepth(faces, depth);

          return GestureDetector(
            behavior: HitTestBehavior.opaque,
            onScaleStart: (details) {
              _lastFocal = details.focalPoint;
              _baseZoom = _zoom;
            },
            onScaleUpdate: (details) {
              final newZoom = (_baseZoom * details.scale).clamp(0.5, 2.5);
              final delta = details.focalPoint - _lastFocal;
              _lastFocal = details.focalPoint;

              setState(() {
                _zoom = newZoom;
                _yaw   += delta.dx * 0.01;
                _pitch  = (_pitch + delta.dy * 0.01).clamp(-1.45, 1.45);
              });
            },
            child: Center(
              child: SizedBox(
                width: s * 2.0,   // give faces room to rotate without clipping
                height: s * 2.0,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    for (final f in sorted)
                      _buildFaceWidget(f, size: s, depth: depth),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  // ---------- Math helpers ----------

  /// Sort faces back-to-front using cube rotation + face local rotation + translation.
  List<_FaceSpec> _sortByDepth(List<_FaceSpec> faces, double depth) {
    final rcube = vm.Matrix4.identity()
      ..rotateX(_pitch)
      ..rotateY(_yaw);

    double zOf(_FaceSpec f) {
      final rface = vm.Matrix4.identity()
        ..rotateX(f.rx)
        ..rotateY(f.ry);
      final t = vm.Matrix4.identity()..translate(0.0, 0.0, f.tzSign * depth);

      // world = rcube * rface * t * origin
      final m = rcube * rface * t;
      final p = vm.Vector3.zero();
      final out = m.transform3(p);
      return out.z;
    }

    final list = faces.toList();
    list.sort((a, b) => zOf(a).compareTo(zOf(b))); // far (small z) first
    return list;
  }

  /// Build the transform for a given face: P * (Rcube * Rface * T * S)
  Matrix4 _matrixForFace(_FaceSpec f, double depth) {
    final persp = Matrix4.identity()..setEntry(3, 2, _persp);

    final rcube = Matrix4.identity()
      ..rotateX(_pitch)
      ..rotateY(_yaw);

    final rface = Matrix4.identity()
      ..rotateX(f.rx)
      ..rotateY(f.ry);

    final t = Matrix4.identity()..translate(0.0, 0.0, f.tzSign * depth);
    final s = Matrix4.identity()..scale(_zoom, _zoom, _zoom);

    // Order matters (right-multiplied): persp * rcube * rface * t * s
    final m = persp;
    m.multiply(rcube);
    m.multiply(rface);
    m.multiply(t);
    m.multiply(s);
    return m;
  }

  // ---------- UI helpers ----------

  Widget _buildFaceWidget(_FaceSpec f, {required double size, required double depth}) {
    final mat = _matrixForFace(f, depth);

    final child = ClipRRect(
      borderRadius: BorderRadius.circular(size * 0.06),
      child: _FaceImage(letter: f.key, url: f.url, size: size),
    );

    return Transform(
      alignment: Alignment.center,
      transform: mat,
      child: SizedBox(width: size, height: size, child: child),
    );
  }
}

// A face spec holds orientation (rx, ry) and whether it sits at +depth or -depth along local Z.
class _FaceSpec {
  final String key;     // U,R,F,D,L,B (for debug/fallback)
  final String? url;    // texture URL
  final double rx;      // rotation around X (radians)
  final double ry;      // rotation around Y (radians)
  final int tzSign;     // +1 or -1

  const _FaceSpec(this.key, this.url, {required this.rx, required this.ry, required this.tzSign});
}

// Network image with a graceful fallback block (letter on colored tile) if the URL is null or fails.
class _FaceImage extends StatelessWidget {
  final String letter;
  final String? url;
  final double size;
  const _FaceImage({Key? key, required this.letter, required this.url, required this.size}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final fallback = _fallbackTile(letter);

    if (url == null || url!.isEmpty) {
      return fallback;
    }
    return Image.network(
      url!,
      fit: BoxFit.cover,
      width: size,
      height: size,
      errorBuilder: (_, __, ___) => fallback,
      // Optional: add a tiny placeholder while loading
      loadingBuilder: (ctx, child, progress) {
        if (progress == null) return child;
        return Stack(
          fit: StackFit.expand,
          children: [
            fallback,
            Container(
              color: Colors.black26,
              child: const Center(child: CircularProgressIndicator(strokeWidth: 2)),
            ),
          ],
        );
      },
    );
  }

  Widget _fallbackTile(String k) {
    final color = {
      'U': const Color(0xFFFFFFFF), // white
      'R': const Color(0xFFFF0000), // red
      'F': const Color(0xFF00AA00), // green
      'D': const Color(0xFFFFFF00), // yellow
      'L': const Color(0xFFFF7F00), // orange-ish
      'B': const Color(0xFF0000FF), // blue
    }[k] ?? Colors.grey;

    final textColor = (k == 'U' || k == 'D') ? Colors.black : Colors.white;

    return Container(
      color: color,
      alignment: Alignment.center,
      child: Text(
        k,
        style: TextStyle(
          color: textColor,
          fontSize: size * 0.35,
          fontWeight: FontWeight.w900,
          shadows: const [Shadow(color: Colors.black26, blurRadius: 4)],
        ),
      ),
    );
  }
}
