import 'package:flutter/material.dart';

class SolveCoachPage extends StatefulWidget {
  const SolveCoachPage({
    Key? key,
    required this.faceGrids,          // Map<String, List<List<String>>> with keys U,R,F,D,L,B
    this.moves = const <String>[],    // List<String> of moves (e.g. ["F'", "R2", "U"])
  }) : super(key: key);

  final Map<String, List<List<String>>> faceGrids;
  final List<String> moves;

  @override
  State<SolveCoachPage> createState() => _SolveCoachPageState();
}

class _SolveCoachPageState extends State<SolveCoachPage> {
  int _step = 0;

  // Color lookup for preview squares
  static const Map<String, Color> _colorMap = {
    'W': Colors.white,
    'Y': Colors.yellow,
    'R': Colors.red,
    'O': Colors.orange,
    'G': Colors.green,
    'B': Colors.blue,
  };

  static const Map<String, String> _colorNames = {
    'W': 'White',
    'Y': 'Yellow',
    'R': 'Red',
    'O': 'Orange',
    'G': 'Green',
    'B': 'Blue',
  };

  static const Map<String, String> _faceNames = {
    'U': 'Up',
    'R': 'Right',
    'F': 'Front',
    'D': 'Down',
    'L': 'Left',
    'B': 'Back',
  };

  String _centerLetter(String face) {
    final g = widget.faceGrids[face];
    if (g == null || g.length != 3 || g[1].length != 3) return '?';
    return g[1][1];
  }

  String _centerName(String face) {
    return _colorNames[_centerLetter(face)] ?? '?';
  }

  /// Turn e.g. "F'", "R2", "U" into human-friendly instructions.
  String _humanizeMove(String m) {
    if (m.isEmpty) return '';
    final face = m[0]; // U R F D L B
    String suffix = '';
    if (m.length > 1) {
      suffix = m.substring(1); // ' or 2
    }

    final faceLabel = _faceNames[face] ?? face;
    switch (suffix) {
      case "'":
        return "Turn the $faceLabel face counter-clockwise 90°";
      case "2":
        return "Turn the $faceLabel face 180°";
      default:
        return "Turn the $faceLabel face clockwise 90°";
    }
  }

  Widget _faceChip(String faceKey) {
    final name = _faceNames[faceKey] ?? faceKey;
    final center = _centerLetter(faceKey);
    final centerName = _centerName(faceKey);
    final color = _colorMap[center] ?? Colors.grey;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      margin: const EdgeInsets.only(right: 8, bottom: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.07),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white12),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(width: 14, height: 14, color: color, margin: const EdgeInsets.only(right: 8)),
          Text('$name = $centerName',
              style: const TextStyle(fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }

  Widget _miniGrid(String faceKey) {
    final g = widget.faceGrids[faceKey] ?? const [
      ['?','?','?'],
      ['?','?','?'],
      ['?','?','?'],
    ];

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          _faceNames[faceKey] ?? faceKey,
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.all(6),
          decoration: BoxDecoration(
            border: Border.all(color: Colors.white24),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: List.generate(3, (r) {
              return Row(
                mainAxisSize: MainAxisSize.min,
                children: List.generate(3, (c) {
                  final ch = g[r][c];
                  final col = _colorMap[ch] ?? Colors.grey.shade700;
                  return Container(
                    width: 18,
                    height: 18,
                    margin: const EdgeInsets.all(2),
                    decoration: BoxDecoration(
                      color: col,
                      border: Border.all(color: Colors.black26, width: 1),
                      borderRadius: BorderRadius.circular(3),
                    ),
                  );
                }),
              );
            }),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final moves = widget.moves;
    final total = moves.length;
    final move = (total == 0 || _step >= total) ? '' : moves[_step];
    final niceText = _humanizeMove(move);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Solve Coach'),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 100),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Orientation chips (Up/Front/Right etc.)
              const Text(
                'Hold the cube like this:',
                style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
              ),
              const SizedBox(height: 8),
              Wrap(
                children: [
                  _faceChip('U'),
                  _faceChip('F'),
                  _faceChip('R'),
                  _faceChip('D'),
                  _faceChip('L'),
                  _faceChip('B'),
                ],
              ),
              const SizedBox(height: 12),

              // Mini face previews (overflow-safe)
              const Text(
                'Your scanned faces:',
                style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 16,
                runSpacing: 16,
                children: const ['U','R','F','D','L','B']
                    .map((f) => _MiniGridWrapper(faceKey: f))
                    .toList()
                    .map((w) => w.buildWith(_miniGrid))
                    .toList(),
              ),
              const SizedBox(height: 16),
              const Divider(),

              // Move list (tap to jump) — HORIZONTALLY SCROLLABLE
              if (moves.isNotEmpty) ...[
                const Text(
                  'Move list (tap to jump):',
                  style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
                ),
                const SizedBox(height: 8),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.06),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.white10),
                  ),
                  child: SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    child: Row(
                      children: List.generate(moves.length, (i) {
                        final m = moves[i];
                        final isCurrent = i == _step;
                        return Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 4),
                          child: ChoiceChip(
                            selected: isCurrent,
                            onSelected: (_) => setState(() => _step = i),
                            label: Text(
                              m,
                              style: const TextStyle(fontFamily: 'monospace', fontWeight: FontWeight.w700),
                            ),
                          ),
                        );
                      }),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
              ],

              // Current step (also safe for narrow screens)
              Row(
                children: [
                  Expanded(
                    child: SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Row(
                        children: [
                          Text(
                            total == 0 ? 'No moves' : 'Step ${_step + 1} of $total',
                            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                          ),
                          const SizedBox(width: 12),
                          if (total > 0)
                            Chip(
                              label: Text(
                                moves[_step],
                                style: const TextStyle(
                                  fontFamily: 'monospace',
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),

              // Instruction box
              if (total > 0)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.06),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.white10),
                  ),
                  child: Text(
                    niceText,
                    style: const TextStyle(fontSize: 18, height: 1.4),
                  ),
                )
              else
                const Text(
                  'Solver returned no moves.',
                  style: TextStyle(color: Colors.white70),
                ),

              const SizedBox(height: 16),

              // Legend / help
              ExpansionTile(
                tilePadding: EdgeInsets.zero,
                title: const Text(
                  'What does U, R, F, D, L, B mean?',
                  style: TextStyle(fontWeight: FontWeight.w600),
                ),
                children: const [
                  Padding(
                    padding: EdgeInsets.only(bottom: 12),
                    child: Text(
                      'U=Up, R=Right, F=Front, D=Down, L=Left, B=Back.\n'
                      'A plain letter (e.g., F) means turn that face clockwise 90°.\n'
                      'A prime (′) like F′ means counter-clockwise 90°.\n'
                      'A “2” like R2 means turn that face 180°.',
                      style: TextStyle(color: Colors.black),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),

      // Navigation buttons
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
        child: Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _step > 0
                    ? () => setState(() => _step -= 1)
                    : null,
                icon: const Icon(Icons.arrow_back),
                label: const Text('Back'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: ElevatedButton.icon(
                onPressed: (widget.moves.isEmpty || _step >= total - 1)
                    ? null
                    : () => setState(() => _step += 1),
                icon: const Icon(Icons.arrow_forward),
                label: const Text('Next'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Tiny helper to pass the builder into Wrap nicely without closures capturing context weirdly.
class _MiniGridWrapper {
  const _MiniGridWrapper({required this.faceKey});
  final String faceKey;

  Widget buildWith(Widget Function(String) builder) => builder(faceKey);
}
