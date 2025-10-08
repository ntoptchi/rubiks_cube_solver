import 'dart:convert';
import 'package:http/http.dart' as http;

const baseUrl = 'http://127.0.0.1:8000/api'; // using adb reverse on a real device

Future<bool> pingHealth() async {
  try {
    final r = await http
        .get(Uri.parse(baseUrl.replaceFirst('/api', '/health')))
        .timeout(const Duration(seconds: 3));
    return r.statusCode == 200;
  } catch (_) {
    return false;
  }
}

Future<Map<String, dynamic>> uploadOneFace(String imagePath) async {
  final uri = Uri.parse('$baseUrl/scan_face');
  final req = http.MultipartRequest('POST', uri)
    ..files.add(await http.MultipartFile.fromPath('image', imagePath));

  final resp = await req.send().timeout(const Duration(seconds: 15));
  final body = await resp.stream.bytesToString();
  if (resp.statusCode != 200) {
    throw Exception('scan_face failed: ${resp.statusCode} $body');
  }
  final map = jsonDecode(body) as Map<String, dynamic>;
  return map;
}

Future<Map<String, dynamic>> solveFromGrids(
  Map<String, List<List<String>>> grids,
) async {
  final uri = Uri.parse('$baseUrl/solve_from_grids');

  final faces = <String, dynamic>{};
  for (final f in ['U','R','F','D','L','B']) {
    faces[f] = {'grid': grids[f], 'rotation': 0};
  }

  final resp = await http
      .post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'faces': faces}),
      )
      .timeout(const Duration(seconds: 30));

  if (resp.statusCode != 200) {
    // Try to surface server message if present
    try {
      final err = jsonDecode(resp.body);
      if (err is Map && err['detail'] != null) {
        throw Exception(err['detail']);
      }
    } catch (_) {}
    throw Exception('server error ${resp.statusCode}');
  }

  final decoded = jsonDecode(resp.body);

  // Always return a Map with a normalized List<String> in 'solution'.
  if (decoded is Map<String, dynamic>) {
    final map = Map<String, dynamic>.from(decoded);
    map['solution'] = _normalizeMoves(map['solution']);
    return map;
  }

  if (decoded is List) {
    return {
      'solution': _normalizeMoves(decoded),
      'textures': null,
      'rotations': null,
    };
  }

  throw Exception('unexpected response shape from server');
}

List<String> _normalizeMoves(dynamic raw) {
  // Accept: List<String>, List<Map>, space-separated String, null, etc.
  if (raw is List) {
    return raw.map<String>((e) {
      if (e is String) return e;
      if (e is Map && e['code'] is String) return e['code'] as String;
      if (e is Map && e['move'] is String) return e['move'] as String;
      return e.toString(); // last-resort stringification
    }).toList();
  }
  if (raw is String) {
    final s = raw.trim();
    return s.isEmpty ? <String>[] : s.split(RegExp(r'\s+')).toList();
  }
  return <String>[];
}


