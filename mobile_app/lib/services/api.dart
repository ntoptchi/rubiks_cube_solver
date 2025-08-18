import 'dart:convert';
import 'package:http/http.dart' as http;

/// Use 127.0.0.1 if you're on a real device with `adb reverse tcp:8000 tcp:8000`
/// Use 10.0.2.2 if you're on the Android emulator.
const String baseUrl = 'http://127.0.0.1:8000';

Future<bool> pingHealth() async {
  try {
    final r = await http.get(Uri.parse('$baseUrl/health'))
                        .timeout(const Duration(seconds: 3));
    return r.statusCode == 200;
  } catch (_) {
    return false;
  }
}

Future<Map<String, dynamic>> uploadOneFace(String imagePath) async {
  final uri = Uri.parse('$baseUrl/scan_face'); // <-- POST /scan_face
  final req = http.MultipartRequest('POST', uri)
    ..files.add(await http.MultipartFile.fromPath('image', imagePath)); // field 'image'

  // Debug
  // ignore: avoid_print
  print('→ POST $uri (image=$imagePath)');
  final resp = await req.send();
  final body = await resp.stream.bytesToString();
  // ignore: avoid_print
  print('← ${resp.statusCode} $body');

  if (resp.statusCode != 200) {
    throw Exception('scan_face failed: ${resp.statusCode} $body');
  }
  return jsonDecode(body) as Map<String, dynamic>;
}

/// Optional helper for the final call
Future<Map<String, dynamic>> solveFromGrids(
  Map<String, List<List<String>>> grids,
) async {
  final uri = Uri.parse('$baseUrl/solve_from_grids');
  final payload = {
    'faces': grids.map((k, v) => MapEntry(k, {'grid': v, 'rotation': 0})),
  };
  final r = await http.post(
    uri,
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode(payload),
  );
  if (r.statusCode != 200) {
    throw Exception('solve_from_grids failed: ${r.statusCode} ${r.body}');
  }
  return jsonDecode(r.body) as Map<String, dynamic>;
}
