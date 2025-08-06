class CubeState {
  final Map<String, String> faces;

  CubeState({required this.faces});

  Map<String, dynamic> toJson() => {
        'faces': faces,
      };
}