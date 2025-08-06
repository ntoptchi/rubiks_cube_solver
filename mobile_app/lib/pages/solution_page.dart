import 'package:flutter/material.dart';

class SolutionPage extends StatelessWidget {
  final List<String> moves;

  const SolutionPage({Key? key, required this.moves}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Solution Steps')),
      body: ListView.builder(
        itemCount: moves.length,
        itemBuilder: (context, index) {
          return ListTile(
            leading: Text('${index + 1}.'),
            title: Text(moves[index]),
          );
        },
      ),
    );
  }
}