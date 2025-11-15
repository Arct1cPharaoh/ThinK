import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  runApp(const CalorieCountingApp());
}

/// ---------- Retro colors ----------
const Color kRetroDarkPurple = Color(0xFF1B0033);
const Color kRetroSurface = Color(0xFF261447);
const Color kRetroMagenta = Color(0xFFFF5DA2);
const Color kRetroOrange = Color(0xFFFFC857);
const Color kRetroTeal = Color(0xFF00F5D4);

final ThemeData retroTheme = ThemeData(
  brightness: Brightness.dark,
  scaffoldBackgroundColor: kRetroDarkPurple,
  useMaterial3: true,
  colorScheme: const ColorScheme.dark(
    primary: kRetroMagenta,
    secondary: kRetroTeal,
    surface: kRetroSurface,
    background: kRetroDarkPurple,
  ),
  textTheme: const TextTheme(
    headlineMedium: TextStyle(
      fontWeight: FontWeight.bold,
      color: Colors.white,
    ),
    bodyLarge: TextStyle(
      color: Colors.white70,
    ),
    bodyMedium: TextStyle(
      color: Colors.white70,
    ),
  ),
);

/// ---------- Models & helpers ----------

enum Gender { male, female, other }

class ProfileData {
  double heightCm;
  double weightKg;
  Gender gender;
  double lossLbsPerWeek; // 0 = maintain

  ProfileData({
    required this.heightCm,
    required this.weightKg,
    required this.gender,
    required this.lossLbsPerWeek,
  });

  Map<String, dynamic> toJson() => {
        'heightCm': heightCm,
        'weightKg': weightKg,
        'genderIndex': gender.index,
        'lossLbsPerWeek': lossLbsPerWeek,
      };

  factory ProfileData.fromJson(Map<String, dynamic> json) {
    return ProfileData(
      heightCm: (json['heightCm'] as num).toDouble(),
      weightKg: (json['weightKg'] as num).toDouble(),
      gender: Gender.values[(json['genderIndex'] as int).clamp(0, 2)],
      lossLbsPerWeek: (json['lossLbsPerWeek'] as num).toDouble(),
    );
  }

  ProfileData copyWith({
    double? heightCm,
    double? weightKg,
    Gender? gender,
    double? lossLbsPerWeek,
  }) {
    return ProfileData(
      heightCm: heightCm ?? this.heightCm,
      weightKg: weightKg ?? this.weightKg,
      gender: gender ?? this.gender,
      lossLbsPerWeek: lossLbsPerWeek ?? this.lossLbsPerWeek,
    );
  }
}

class MealEntry {
  final String name;
  final double calories;
  final DateTime time;
  final String? imagePath;

  MealEntry({
    required this.name,
    required this.calories,
    required this.time,
    this.imagePath,
  });

  Map<String, dynamic> toJson() => {
        'name': name,
        'calories': calories,
        'time': time.toIso8601String(),
        'imagePath': imagePath,
      };

  factory MealEntry.fromJson(Map<String, dynamic> json) {
    return MealEntry(
      name: json['name'] as String,
      calories: (json['calories'] as num).toDouble(),
      time: DateTime.parse(json['time'] as String),
      imagePath: json['imagePath'] as String?,
    );
  }
}

bool isSameDay(DateTime a, DateTime b) {
  return a.year == b.year && a.month == b.month && a.day == b.day;
}

/// Rough maintenance calories based on weight & gender.
double maintenanceCalories(ProfileData profile) {
  final double basePerKg = switch (profile.gender) {
    Gender.male => 24.0,
    Gender.female => 22.0,
    Gender.other => 23.0,
  };

  const activityFactor = 1.2; // mostly sedentary
  return profile.weightKg * basePerKg * activityFactor;
}

/// 500 kcal/day deficit ≈ 1 lb/week
double targetCalories(ProfileData profile) {
  final maintenance = maintenanceCalories(profile);
  final deficitPerDay = profile.lossLbsPerWeek * 500.0;
  final target = maintenance - deficitPerDay;
  return target.clamp(1000.0, 5000.0);
}

/// Helper conversions between imperial & metric (for internal math)

double lbsToKg(double lbs) => lbs / 2.20462;
double kgToLbs(double kg) => kg * 2.20462;
double inchesToCm(double inches) => inches * 2.54;
double cmToInches(double cm) => cm / 2.54;

/// ---------- Root app with bottom navigation + persistence ----------

class CalorieCountingApp extends StatefulWidget {
  const CalorieCountingApp({super.key});

  @override
  State<CalorieCountingApp> createState() => _CalorieCountingAppState();
}

class _CalorieCountingAppState extends State<CalorieCountingApp> {
  int _selectedIndex = 0;

  ProfileData _profile = ProfileData(
    heightCm: inchesToCm(69), // ~5'9"
    weightKg: lbsToKg(154),   // ~154 lbs
    gender: Gender.male,
    lossLbsPerWeek: 0.0,      // maintain by default
  );

  final List<MealEntry> _meals = [];
  bool _isLoading = true;

  static const _profileKey = 'profile';
  static const _mealsKey = 'meals';

  @override
  void initState() {
    super.initState();
    _loadState();
  }

  Future<void> _loadState() async {
    final prefs = await SharedPreferences.getInstance();
    final profileString = prefs.getString(_profileKey);
    final mealsString = prefs.getString(_mealsKey);

    if (profileString != null) {
      try {
        final map = jsonDecode(profileString) as Map<String, dynamic>;
        _profile = ProfileData.fromJson(map);
      } catch (_) {
        // ignore and keep defaults
      }
    }

    if (mealsString != null) {
      try {
        final list = jsonDecode(mealsString) as List<dynamic>;
        _meals
          ..clear()
          ..addAll(
            list
                .whereType<Map<String, dynamic>>()
                .map(MealEntry.fromJson),
          );
      } catch (_) {
        // ignore bad data
      }
    }

    setState(() {
      _isLoading = false;
    });
  }

  Future<void> _saveProfile() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_profileKey, jsonEncode(_profile.toJson()));
  }

  Future<void> _saveMeals() async {
    final prefs = await SharedPreferences.getInstance();
    final list = _meals.map((m) => m.toJson()).toList();
    await prefs.setString(_mealsKey, jsonEncode(list));
  }

  void _onNavTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  void _updateProfile(ProfileData updated) {
    setState(() {
      _profile = updated;
    });
    _saveProfile();
  }

  void _addMeal(MealEntry meal) {
    setState(() {
      _meals.add(meal);
    });
    _saveMeals();
  }

  void _updateMealAt(int index, MealEntry updated) {
    setState(() {
      _meals[index] = updated;
    });
    _saveMeals();
  }

  void _deleteMealAt(int index) {
    setState(() {
      _meals.removeAt(index);
    });
    _saveMeals();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      // Simple loading screen while reading prefs
      return MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: retroTheme,
        home: const Scaffold(
          body: Center(
            child: CircularProgressIndicator(),
          ),
        ),
      );
    }

    Widget body;
    String title;

    switch (_selectedIndex) {
      case 0:
        title = "Today’s Vibes";
        body = HomePage(
          profile: _profile,
          meals: _meals,
          onUpdateMeal: _updateMealAt,
          onDeleteMeal: _deleteMealAt,
        );
        break;
      case 1:
        title = "Snap a Meal";
        body = AddMealPage(profile: _profile, onMealAdded: _addMeal);
        break;
      case 2:
      default:
        title = "Profile & Goals";
        body =
            ProfilePage(profile: _profile, onProfileChanged: _updateProfile);
        break;
    }

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: retroTheme,
      home: Scaffold(
        appBar: AppBar(
          title: Text(title),
          backgroundColor: Colors.transparent,
        ),
        body: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [kRetroDarkPurple, kRetroSurface],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          child: SafeArea(child: body),
        ),
        bottomNavigationBar: BottomNavigationBar(
          backgroundColor: kRetroSurface,
          selectedItemColor: kRetroMagenta,
          unselectedItemColor: Colors.white60,
          currentIndex: _selectedIndex,
          onTap: _onNavTapped,
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.bubble_chart),
              label: 'Today',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.camera_alt),
              label: 'Add Meal',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.person),
              label: 'Profile',
            ),
          ],
        ),
      ),
    );
  }
}

/// ---------- Home page (summary + today’s meals, with edit mode) ----------

class HomePage extends StatefulWidget {
  final ProfileData profile;
  final List<MealEntry> meals;
  final void Function(int, MealEntry) onUpdateMeal;
  final void Function(int) onDeleteMeal;

  const HomePage({
    super.key,
    required this.profile,
    required this.meals,
    required this.onUpdateMeal,
    required this.onDeleteMeal,
  });

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool _isEditing = false;

  void _enterEditingMode() {
    if (!_isEditing) {
      setState(() {
        _isEditing = true;
      });
    }
  }

  void _exitEditingMode() {
    if (_isEditing) {
      setState(() {
        _isEditing = false;
      });
    }
  }

  Future<void> _showEditMealDialog(MealEntry meal, int index) async {
    final nameController = TextEditingController(text: meal.name);
    final caloriesController =
        TextEditingController(text: meal.calories.toStringAsFixed(0));

    await showDialog(
      context: context,
      builder: (ctx) {
        return AlertDialog(
          backgroundColor: kRetroSurface,
          title: const Text("Edit meal"),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: nameController,
                decoration: const InputDecoration(
                  labelText: "Meal name",
                ),
              ),
              const SizedBox(height: 8),
              TextField(
                controller: caloriesController,
                decoration: const InputDecoration(
                  labelText: "Calories (kcal)",
                ),
                keyboardType: TextInputType.number,
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(ctx).pop(),
              child: const Text("Cancel"),
            ),
            TextButton(
              onPressed: () {
                final newName = nameController.text.trim().isEmpty
                    ? meal.name
                    : nameController.text.trim();
                final newCalories =
                    double.tryParse(caloriesController.text) ?? meal.calories;

                final updated = MealEntry(
                  name: newName,
                  calories: newCalories,
                  time: meal.time,
                  imagePath: meal.imagePath,
                );

                widget.onUpdateMeal(index, updated);
                Navigator.of(ctx).pop();
              },
              child: const Text("Save"),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final today = DateTime.now();

    // Map today's meals to their original indices, so edits/deletes map back
    final List<int> todayIndexes = [];
    for (var i = 0; i < widget.meals.length; i++) {
      if (isSameDay(widget.meals[i].time, today)) {
        todayIndexes.add(i);
      }
    }
    final todaysMeals = todayIndexes.map((i) => widget.meals[i]).toList();

    final goal = targetCalories(widget.profile);
    final eaten =
        todaysMeals.fold<double>(0, (sum, m) => sum + m.calories);
    final remaining = (goal - eaten).clamp(0, goal);
    final pct = goal > 0 ? (eaten / goal).clamp(0.0, 1.0) : 0.0;

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          _RetroCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Daily Disco Calories",
                  style: Theme.of(context).textTheme.headlineMedium,
                ),
                const SizedBox(height: 8),
                Text(
                  "Stay in groove with your goal.",
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
                const SizedBox(height: 16),
                LinearProgressIndicator(
                  value: pct,
                  minHeight: 10,
                  borderRadius: BorderRadius.circular(999),
                  backgroundColor: Colors.white10,
                  valueColor:
                      const AlwaysStoppedAnimation<Color>(kRetroMagenta),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    _SummaryNumber(
                      label: "Goal",
                      value: goal.round(),
                      color: kRetroTeal,
                    ),
                    const SizedBox(width: 12),
                    _SummaryNumber(
                      label: "Eaten",
                      value: eaten.round(),
                      color: kRetroOrange,
                    ),
                    const SizedBox(width: 12),
                    _SummaryNumber(
                      label: "Left",
                      value: remaining.round(),
                      color: kRetroMagenta,
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          if (todaysMeals.isNotEmpty)
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  _isEditing
                      ? "Editing mode: tap Done when finished."
                      : "Long-press a meal to edit or delete.",
                  style: Theme.of(context).textTheme.bodySmall,
                ),
                if (_isEditing)
                  TextButton.icon(
                    onPressed: _exitEditingMode,
                    icon: const Icon(Icons.check, color: kRetroTeal),
                    label: const Text("Done"),
                  ),
              ],
            ),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              "Today’s Eats",
              style: Theme.of(context).textTheme.headlineMedium,
            ),
          ),
          const SizedBox(height: 8),
          if (todaysMeals.isEmpty)
            const Text(
              "No meals logged yet. Snap your first disco snack!",
            )
          else
            Expanded(
              child: ListView.builder(
                itemCount: todaysMeals.length,
                itemBuilder: (context, index) {
                  final mealIndex = todayIndexes[index];
                  final meal = todaysMeals[index];
                  return _MealTile(
                    meal: meal,
                    isEditing: _isEditing,
                    onLongPressEnterEdit: _enterEditingMode,
                    onEdit: () {
                      _showEditMealDialog(meal, mealIndex);
                    },
                    onDelete: () {
                      widget.onDeleteMeal(mealIndex);
                    },
                  );
                },
              ),
            ),
        ],
      ),
    );
  }
}

class _SummaryNumber extends StatelessWidget {
  final String label;
  final int value;
  final Color color;

  const _SummaryNumber({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding:
            const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
        decoration: BoxDecoration(
          color: Colors.white10,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color.withOpacity(0.5), width: 1.5),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label.toUpperCase(),
                style: Theme.of(context).textTheme.bodySmall),
            const SizedBox(height: 4),
            Text(
              "$value kcal",
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    color: color,
                    fontWeight: FontWeight.bold,
                  ),
            ),
          ],
        ),
      ),
    );
  }
}

class _MealTile extends StatelessWidget {
  final MealEntry meal;
  final bool isEditing;
  final VoidCallback onLongPressEnterEdit;
  final VoidCallback onEdit;
  final VoidCallback onDelete;

  const _MealTile({
    required this.meal,
    required this.isEditing,
    required this.onLongPressEnterEdit,
    required this.onEdit,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onLongPress: onLongPressEnterEdit,
      child: Card(
        color: kRetroSurface.withOpacity(0.9),
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16)),
        margin: const EdgeInsets.symmetric(vertical: 6),
        child: ListTile(
          leading: meal.imagePath != null
              ? ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.file(
                    File(meal.imagePath!),
                    width: 48,
                    height: 48,
                    fit: BoxFit.cover,
                  ),
                )
              : const Icon(Icons.restaurant, color: kRetroTeal),
          title: Text(meal.name),
          subtitle: Text(
            "${meal.calories.toStringAsFixed(0)} kcal • "
            "${TimeOfDay.fromDateTime(meal.time).format(context)}",
          ),
          trailing: isEditing
              ? Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      icon:
                          const Icon(Icons.edit, color: kRetroTeal),
                      onPressed: onEdit,
                    ),
                    IconButton(
                      icon: const Icon(Icons.delete,
                          color: kRetroMagenta),
                      onPressed: onDelete,
                    ),
                  ],
                )
              : null,
        ),
      ),
    );
  }
}

/// Simple card with retro styling
class _RetroCard extends StatelessWidget {
  final Widget child;

  const _RetroCard({required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: kRetroSurface.withOpacity(0.95),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: kRetroMagenta.withOpacity(0.4),
          width: 1.5,
        ),
        boxShadow: const [
          BoxShadow(
            color: Colors.black54,
            blurRadius: 16,
            offset: Offset(0, 6),
          ),
        ],
      ),
      child: child,
    );
  }
}

/// ---------- Add Meal page (camera + mock AI, keyboard-safe) ----------

class AddMealPage extends StatefulWidget {
  final ProfileData profile;
  final void Function(MealEntry) onMealAdded;

  const AddMealPage({
    super.key,
    required this.profile,
    required this.onMealAdded,
  });

  @override
  State<AddMealPage> createState() => _AddMealPageState();
}

class _AddMealPageState extends State<AddMealPage> {
  final ImagePicker _picker = ImagePicker();
  XFile? _imageFile;
  bool _isEstimating = false;
  double? _estimatedCalories;
  final TextEditingController _mealNameController =
      TextEditingController(text: "Meal");

  Future<void> _takePhoto() async {
    try {
      final XFile? photo = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1024,
        imageQuality: 85,
      );
      if (photo == null) return;

      setState(() {
        _imageFile = photo;
        _estimatedCalories = null;
      });
    } catch (e) {
      _showSnackBar("Error opening camera: $e");
    }
  }

  Future<void> _mockAnalyzeCalories() async {
    if (_imageFile == null) {
      _showSnackBar("Take a photo first!");
      return;
    }

    setState(() {
      _isEstimating = true;
      _estimatedCalories = null;
    });

    // TODO: Replace this with actual backend call.
    await Future.delayed(const Duration(seconds: 1));

    setState(() {
      _isEstimating = false;
      _estimatedCalories = 550; // mock value
    });
  }

  void _saveMeal() {
    if (_estimatedCalories == null) {
      _showSnackBar("Analyze the photo first to get calories.");
      return;
    }

    final meal = MealEntry(
      name: _mealNameController.text.trim().isEmpty
          ? "Meal"
          : _mealNameController.text.trim(),
      calories: _estimatedCalories!,
      time: DateTime.now(),
      imagePath: _imageFile?.path,
    );

    widget.onMealAdded(meal);
    _showSnackBar("Meal added to today!");

    setState(() {
      _imageFile = null;
      _estimatedCalories = null;
      _mealNameController.text = "Meal";
    });
  }

  void _showSnackBar(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final bottomInset = MediaQuery.of(context).viewInsets.bottom;

    // Use a scroll view + bottom padding so keyboard never causes overflow
    return SingleChildScrollView(
      padding: EdgeInsets.fromLTRB(16, 16, 16, bottomInset + 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _RetroCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("Snap Your Snack",
                    style: Theme.of(context).textTheme.headlineMedium),
                const SizedBox(height: 8),
                Text(
                  "Take a disco shot of your food and let the AI guess the calories.",
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(
            height: 220,
            child: Center(
              child: _imageFile == null
                  ? const Text("No photo yet. Hit the camera below!")
                  : ClipRRect(
                      borderRadius: BorderRadius.circular(24),
                      child: Image.file(
                        File(_imageFile!.path),
                        fit: BoxFit.cover,
                        width: double.infinity,
                      ),
                    ),
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _mealNameController,
            decoration: InputDecoration(
              labelText: "Meal name",
              filled: true,
              fillColor: kRetroSurface.withOpacity(0.9),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
              ),
            ),
          ),
          const SizedBox(height: 12),
          if (_estimatedCalories != null)
            Text(
              "Estimated: ${_estimatedCalories!.toStringAsFixed(0)} kcal",
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    color: kRetroTeal,
                    fontWeight: FontWeight.bold,
                  ),
            ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: _takePhoto,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text("Take Photo"),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: FilledButton.icon(
                  onPressed: _isEstimating ? null : _mockAnalyzeCalories,
                  icon: _isEstimating
                      ? const SizedBox(
                          height: 16,
                          width: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.auto_awesome),
                  label: Text(_isEstimating ? "Analyzing..." : "Estimate"),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: FilledButton.icon(
                  onPressed: _saveMeal,
                  icon: const Icon(Icons.save),
                  label: const Text("Save"),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// ---------- Profile page (height slider + weight input) ----------

enum LossTimeframe { week, month }

class ProfilePage extends StatefulWidget {
  final ProfileData profile;
  final void Function(ProfileData) onProfileChanged;

  const ProfilePage({
    super.key,
    required this.profile,
    required this.onProfileChanged,
  });

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  late Gender _gender;

  // Height in inches, controlled by slider
  late double _heightInchesTotal;

  // Weight input controller in lbs
  late TextEditingController _weightController;

  // Weight loss target & timeframe
  double _lossAmount = 0.0; // in lbs (per week or month depending on _lossTimeframe)
  LossTimeframe _lossTimeframe = LossTimeframe.week;

  @override
  void initState() {
    super.initState();
    _gender = widget.profile.gender;

    // Convert cm to inches
    final totalInches = cmToInches(widget.profile.heightCm);
    _heightInchesTotal = totalInches.clamp(48.0, 84.0); // 4'0" to 7'0"

    // Convert kg to lbs
    final weightLbs = kgToLbs(widget.profile.weightKg);
    _weightController =
        TextEditingController(text: weightLbs.toStringAsFixed(1));

    // Convert stored weekly loss into current UI (default to "per week")
    _lossAmount = widget.profile.lossLbsPerWeek;
    _lossTimeframe = LossTimeframe.week;
  }

  @override
  void dispose() {
    _weightController.dispose();
    super.dispose();
  }

  void _saveProfile() {
    // Convert inches to cm
    final heightCm = inchesToCm(_heightInchesTotal);

    // Parse lbs from text
    final parsedLbs =
        double.tryParse(_weightController.text) ?? kgToLbs(widget.profile.weightKg);
    final weightKg = lbsToKg(parsedLbs);

    // Convert current UI loss rate into weekly
    double lossLbsPerWeek;
    if (_lossTimeframe == LossTimeframe.week) {
      lossLbsPerWeek = _lossAmount;
    } else {
      // "per month" -> approximate 4 weeks/month
      lossLbsPerWeek = _lossAmount / 4.0;
    }

    final updated = widget.profile.copyWith(
      heightCm: heightCm,
      weightKg: weightKg,
      gender: _gender,
      lossLbsPerWeek: lossLbsPerWeek,
    );

    widget.onProfileChanged(updated);
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("Profile updated.")),
    );
  }

  @override
  Widget build(BuildContext context) {
    final maintenance = maintenanceCalories(widget.profile);
    final target = targetCalories(widget.profile);
    final lossPerWeekDisplay = _lossTimeframe == LossTimeframe.week
        ? _lossAmount
        : _lossAmount / 4.0;

    final int feet = _heightInchesTotal ~/ 12;
    final int inches = (_heightInchesTotal - feet * 12).round();

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          _RetroCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("Your Disco Stats",
                    style: Theme.of(context).textTheme.headlineMedium),
                const SizedBox(height: 8),
                Text(
                  "We use your height, weight, and desired weight loss speed to estimate daily calories.",
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          // HEIGHT SLIDER
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              "Height",
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            "$feet ft $inches in",
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          Slider(
            value: _heightInchesTotal,
            min: 48.0, // 4'0"
            max: 84.0, // 7'0"
            divisions: 36,
            label: "$feet'${inches.toString().padLeft(2, '0')}\"",
            onChanged: (v) {
              setState(() {
                _heightInchesTotal = v;
              });
            },
          ),
          const SizedBox(height: 16),
          // WEIGHT TEXT FIELD
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              "Weight (lbs)",
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _weightController,
            keyboardType:
                const TextInputType.numberWithOptions(decimal: true),
            decoration: InputDecoration(
              labelText: "Weight in lbs",
              filled: true,
              fillColor: kRetroSurface.withOpacity(0.9),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
              ),
            ),
          ),
          const SizedBox(height: 16),
          // GENDER DROPDOWN
          DropdownButtonFormField<Gender>(
            value: _gender,
            decoration: InputDecoration(
              labelText: "Gender",
              filled: true,
              fillColor: kRetroSurface.withOpacity(0.9),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(16),
              ),
            ),
            items: const [
              DropdownMenuItem(
                value: Gender.male,
                child: Text("Male"),
              ),
              DropdownMenuItem(
                value: Gender.female,
                child: Text("Female"),
              ),
              DropdownMenuItem(
                value: Gender.other,
                child: Text("Other"),
              ),
            ],
            onChanged: (g) {
              if (g != null) {
                setState(() => _gender = g);
              }
            },
          ),
          const SizedBox(height: 16),
          // WEIGHT LOSS RATE
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              "Target weight loss",
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ),
          const SizedBox(height: 8),
          Slider(
            value: _lossAmount,
            min: 0.0,
            max: 3.0,
            divisions: 12, // 0.25 increments
            label:
                "${_lossAmount.toStringAsFixed(2)} lbs ${_lossTimeframe == LossTimeframe.week ? "per week" : "per month"}",
            onChanged: (v) {
              setState(() => _lossAmount = v);
            },
          ),
          const SizedBox(height: 4),
          CupertinoSegmentedControl<LossTimeframe>(
            groupValue: _lossTimeframe,
            onValueChanged: (value) {
              setState(() {
                _lossTimeframe = value;
              });
            },
            children: const {
              LossTimeframe.week: Padding(
                padding:
                    EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                child: Text("Per week"),
              ),
              LossTimeframe.month: Padding(
                padding:
                    EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                child: Text("Per month"),
              ),
            },
          ),
          const SizedBox(height: 8),
          Text(
            "Equivalent weekly loss: ${lossPerWeekDisplay.toStringAsFixed(2)} lbs/week",
            style:
                const TextStyle(fontSize: 12, color: Colors.white70),
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: _saveProfile,
            icon: const Icon(Icons.save),
            label: const Text("Save Profile"),
          ),
          const SizedBox(height: 16),
          _RetroCard(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Estimates",
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 8),
                Text(
                  "Maintenance: ${maintenance.toStringAsFixed(0)} kcal/day",
                ),
                Text(
                  "Target (with goal): ${target.toStringAsFixed(0)} kcal/day",
                ),
                const SizedBox(height: 8),
                const Text(
                  "Uses 500 kcal/day ≈ 1 lb/week rule. Demo-only estimates, not medical advice.",
                  style: TextStyle(fontSize: 12, color: Colors.white60),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
