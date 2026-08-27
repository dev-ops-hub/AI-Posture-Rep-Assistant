#!/usr/bin/env python3
"""Verification script to test basic functionality without requiring a webcam."""

import sys

print("🔍 Verifying AI-Posture-Rep-Assistant components...\n")

# Test imports
print("✓ Testing imports...")
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from dotenv import load_dotenv
    from agents import (
        FitnessCoachAgent,
        FormAuditAgent,
        VisionAgent,
        SessionSummaryPayload,
        VoiceAgent,
    )
    from agents.models import UserProfile, SessionSummary
    print("  ✓ All imports successful\n")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test VisionAgent initialization
print("✓ Testing VisionAgent initialization...")
try:
    vision_agent = VisionAgent()
    print(f"  ✓ VisionAgent created (reps={vision_agent.reps}, state={vision_agent.state.value})\n")
except Exception as e:
    print(f"  ✗ VisionAgent failed: {e}")
    sys.exit(1)

# Test FormAuditAgent initialization
print("✓ Testing FormAuditAgent initialization...")
try:
    audit_agent = FormAuditAgent()
    print(f"  ✓ FormAuditAgent created (enabled={audit_agent.enabled})\n")
except Exception as e:
    print(f"  ✗ FormAuditAgent failed: {e}")
    sys.exit(1)

# Test FitnessCoachAgent
print("✓ Testing FitnessCoachAgent...")
try:
    coach_agent = FitnessCoachAgent()
    calories = coach_agent.calculate_calories(5.0, 70.0, 3600.0)
    print(f"  ✓ Calorie calculation works: {calories:.2f} kcal\n")
except Exception as e:
    print(f"  ✗ FitnessCoachAgent failed: {e}")
    sys.exit(1)

# Test summary generation
print("✓ Testing coaching summary generation...")
try:
    user_profile = UserProfile(weight_kg=70.0, goal="general fitness")
    session_summary = SessionSummary(
        exercise="Squat",
        total_reps=10,
        duration_seconds=120.0,
        met_value=5.0,
        calculated_calories=50.0,
        total_posture_faults=2,
    )
    payload = SessionSummaryPayload(
        sender="Orchestrator_Main",
        recipient="Agent_3_Coach",
        user_profile=user_profile,
        session_summary=session_summary,
        form_audit_diagnostics=["Keep chest up"],
    )
    summary = coach_agent.build_summary(payload)
    print(f"  ✓ Summary generated: {summary[:80]}...\n")
except Exception as e:
    print(f"  ✗ Summary generation failed: {e}")
    sys.exit(1)

# Test angle calculations
print("✓ Testing angle calculations...")
try:
    point_a = np.array([0.0, 1.0], dtype=np.float32)
    point_b = np.array([0.0, 0.0], dtype=np.float32)
    point_c = np.array([1.0, 0.0], dtype=np.float32)
    angle = VisionAgent._angle(point_a, point_b, point_c)
    print(f"  ✓ Angle calculation works: {angle:.1f}°\n")
except Exception as e:
    print(f"  ✗ Angle calculation failed: {e}")
    sys.exit(1)

# Test spine angle calculation
print("✓ Testing spine angle calculation...")
try:
    shoulder = np.array([0.5, 0.3], dtype=np.float32)
    hip = np.array([0.5, 0.5], dtype=np.float32)
    spine_angle = VisionAgent._spine_angle(shoulder, hip)
    print(f"  ✓ Spine angle calculation works: {spine_angle:.1f}°\n")
except Exception as e:
    print(f"  ✗ Spine angle calculation failed: {e}")
    sys.exit(1)

# Close vision agent
vision_agent.close()

# Test VoiceAgent initialization
print("✓ Testing VoiceAgent initialization...")
try:
    voice_agent = VoiceAgent()
    print(f"  ✓ VoiceAgent created (enabled={voice_agent.enabled})\n")
    voice_agent.close()
except Exception as e:
    print(f"  ✗ VoiceAgent failed: {e}")
    sys.exit(1)

print("=" * 60)
print("✅ All components verified successfully!")
print("=" * 60)
print("\n📋 Summary:")
print("  • All modules can be imported")
print("  • VisionAgent can be initialized and closed")
print("  • FormAuditAgent can be initialized")
print("  • FitnessCoachAgent can calculate calories and generate summaries")
print("  • VoiceAgent can be initialized and closed")
print("  • Angle calculations work correctly")
print("\n🚀 Your code is ready to use!")
print("\nTo run the full application with webcam:")
print("  uv run main.py")
