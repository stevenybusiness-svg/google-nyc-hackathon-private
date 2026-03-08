#!/usr/bin/env python3
"""
Test script to demonstrate billing monitor warnings.
Simulates API usage to trigger threshold alerts.
"""
import sys
import time

sys.path.insert(0, '/Users/stevenyang/Documents/google-nyc-hackathon')

from billing_monitor import log_api_call, get_billing_summary, monitor

print("🧪 Billing Monitor Test\n")
print("This will simulate API calls to trigger warnings at each threshold.\n")
print("=" * 60)

# Simulate a 5-minute Gemini Live call
print("\n1️⃣  Simulating 5-minute audio call...")
log_api_call("gemini_live_audio_input_min", 5.0, "Test call input")
log_api_call("gemini_live_audio_output_min", 5.0, "Test call output")
time.sleep(0.5)

summary = get_billing_summary()
print(f"   Current cost: ${summary['total_cost_usd']:.4f}")
print(f"   Next threshold: ${summary['next_threshold']}")

# Simulate 2 storybook generations
print("\n2️⃣  Simulating 2 storybook generations...")
for i in range(2):
    log_api_call("gemini_flash_image_input", 5, f"Storybook {i+1} input")
    log_api_call("gemini_flash_image_output", 3, f"Storybook {i+1} output")
time.sleep(0.5)

summary = get_billing_summary()
print(f"   Current cost: ${summary['total_cost_usd']:.4f}")

# Simulate memory video generation
print("\n3️⃣  Simulating memory video (8 seconds)...")
log_api_call("gemini_flash_image_output", 5, "Memory video stylization")
log_api_call("veo_video_sec", 8, "Memory video generation")
time.sleep(0.5)

summary = get_billing_summary()
print(f"   Current cost: ${summary['total_cost_usd']:.4f}")

# Simulate another call with vision
print("\n4️⃣  Simulating 3-minute call + 5 vision triggers...")
log_api_call("gemini_live_audio_input_min", 3.0, "Call 2 input")
log_api_call("gemini_live_audio_output_min", 3.0, "Call 2 output")
for i in range(5):
    log_api_call("vision_api_label", 0.001, f"Vision trigger {i+1}")
    log_api_call("gemini_flash_image_input", 0.001, f"Narration {i+1}")
time.sleep(0.5)

summary = get_billing_summary()
print(f"   Current cost: ${summary['total_cost_usd']:.4f}")

# Push towards higher thresholds
print("\n5️⃣  Simulating heavy usage (multiple videos)...")
for i in range(3):
    log_api_call("veo_video_sec", 8, f"Video {i+1}")
    time.sleep(0.3)

summary = get_billing_summary()
print(f"   Current cost: ${summary['total_cost_usd']:.4f}")

print("\n" + "=" * 60)
print("📊 FINAL SUMMARY")
print("=" * 60)
summary = get_billing_summary()
print(f"Total cost:        ${summary['total_cost_usd']:.4f}")
print(f"Runtime:           {summary['runtime_minutes']:.2f} minutes")
print(f"Warnings fired:    {len(summary['warnings_triggered'])}")
print(f"Thresholds hit:    {', '.join(f'${t:.2f}' for t in summary['warnings_triggered'])}")
print(f"Operations:        {summary['operations_count']}")
print(f"Next threshold:    ${summary['next_threshold']:.2f}" if summary['next_threshold'] else "All thresholds passed")

# Show most expensive operations
print("\n💸 Most Expensive Operations:")
most_expensive = sorted(monitor.usage_log, key=lambda x: x['cost_usd'], reverse=True)[:5]
for i, entry in enumerate(most_expensive, 1):
    print(f"   {i}. {entry['description']:30} ${entry['cost_usd']:.4f}")

print("\n✅ Test complete!")
print(f"⚠️  Check the console above for threshold warning messages\n")
