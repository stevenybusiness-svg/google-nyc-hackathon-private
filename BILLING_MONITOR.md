# Billing Monitor

## Overview

The billing monitor tracks API usage in real-time and warns you when spending exceeds predefined thresholds. This helps prevent unexpected costs during development and demo sessions.

## Features

### Real-Time Cost Tracking

The system logs every API call with its approximate cost:

- **Gemini Live Audio Input**: $0.0375 per minute
- **Gemini Live Audio Output**: $0.15 per minute
- **Gemini Text Input**: $0.075 per 1M tokens
- **Gemini Text Output**: $0.30 per 1M tokens
- **Gemini Image Input**: $0.00026 per image (vision narration)
- **Gemini Image Output**: $0.016 per image (stylization)
- **Veo Video**: $0.30 per second
- **Vision API**: $1.50 per 1000 images
- **ElevenLabs STT**: $24 per 1M characters
- **ElevenLabs TTS**: $240 per 1M characters

### Warning Thresholds

Automatic console warnings appear when spending crosses these thresholds:

- $3.00
- $5.00
- $9.00
- $12.00
- $15.00
- $19.00
- $21.00
- $23.00
- $24.90

When a threshold is crossed, the system logs:
- ⚠️ **BILLING ALERT** with current total
- Recent high-cost operations (> $0.01)
- Operation details (API, cost, description)

### Visual UI Indicator

A live billing widget appears in the top-right corner of both the landing page and participant page:

```
💰 BILLING MONITOR
$0.0000
Next alert: $3.00
```

**Color coding:**
- **Green** ($0-$9): Normal operation
- **Yellow-Green** ($10-$14): Moderate usage
- **Yellow** ($15-$19): High usage
- **Orange** ($20-$23): Critical
- **Red** ($24+): Stop immediately

### API Endpoint

**GET** `/api/billing`

Returns:

```json
{
  "summary": {
    "total_cost_usd": 0.0,
    "runtime_minutes": 1.51,
    "warnings_triggered": [],
    "next_threshold": 3.0,
    "operations_count": 0
  },
  "estimates": {
    "5_min_call": "$0.9375",
    "storybook_5_screenshots": "$0.0500",
    "memory_video_5_screenshots_8sec": "$2.4800"
  }
}
```

## Cost Estimates

### Typical Use Cases

| Operation | Estimated Cost |
|---|---|
| 5-minute call (audio only) | $0.94 |
| 10-minute call (audio only) | $1.88 |
| 5-minute call + 3 vision triggers | $1.04 |
| Interactive Storybook (5 screenshots) | $0.05 |
| Memory Video (5 screenshots, 8 sec) | $2.48 |
| **Full demo (5 min + all features)** | **$4.47** |

### Budget Planning

With your thresholds, here's a suggested usage plan:

- **$0-$3**: Initial testing, UI navigation, short audio clips
- **$3-$5**: 1-2 short test calls (2-3 minutes each)
- **$5-$12**: Main demo preparation (multiple 5-minute calls)
- **$12-$19**: Full feature testing (calls + storybooks + videos)
- **$19-$25**: Final demo run + backup recordings

**⚠️ CRITICAL**: Memory video generation ($2.48 each) is the most expensive operation. Test storybook generation first (only $0.05).

## Usage Tips

1. **Test incrementally**: Start with audio-only calls before enabling vision
2. **Monitor the widget**: Check the top-right corner during every call
3. **Console warnings**: Keep terminal visible to see threshold alerts
4. **Pre-record backups**: Record demo videos early to avoid last-minute stress
5. **Disable auto-capture**: Reduce screenshot frequency if costs climb too fast

## Implementation Details

- **Location**: `billing_monitor.py`
- **Integration**: Imported in `server.py`, called at every API interaction
- **Persistence**: Costs are tracked for the server's lifetime (resets on restart)
- **Thread-safe**: All tracking uses a single global `BillingMonitor` instance

## Modifying Thresholds

Edit `THRESHOLDS` in `billing_monitor.py`:

```python
THRESHOLDS = [3.0, 5.0, 9.0, 12.0, 15.0, 19.0, 21.0, 23.0, 24.9]
```

Add or remove values as needed, then restart the server.

## Troubleshooting

**Widget shows $0.0000 for a long time**
- This is normal if you're only viewing the UI without making API calls
- Audio streaming starts tracking once WebSocket connects

**Costs seem too high**
- Check `usage_log` in the terminal for detailed breakdown
- Vision API calls add up quickly (each frame = $0.0015)
- Video generation is expensive ($0.30/sec)

**Thresholds not triggering**
- Ensure server hasn't been restarted (resets counter)
- Check logs for `BILLING ALERT` messages
- Verify `log_api_call()` is being invoked correctly

---

**Need help?** Check server logs for detailed cost breakdowns after each session.
