"""
emergency_auth.py - Tier 3: HMAC-SHA256 Signed Emergency Vehicle Authentication
KPIT Sparkle 2027 - Team InnovateX - RMK Engineering College

Every emergency-vehicle priority request is cryptographically signed.
A junction runs three checks before acting on a priority claim:
  1. Vehicle type is on the whitelist
  2. Message is not older than max_age_seconds (replay-attack defence)
  3. HMAC-SHA256 signature matches
"""

import hmac
import hashlib
import time

# Shared secret for demo - in production this would be per-junction asymmetric key
SECRET_KEY = b'kpit-sparkle-2027-swarm-traffic'

EMERGENCY_WHITELIST = [
    'ambulance', 'fire_truck', 'fire', 'police',
    'rescue', 'paramedic', 'emergency_doctor', 'police_swat', 'fire_chief'
]


def sign_emergency(vehicle_id: str, junction_id: str, timestamp: float) -> str:
    """Return HMAC-SHA256 hex signature for an emergency broadcast message."""
    message = f"{vehicle_id}:{junction_id}:{timestamp:.0f}".encode('utf-8')
    return hmac.new(SECRET_KEY, message, hashlib.sha256).hexdigest()


def verify_emergency(
    vehicle_id: str,
    junction_id: str,
    timestamp: float,
    signature: str,
    max_age_seconds: int = 30
) -> bool:
    """
    Return True only if ALL three security checks pass:
      1. vehicle_id contains a whitelisted emergency type
      2. message is not older than max_age_seconds
      3. HMAC signature matches (timing-safe comparison)
    """
    # Check 1 - whitelist
    vid_lower = vehicle_id.lower()
    if not any(w in vid_lower for w in EMERGENCY_WHITELIST):
        return False

    # Check 2 - freshness (replay-attack defence)
    if abs(time.time() - timestamp) > max_age_seconds:
        return False

    # Check 3 - HMAC integrity
    expected = sign_emergency(vehicle_id, junction_id, timestamp)
    return hmac.compare_digest(expected, signature)


def build_emergency_payload(vehicle_id: str, junction_id: str) -> dict:
    """Build a signed, JSON-serialisable emergency payload dict."""
    ts = time.time()
    return {
        'vehicle_id': vehicle_id,
        'junction_id': junction_id,
        'timestamp': ts,
        'signature': sign_emergency(vehicle_id, junction_id, ts),
    }


def validate_emergency_payload(payload: dict) -> bool:
    """Extract fields from a received payload dict and verify."""
    try:
        return verify_emergency(
            vehicle_id=payload['vehicle_id'],
            junction_id=payload['junction_id'],
            timestamp=float(payload['timestamp']),
            signature=payload['signature'],
        )
    except (KeyError, TypeError, ValueError):
        return False


if __name__ == '__main__':
    print("=" * 50)
    print("Emergency Auth -- Self-Test")
    print("=" * 50)
    results = []

    # Test 1 - valid ambulance payload should pass
    p = build_emergency_payload('ambulance_1500', 'J1')
    r1 = validate_emergency_payload(p)
    results.append(r1)
    print(f"  Valid ambulance payload:       {'PASS' if r1 else 'FAIL'}")

    # Test 2 - tampered vehicle_id should fail
    p2 = dict(p)
    p2['vehicle_id'] = 'ambulance_HACKED'
    r2 = not validate_emergency_payload(p2)
    results.append(r2)
    print(f"  Tampered vehicle_id rejected:  {'PASS' if r2 else 'FAIL'}")

    # Test 3 - non-whitelisted vehicle type should fail
    p3 = build_emergency_payload('random_car_99', 'J1')
    r3 = not validate_emergency_payload(p3)
    results.append(r3)
    print(f"  Non-whitelisted type rejected: {'PASS' if r3 else 'FAIL'}")

    # Test 4 - expired message (backdated 60s) should fail
    p4 = build_emergency_payload('ambulance_old', 'J1')
    p4['timestamp'] -= 60
    p4['signature'] = sign_emergency(p4['vehicle_id'], p4['junction_id'], p4['timestamp'])
    r4 = not validate_emergency_payload(p4)
    results.append(r4)
    print(f"  Expired message rejected:      {'PASS' if r4 else 'FAIL'}")

    print("=" * 50)
    print(f"  Overall: {'ALL TESTS PASSED' if all(results) else 'SOME TESTS FAILED'}")
    print("=" * 50)
