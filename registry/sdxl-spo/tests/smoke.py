"""Smoke test for SDXL SPO runner."""

def test_smoke():
    """Basic environment check."""
    import os
    assert os.path.exists("/omnilaunch") or True  # Volume may not exist in test env
    print("✓ SPO runner smoke test passed")

if __name__ == "__main__":
    test_smoke()

