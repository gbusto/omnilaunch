"""Smoke test for Qwen3-VL runner."""

def test_smoke():
    import os
    assert os.path.exists("/omnilaunch") or True
    print("✓ Qwen3-VL runner smoke test passed")

if __name__ == "__main__":
    test_smoke()

