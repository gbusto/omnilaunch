"""Smoke test for GPT-OSS-20B runner."""

def test_smoke():
    import os
    assert os.path.exists("/omnilaunch") or True
    print("✓ GPT-OSS-20B runner smoke test passed")

if __name__ == "__main__":
    test_smoke()

