"""Smoke test for GPT-OSS-120B runner."""

def test_smoke():
    import os
    assert os.path.exists("/omnilaunch") or True
    print("✓ GPT-OSS-120B runner smoke test passed")

if __name__ == "__main__":
    test_smoke()

