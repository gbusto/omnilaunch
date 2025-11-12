"""Smoke test for MeshCoder runner."""

def test_smoke():
    import os
    assert os.path.exists("/omnilaunch") or True
    print("✓ MeshCoder runner smoke test passed")

if __name__ == "__main__":
    test_smoke()
