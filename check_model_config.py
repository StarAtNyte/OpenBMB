"""
Check MiniCPM-SALA model config for hybrid architecture support.
"""
import json
from pathlib import Path


def check_model_config(model_path: str = "/models/MiniCPM-SALA"):
    """Check if model has hybrid_lightning_config support."""
    config_path = Path(model_path) / "config.json"

    print(f"Reading config from: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    print("\n=== Model Config ===")
    print(f"Model Type: {config.get('model_type', 'N/A')}")
    print(f"Architecture: {config.get('architectures', ['N/A'])[0]}")

    # Check for hybrid architecture indicators
    print("\n=== Hybrid Architecture Check ===")

    # Check if it's BailingHybridConfig
    if config.get('model_type') == 'bailing_hybrid':
        print("✅ Model type is 'bailing_hybrid' - HYBRID BACKEND SUPPORTED!")
        print(f"   Num layers: {config.get('num_hidden_layers', 'N/A')}")
        print(f"   Layer group size: {config.get('layer_group_size', 'N/A')}")
        if 'layers_block_type' in config:
            print(f"   Layer types: {config['layers_block_type']}")
        return True

    # Check for other hybrid indicators
    hybrid_indicators = [
        'hybrid_config',
        'linear_attention_layers',
        'full_attention_layers',
        'mambaish_config',
        'hybrid_lightning_config',
        'layer_group_size',
    ]

    found = []
    for key in hybrid_indicators:
        if key in config:
            found.append(key)
            print(f"✅ Found: {key} = {config[key]}")

    if not found:
        print("❌ No hybrid architecture indicators found")
        print("   Model will use standard attention backend")
        return False

    print(f"\n✅ Found {len(found)} hybrid indicators - backend MAY support hybrid")
    return True


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/models/MiniCPM-SALA"
    check_model_config(model_path)
