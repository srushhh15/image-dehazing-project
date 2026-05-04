# =====================================================
# CONFIGURATION FILE - 25 EPOCHS FOR FULL DATASET
# =====================================================

# Dataset Configuration
DATASET_CONFIG = {
    "use_full_dataset": True,  # ← Use ALL images
    "subset_size": 50,         # ← Not used if use_full_dataset = True
}

# Training Configuration  
TRAINING_CONFIG = {
    "epochs": 25,
    "batch_size": 16,
}

# Print Config
def print_config():
    print("\n" + "="*70)
    print("📋 CONFIGURATION - 200 EPOCHS MODE")
    print("="*70)
    print(f"✅ Use full dataset: {DATASET_CONFIG['use_full_dataset']}")
    print(f"✅ Total epochs: {TRAINING_CONFIG['epochs']}")
    print(f"✅ Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"⏱️  Expected training time: 4-6 hours (GPU T4/V100)")
    print("="*70 + "\n")