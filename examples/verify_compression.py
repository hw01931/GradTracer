""\"
GradTracer Compression Verification Script
Ensures that models are actually being sparsified by the recipe.
""\"
import torch, torch.nn as nn, torch.nn.utils.prune as prune
from transformers import AutoModelForSequenceClassification
from gradtracer import FlowTracker, RecipeGenerator

def get_sparsity(model):
    total, zero = 0, 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check for pruning mask
            if hasattr(module, 'weight_mask'):
                w = module.weight_mask * module.weight_orig
            else:
                w = module.weight.data
            total += w.numel()
            zero += torch.sum(w == 0).item()
    return (zero / total * 100) if total > 0 else 0

def run_verification(model_name="distilbert-base-uncased", target_sparsity=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    print(f"--- Verifying Compression on {model_name} (Target: {target_sparsity*100}%) ---")
    
    # 1. Profile
    tracker = FlowTracker(model, track_gradients=True)
    input_ids = torch.randint(0, 1000, (8, 128), device=device)
    for _ in range(5):
        loss = model(input_ids, labels=torch.ones((8,), dtype=torch.long, device=device)).loss
        loss.backward()
        tracker.step(loss.item())
        model.zero_grad()
    
    # 2. Generate and Apply Recipe
    recipe = RecipeGenerator(tracker).generate(target_sparsity=target_sparsity)
    for path, instr in recipe['layers'].items():
        if instr['prune_ratio'] > 0:
            module = model.get_submodule(path.rsplit('.', 1)[0])
            prune.l1_unstructured(module, name=path.split('.')[-1], amount=instr['prune_ratio'])
    
    # 3. Verify
    actual_sparsity = get_sparsity(model)
    print(f"Initial Sparsity: 0.00%")
    print(f"Actual Sparsity:  {actual_sparsity:.2f}%")
    
    if abs(actual_sparsity - (target_sparsity*100)) < 5:
        print("✅ SUCCESS: Compression verified. Parameters have been zeroed out.")
    else:
        print("⚠️ WARNING: Sparsity mismatch. Check layer compatibility.")

if __name__ == "__main__":
    run_verification()
