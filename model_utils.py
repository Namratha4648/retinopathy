import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

try:
    import timm
except Exception:
    timm = None

try:
    from efficientnet_pytorch import EfficientNet
except Exception:
    EfficientNet = None


def load_model(model_path: str, device_str: str = "cpu", backend: str = "timm"):
    """Load model from checkpoint. backend: 'timm' or 'efficientnet_pytorch'.

    Returns (model, device)
    """
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if backend == "timm":
        if timm is None:
            raise RuntimeError("timm is not installed. Install timm or choose efficientnet_pytorch backend.")
        model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=5)
        model.load_state_dict(checkpoint, strict=False)

    elif backend == "efficientnet_pytorch":
        if EfficientNet is None:
            raise RuntimeError("efficientnet_pytorch not installed. Install efficientnet-pytorch or choose timm backend.")
        # EfficientNet-PyTorch uses names like 'efficientnet-b2'
        model = EfficientNet.from_name('efficientnet-b2', num_classes=5)
        # checkpoint may be either state_dict or full model state
        if isinstance(checkpoint, dict) and ('state_dict' in checkpoint):
            sd = checkpoint['state_dict']
        else:
            sd = checkpoint
        model.load_state_dict(sd, strict=False)

    else:
        raise ValueError(f"Unknown backend: {backend}")

    model.to(device)
    model.eval()
    return model, device


def _get_target_module(model, target_layer: str):
    # Map target_layer string to actual module on the model
    if target_layer == 'conv_head':
        if hasattr(model, 'conv_head'):
            return model.conv_head
        if hasattr(model, '_conv_head'):
            return model._conv_head
    # fallback to last block
    if hasattr(model, 'blocks'):
        return model.blocks[-1]
    if hasattr(model, '_blocks'):
        return model._blocks[-1]
    # last resort: return model
    return model


def _register_hooks_and_forward(model, input_tensor, target_layer):
    activations = {}
    gradients = {}

    def save_activation(module, inp, out):
        activations['value'] = out.detach()

    def save_gradient(module, grad_in, grad_out):
        # grad_out can be a tuple
        gradients['value'] = grad_out[0].detach()

    layer = _get_target_module(model, target_layer)

    fh = layer.register_forward_hook(save_activation)
    # use full backward hook for modern PyTorch
    try:
        bh = layer.register_full_backward_hook(save_gradient)
    except Exception:
        bh = layer.register_backward_hook(save_gradient)

    outputs = model(input_tensor)
    return outputs, activations, gradients, fh, bh


def predict_and_explain(pil_image: Image.Image, model, device, target_layer='last_block'):
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = pil_image.convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    outputs, activations, gradients, fh, bh = _register_hooks_and_forward(model, input_tensor, target_layer)

    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred = int(outputs.argmax(dim=1).item())

    # Backprop to get gradients
    model.zero_grad()
    one_hot = torch.zeros_like(outputs)
    one_hot[0, pred] = 1
    outputs.backward(gradient=one_hot)

    grads = gradients['value']  # [N, C, H, W]
    act = activations['value']   # [N, C, H, W]
    pooled = torch.mean(grads, dim=(2, 3))  # [N, C]

    act = act.squeeze(0).cpu().numpy()
    pooled = pooled.squeeze(0).cpu().numpy()

    for i in range(act.shape[0]):
        act[i, :, :] *= pooled[i]

    heatmap = np.mean(act, axis=0)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    img_np = np.array(img)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay_bgr = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    overlay_pil = Image.fromarray(overlay_rgb)
    heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))

    fh.remove()
    bh.remove()

    return pred, probs, heatmap_pil, overlay_pil

