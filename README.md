# siglip-pytorch

```python
import torch
from siglip import SiglipModel, VisionConfig, TextConfig

if __name__ == '__main__':
    vision_config = VisionConfig()
    text_config = TextConfig()

    model = SiglipModel(vision_config, text_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'params = {total_params:,}')
    # (batch, seq_len)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    pixel_values = torch.randn(2, 3, 224, 224)

    outputs = model(input_ids=input_ids, pixel_values=pixel_values, return_loss=True)
    logits_per_text = outputs[0]
    logits_per_image = outputs[1]
    text_embeds = outputs[2]
    image_embeds = outputs[3]
    loss = outputs[4]

    print(
        logits_per_text.shape,
        logits_per_image.shape,
        text_embeds.shape,
        image_embeds.shape,
        loss.shape
    )

    print(loss)
```
