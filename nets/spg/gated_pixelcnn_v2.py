import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10, bh_model=False):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        self.bh_model = bh_model

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, 3 if self.bh_model else 1)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, 1 if self.bh_model else 0)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, 2)
        padding_shp = (0, 1)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-2), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        if self.bh_model:
            h_horiz = self.horiz_stack(x_h)
            h_horiz = h_horiz[:, :, :, :x_h.size(-1)]
            v2h = self.vert_to_horiz(h_vert)

            out = self.gate(v2h + h_horiz + h[:, :, None, None])
            if self.residual:
                out_h = self.horiz_resid(out) + x_h
            else:
                out_h = self.horiz_resid(out)
        else:
            if self.residual:
                out_v = self.horiz_resid(out_v) + x_v
            else:
                out_v = self.horiz_resid(out_v)
            out_h = out_v

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
        super().__init__()
        self.dim = dim
        self.audio = audio
        self.bh_model = bh_model

        if self.audio:
            self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
            self.fusion_v = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)
            self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes, bh_model)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

        self.dp = nn.Dropout(0.1)

    def forward(self, x, label, aud=None):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            if i == 1 and self.audio is True:
                aud = self.embedding_aud(aud)
                a = torch.ones(aud.shape[-2]).to(aud.device)
                a = self.dp(a)
                aud = (aud.transpose(-1, -2) * a).transpose(-1, -2)
                x_v = self.fusion_v(torch.cat([x_v, aud], dim=1))
                if self.bh_model:
                    x_h = self.fusion_h(torch.cat([x_h, aud], dim=1))
            x_v, x_h = layer(x_v, x_h, label)

        if self.bh_model:
            return self.output_conv(x_h)
        else:
            return self.output_conv(x_v)

    def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )
        if pre_latents is not None:
            x = torch.cat([pre_latents, x], dim=1)
            aud_feat = torch.cat([pre_audio, aud_feat], dim=2)
            h0 = pre_latents.shape[1]
            h = h0 + shape[0]
        else:
            h0 = 0
            h = shape[0]

        for i in range(h0, h):
            for j in range(shape[1]):
                if self.audio:
                    logits = self.forward(x, label, aud_feat)
                else:
                    logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x[:, h0:h]
