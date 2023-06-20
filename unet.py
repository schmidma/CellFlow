import torch
import torch.nn as nn
import torch.optim as optim
import lightning
import torch.nn.functional as F
from torchmetrics.classification import JaccardIndex as IoU


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.conv3(x)
        return F.relu(out)


class UpResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpResidualBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(lightning.LightningModule):
    def __init__(
        self, in_channels=1, out_channels=3, init_features=32, learning_rate=1e-3
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.learning_rate = learning_rate
        # Encoder
        self.encoder1 = ResidualBlock(in_channels, init_features)
        self.encoder2 = ResidualBlock(init_features, init_features * 2)
        self.encoder3 = ResidualBlock(init_features * 2, init_features * 4)
        self.encoder4 = ResidualBlock(init_features * 4, init_features * 8)

        # Decoder
        self.decoder4 = UpResidualBlock(init_features * 8, init_features * 4)
        self.decoder3 = UpResidualBlock(init_features * 4, init_features * 2)
        self.decoder2 = UpResidualBlock(init_features * 2, init_features)
        self.decoder1 = nn.Conv2d(init_features, out_channels, kernel_size=1)

        # Metrics
        self.iou = IoU(task="multiclass", num_classes=3)

        # Loss
        self.binary_cross_entropy_loss = nn.BCELoss()
        self.mean_squared_error = nn.MSELoss()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2, stride=2))
        dec4 = self.decoder4(enc4, enc3)
        dec3 = self.decoder3(dec4, enc2)
        dec2 = self.decoder2(dec3, enc1)
        out = self.decoder1(dec2)
        return out

    def _shared_evaluation_step(self, batch):
        images, target_masks, target_flow_gradients = batch
        assert target_flow_gradients.shape[1] == 2
        assert target_masks.shape == images.shape
        outputs = self(images)
        logits = outputs[:, 0, ...]
        predicted_flow_gradients = outputs[:, 1:, ...]
        predicted_object_probabilities = F.sigmoid(logits)
        classification_loss = self.binary_cross_entropy_loss(
            predicted_object_probabilities, target_masks.type(torch.float32)
        )
        flow_gradient_loss = self.mean_squared_error(
            torch.masked_select(target_flow_gradients, target_masks.unsqueeze(1)),
            torch.masked_select(predicted_flow_gradients, target_masks.unsqueeze(1)),
        )
        total_loss = classification_loss + flow_gradient_loss
        return total_loss, predicted_object_probabilities

    def training_step(self, batch, batch_idx):
        total_loss, _ = self._shared_evaluation_step(batch)
        self.log("train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        _, target_masks, _ = batch
        total_loss, predicted_object_probabilities = self._shared_evaluation_step(batch)
        iou = self.iou(predicted_object_probabilities > 0.5, target_masks)
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    images = torch.zeros((1, 1, 960, 768))
    masks = torch.zeros((1, 960, 768), dtype=torch.bool)
    masks[0, 0, 0] = True
    flow_gradients = torch.zeros((1, 2, 960, 768))
    batch = (images, masks, flow_gradients)
    unet = UNet()

    print(unet.training_step(batch, 0))
    print(unet.validation_step(batch, 0))
