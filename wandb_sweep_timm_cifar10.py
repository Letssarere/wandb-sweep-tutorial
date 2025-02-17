import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from torchvision import datasets, transforms
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import os

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            los_pos *= one_sided_w
            los_neg *= one_sided_w

        return -(los_pos + los_neg).sum()

class FocalLoss(nn.Module):
    """Focal Loss for dealing with class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class ModelManager:
    """TIMM 모델 및 학습 관리"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path("model_checkpoints")
        
        if not self.checkpoint_dir.is_dir():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        
    @staticmethod
    def get_sweep_config() -> Dict[str, Any]:
        """WandB sweep 설정 반환"""
        return {
            'method': 'random',
            'metric': {
                'name': 'val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'model_name': {
                    # CIFAR-10에 적합한 작은/중간 크기 모델들 선택
                    'values': [
                        'resnet18',
                        'mobilenetv3_small_100',
                        'efficientnet_b0',
                        'vit_tiny_patch16_224',
                        'convnext_tiny',
                        'mobilevitv2_050'
                    ]
                },
                'optimizer': {
                    'values': ['adam', 'sgd', 'adamw']
                },
                'loss_function': {
                    'values': ['cross_entropy', 'focal']
                },
                'dropout': {
                    'values': [0.3, 0.4, 0.5]
                },
                'learning_rate': {
                    'distribution': 'log_uniform',
                    'min': -9.21,  # log(1e-4)
                    'max': -4.61   # log(1e-2)
                },
                'batch_size': {
                    'distribution': 'q_log_uniform_values',
                    'q': 8,
                    'min': 32,
                    'max': 256,
                },
                'epochs': {
                    'value': 5
                }
            }
        }
    
    def get_model(self, model_name: str, num_classes: int) -> nn.Module:
        """TIMM 모델 로드 및 CIFAR-10 크기에 맞게 조정"""
        if 'vit_tiny_patch16_224' in model_name:
            # ViT 모델의 경우에만 patch_size 적용
            model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes,
                img_size=32,
                patch_size=4  # 32x32 이미지에 맞게 patch size 조정
            )
        else:
            # 다른 모델들의 경우
            model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes,
                in_chans=3
            )
                
        return model.to(self.device)
    
    def get_loss_function(self, loss_name: str) -> nn.Module:
        """Loss function 생성"""
        if loss_name == 'focal':
            return FocalLoss()
        # elif loss_name == 'asymmetric':
        #     return AsymmetricLoss()
        return nn.CrossEntropyLoss()

    def get_optimizer(self, model: nn.Module, optimizer_name: str, learning_rate: float) -> torch.optim.Optimizer:
        """옵티마이저 생성"""
        if optimizer_name == "sgd":
            return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == "adamw":
            return optim.AdamW(model.parameters(), lr=learning_rate)
        return optim.Adam(model.parameters(), lr=learning_rate)
    
    def get_data_loaders(self, batch_size: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """CIFAR-10 데이터 로더 생성"""
        # CIFAR-10에 최적화된 데이터 증강
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        
        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=train_transform
        )
        
        val_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=test_transform
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def save_best_model(self, model: nn.Module, config: wandb.Config, val_loss: float):
        """최고 성능 모델 저장"""
        checkpoint_name = f"{config.model_name}_{config.optimizer}_{config.loss_function}_loss{val_loss:.4f}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # 같은 조합의 이전 체크포인트 삭제
        for old_checkpoint in self.checkpoint_dir.glob(f"{config.model_name}_{config.optimizer}_{config.loss_function}_*.pth"):
            os.remove(old_checkpoint)

        # 항상 저장
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': dict(config),
            'val_loss': val_loss
        }, checkpoint_path)

    
    def train_epoch(self, model: nn.Module, loader: torch.utils.data.DataLoader,
                   criterion: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """한 에폭 학습 수행"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # 정확도 계산
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            loss.backward()
            optimizer.step()
            
            wandb.log({
                "batch_loss": loss.item(),
                "batch_accuracy": 100. * correct / total
            })
            
        epoch_loss = total_loss / len(loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, model: nn.Module, loader: torch.utils.data.DataLoader,
                criterion: nn.Module) -> Tuple[float, float]:
        """검증 수행"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # 정확도 계산
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss = total_loss / len(loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
    
    def train(self, config: wandb.Config = None):
        """전체 학습 프로세스 실행"""
        with wandb.init(config=config):
            config = wandb.config
            
            # 모델, 데이터, 손실함수, 옵티마이저 초기화
            model = self.get_model(config.model_name, num_classes=10)
            train_loader, val_loader = self.get_data_loaders(config.batch_size)
            criterion = self.get_loss_function(config.loss_function)
            optimizer = self.get_optimizer(model, config.optimizer, config.learning_rate)
            
            best_val_loss = float('inf')
            
            # wandb에 클래스 이름 기록
            wandb.config.update({"classes": self.classes})
            
            # 학습 수행
            for epoch in range(config.epochs):
                train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
                val_loss, val_acc = self.validate(model, val_loader, criterion)
                
                wandb.log({
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "epoch": epoch
                })
                
                # 최고 성능 모델 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_best_model(model, config, val_loss)
                    # 모델 저장했다는 메시지 출력
                    print(f"Model saved with loss: {val_loss:.4f}")
                else:
                    print(f"Model not saved, best loss: {best_val_loss:.4f}")
                    
                    
                print(f'Epoch: {epoch+1}/{config.epochs}')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print('-' * 50)

def main():
    """메인 실행 함수"""
    wandb.login(key="insert_you_wandb_api_key_here")
    
    # 모델 매니저 초기화
    manager = ModelManager()
    
    # Sweep 설정 및 실행
    sweep_config = manager.get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="timm-cifar10-sweeps-5")
    
    # Sweep Agent 실행
    wandb.agent(sweep_id, function=manager.train, count=30)

if __name__ == "__main__":
    main()