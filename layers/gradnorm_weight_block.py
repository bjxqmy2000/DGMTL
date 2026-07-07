import torch
import torch.nn as nn

class GradNormWeighting(nn.Module):
    def __init__(self, num_tasks: int, alpha: float = 1.5, init_weights=None):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        
        if init_weights is None:
            init_weights = torch.ones(num_tasks, dtype=torch.float32)
        else:
            init_weights = torch.tensor(init_weights, dtype=torch.float32)
        
        self.task_weights = nn.Parameter(init_weights)
        self.register_buffer('initial_losses', torch.zeros(num_tasks))
        self.register_buffer('has_initial_losses', torch.tensor(False))
    
    def forward(self, task_losses):
        """只负责前向计算加权Loss"""
        if not torch.is_tensor(task_losses):
            losses = torch.stack(task_losses)
        else:
            losses = task_losses
        
        if not self.has_initial_losses:
            self.initial_losses.copy_(losses.detach())
            self.has_initial_losses.fill_(True)
        
        weighted_losses = self.task_weights * losses
        return weighted_losses

    def compute_gradnorm_loss(self, weighted_losses, task_losses, shared_params):
        """核心：通过Autograd计算出GradNorm的L_grad"""
        losses = torch.stack(task_losses) if not torch.is_tensor(task_losses) else task_losses
        
        grad_norms = []
        for w_loss in weighted_losses:
            # retain_graph=True, create_graph=True 开启高阶求导，允许梯度通过范数流回 weights
            grads = torch.autograd.grad(w_loss, shared_params, retain_graph=True, create_graph=True)
            # 计算L2范数
            grad_norm = torch.sqrt(sum([torch.sum(g ** 2) for g in grads]) + 1e-8)
            grad_norms.append(grad_norm)
            
        grad_norms = torch.stack(grad_norms)
        
        # 相对收敛速度 r_i(t)
        loss_ratios = losses.detach() / (self.initial_losses + 1e-8)
        mean_loss_ratio = loss_ratios.mean()
        relative_inverse_train_rates = loss_ratios / (mean_loss_ratio + 1e-8)
        
        # 目标梯度范数 Z_i(t)
        mean_grad_norm = grad_norms.mean().detach()
        target_grad_norms = mean_grad_norm * (relative_inverse_train_rates ** self.alpha)
        
        # GradNorm 损失函数：L_grad = sum(|G_i - Z_i|)
        loss_gradnorm = torch.sum(torch.abs(grad_norms - target_grad_norms.detach()))
        return loss_gradnorm

    def renormalize_weights(self):
        """更新完权重后，手动重归一化"""
        with torch.no_grad():
            # 使用原地 clamp_ 和 copy_ 操作
            self.task_weights.clamp_(min=0.001)
            normalize_coeff = self.num_tasks / self.task_weights.sum()
            self.task_weights.copy_(self.task_weights * normalize_coeff)

    def get_weights(self):
        with torch.no_grad(): return self.task_weights.clone()
        
    def get_loss_ratios(self, current_losses):
        losses = torch.stack(current_losses) if not torch.is_tensor(current_losses) else current_losses
        with torch.no_grad(): return losses / (self.initial_losses + 1e-8)