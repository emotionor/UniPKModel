import torch
import torch.nn.functional as F
import os
import pandas as pd

# 定义损失配置字典，用于不同类型损失函数
LOSS_CONFIGS = {
    'log_mae': {'error_type': 'mae', 'log_space': True},
    'log_mse': {'error_type': 'mse', 'log_space': True},
    'log_mae_time_exp_decay': {'error_type': 'mae', 'log_space': True, 'decay_type': 'exp'},
    'log_mse_time_exp_decay': {'error_type': 'mse', 'log_space': True, 'decay_type': 'exp'},
    'log_mae_time_linear_decay': {'error_type': 'mae', 'log_space': True, 'decay_type': 'linear'},
    'log_mse_time_linear_decay': {'error_type': 'mse', 'log_space': True, 'decay_type': 'linear'},
    'log_mae_time_cos_decay': {'error_type': 'mae', 'log_space': True, 'decay_type': 'cos'},
    'log_mse_time_cos_decay': {'error_type': 'mse', 'log_space': True, 'decay_type': 'cos'},
}

def get_loss_fn(loss_fn_name):
    """根据损失函数名称获取对应的损失函数"""
    if loss_fn_name in ['mixed_pk_loss', 'mixed_pk_tail_loss']:
        # 这些是特殊的复合损失函数，直接返回
        return globals()[loss_fn_name]
    
    # 从配置字典获取基本损失函数配置
    config = LOSS_CONFIGS.get(loss_fn_name)
    if config is None:
        return compute_basic_loss  # 默认返回基本损失函数
    
    # 返回适配特定配置的损失函数
    return lambda y_preds, y_true, times, **kwargs: compute_basic_loss(
        y_preds, y_true, times, **{**config, **kwargs}
    )

def compute_basic_loss(y_preds, y_true, times, error_type='mae', log_space=True,
                       normalize=False, decay_type=None, alpha=1.0, eps=1e-5, **kwargs):
    """统一的基础损失计算函数"""
    device = y_preds.device
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    
    # 处理数据和掩码
    y_preds_masked = torch.where(mask, y_preds, torch.tensor(eps, device=device))
    y_true_masked = torch.where(mask, y_true, torch.tensor(eps, device=device))
    
    # 处理时间衰减
    time_decay = torch.ones_like(y_preds_masked)
    if decay_type is not None:
        norm_times = times / times.max() * alpha
        norm_times = norm_times.broadcast_to(y_preds.shape)
        
        if decay_type == 'exp':
            time_decay = torch.exp(-norm_times)
        elif decay_type == 'linear':
            time_decay = 1 - norm_times
        elif decay_type == 'cos':
            time_decay = torch.cos(norm_times * 3.14159 / 2)
        
    time_decay = torch.where(mask, time_decay, torch.tensor(0.0, device=device)) #
    
    # 计算误差
    if log_space:
        if error_type == 'mae':
            # error = torch.abs(torch.log(y_preds_masked + eps) - torch.log(y_true_masked + eps))
            error = F.l1_loss(torch.log(y_preds_masked + eps), torch.log(y_true_masked + eps), reduction='none')
        else:  # mse
            # error = (torch.log(y_preds_masked + eps) - torch.log(y_true_masked + eps)) ** 2
            error = F.mse_loss(torch.log(y_preds_masked + eps), torch.log(y_true_masked + eps), reduction='none')
    else:
        if error_type == 'mae':
            # error = torch.abs(y_preds_masked - y_true_masked)
            error = F.l1_loss(y_preds_masked, y_true_masked, reduction='none')
        else:  # mse
            # error = (y_preds_masked - y_true_masked) ** 2
            error = F.mse_loss(y_preds_masked, y_true_masked, reduction='none')
    
    # 应用时间衰减
    weighted_error = error * time_decay
    
    # 计算每个样本的平均误差
    mean_error = torch.sum(weighted_error, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1)
    
    # 计算最终损失
    loss = torch.mean(mean_error[torch.isfinite(mean_error)])
    return loss

def cal_all_losses(y_preds, y_true, **kwargs):
    """计算并返回多种损失指标"""
    eps = 1e-5
    device = y_preds.device
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds_masked = torch.where(mask, y_preds, torch.tensor(eps, device=device))
    y_true_masked = torch.where(mask, y_true, torch.tensor(eps, device=device))
    
    # 计算各种误差
    log_mae = torch.abs(torch.log(y_preds_masked + eps) - torch.log(y_true_masked + eps))
    log_mse = (torch.log(y_preds_masked + eps) - torch.log(y_true_masked + eps)) ** 2
    mae = torch.abs(y_preds_masked - y_true_masked)
    mse = (y_preds_masked - y_true_masked) ** 2
    
    # 计算每个样本的平均误差
    mean_log_mae = torch.sum(log_mae, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1)
    mean_log_mse = torch.sum(log_mse, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1)
    mean_mae = torch.sum(mae, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1)
    mean_mse = torch.sum(mse, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1)
    
    # 计算R2分数
    r2_scores = []
    for i in range(mask.shape[0]):
        sample_mask = mask[i]
        if torch.sum(sample_mask) > 1:
            r2 = torch_r2(y_preds_masked[i, sample_mask], y_true_masked[i, sample_mask])
            r2_scores.append(r2)
    
    if len(r2_scores) > 0:
        r2_scores = torch.stack(r2_scores)
        r2_scores.clip_(min=0, max=1)
        r2_nonzero_radio = torch.sum(r2_scores > 0) / r2_scores.shape[0]
        top30_size = max(1, int(0.3 * r2_scores.shape[0]))
        top60_size = max(1, int(0.6 * r2_scores.shape[0]))
        top30_r2 = torch.topk(r2_scores, top30_size, largest=True).values
        top60_r2 = torch.topk(r2_scores, top60_size, largest=True).values
        r2_score = torch.mean(r2_scores)
    else:
        r2_nonzero_radio = torch.tensor(0.0, device=device)
        top30_r2 = torch.tensor(0.0, device=device)
        top60_r2 = torch.tensor(0.0, device=device)
        r2_score = torch.tensor(0.0, device=device)
    
    # 计算最终损失
    loss_log_mae = torch.mean(mean_log_mae[torch.isfinite(mean_log_mae)])
    loss_log_mse = torch.mean(mean_log_mse[torch.isfinite(mean_log_mse)])
    loss_mae = torch.mean(mean_mae[torch.isfinite(mean_mae)])
    loss_mse = torch.mean(mean_mse[torch.isfinite(mean_mse)])
    loss_rmse = torch.sqrt(loss_mse)
    
    # 保存结果
    if kwargs.get('save_path', False):
        save_path = kwargs['save_path']
        df = pd.DataFrame({
            'mean_log_mae': mean_log_mae.cpu().numpy(),
            'mean_log_mse': mean_log_mse.cpu().numpy(),
            'r2_scores': r2_scores.cpu().numpy() if len(r2_scores) > 0 else [],
        })
        df.to_csv(os.path.join(save_path, 'losses.csv'), index=False)

    return {
        'log_mae': loss_log_mae.item(),
        'log_mse': loss_log_mse.item(),
        'mae': loss_mae.item(),
        'rmse': loss_rmse.item(),
        'r2_score': r2_score.item(),
        'r2_nonzero_radio': r2_nonzero_radio.item(),
        'top30_r2': top30_r2.mean().item() if len(r2_scores) > 0 else 0.0,
        'top60_r2': top60_r2.mean().item() if len(r2_scores) > 0 else 0.0
    }

def torch_r2(y_preds, y_true):
    """计算R^2分数"""
    mean_y = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean_y)**2)
    ss_res = torch.sum((y_true - y_preds)**2)
    return 1 - ss_res / (ss_tot + 1e-8)  # 避免除零

def average_r2_score(y_preds, y_true, **kwargs):
    """计算平均R^2分数"""
    eps = 1e-5
    device = y_preds.device
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds_masked = torch.where(mask, y_preds, torch.tensor(eps, device=device))
    y_true_masked = torch.where(mask, y_true, torch.tensor(eps, device=device))
    
    r2_scores = []
    for i in range(mask.shape[0]):
        sample_mask = mask[i]
        if torch.sum(sample_mask) > 1:
            r2 = torch_r2(y_preds_masked[i, sample_mask], y_true_masked[i, sample_mask])
            if torch.isfinite(r2):
                r2_scores.append(r2)
    
    if len(r2_scores) == 0:
        return torch.tensor(0.0, device=device)
    
    return torch.mean(torch.stack(r2_scores))

def torch_mfce(y_preds, y_true):
    """计算MFCE (Median Fold Change Error)"""
    eps = 1e-5
    mfce = torch.exp(torch.median(torch.abs(torch.log(y_preds + eps) - torch.log(y_true + eps))))
    return torch.clamp(mfce, max=1000)

def average_mfce(y_preds, y_true, **kwargs):
    """计算平均MFCE"""
    eps = 1e-5
    device = y_preds.device
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds_masked = torch.where(mask, y_preds, torch.tensor(eps, device=device))
    y_true_masked = torch.where(mask, y_true, torch.tensor(eps, device=device))
    
    mfce_scores = []
    for i in range(mask.shape[0]):
        sample_mask = mask[i]
        if torch.sum(sample_mask) > 0:
            mfce = torch_mfce(y_preds_masked[i, sample_mask], y_true_masked[i, sample_mask])
            if torch.isfinite(mfce):
                mfce_scores.append(mfce)
    
    if len(mfce_scores) == 0:
        return torch.tensor(0.0, device=device)
    
    return torch.mean(torch.stack(mfce_scores))

def mixed_pk_loss(y_preds, y_true, times, alpha=1.0, lambda_peak=0.5, lambda_auc=0.1, eps=1e-5, **kwargs):
    """混合PK损失：结合log-MAE、峰值损失和AUC损失"""
    device = y_preds.device
    
    # 获取基本损失（使用统一配置）
    loss_main = compute_basic_loss(
        y_preds, y_true, times, 
        error_type='mae', 
        log_space=True, 
        decay_type='exp', 
        alpha=alpha,
        eps=eps
    )
    
    # 处理数据和掩码
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds_masked = torch.where(mask, y_preds, torch.tensor(eps, device=device))
    y_true_masked = torch.where(mask, y_true, torch.tensor(eps, device=device))
    
    # 归一化时间
    norm_times = times / times.max() * alpha
    norm_times = norm_times.broadcast_to(y_preds.shape)
    
    # Cmax损失
    cmax_pred = torch.max(y_preds_masked, dim=1).values
    cmax_true = torch.max(y_true_masked, dim=1).values
    loss_peak = F.l1_loss(cmax_pred, cmax_true)
    
    # AUC损失
    auc_pred = torch.trapz(y_preds_masked, norm_times, dim=1)
    auc_true = torch.trapz(y_true_masked, norm_times, dim=1)
    loss_auc = F.l1_loss(auc_pred, auc_true)
    
    # 总损失
    loss = loss_main + lambda_peak * loss_peak + lambda_auc * loss_auc
    
    return loss, {
        "loss_main": loss_main.item(),
        "loss_peak": loss_peak.item(),
        "loss_auc": loss_auc.item(),
    }

def compute_tail_slope_penalty(y_preds, times, eps):
    """计算尾部斜率惩罚，用于改善长尾预测"""
    c1 = torch.clamp(y_preds[:, -2], min=eps)
    c2 = torch.clamp(y_preds[:, -1], min=eps)
    t1 = times[:, -2]
    t2 = times[:, -1]
    dt = torch.clamp(t2 - t1, min=eps)
    
    slope = (torch.log(c2) - torch.log(c1)) / dt  # log-slope
    penalty_tail = torch.clamp(-slope, min=0.0) ** 2  # 仅惩罚快速下降
    loss_tail = torch.mean(penalty_tail)
    
    return loss_tail

def mixed_pk_tail_loss(y_preds, y_true, times, alpha=1.0, lambda_peak=0.5, 
                       lambda_auc=0.1, lambda_log=0.0, lambda_tail=0.0, eps=1e-5, **kwargs):
    """带尾部惩罚的混合PK损失"""
    device = y_preds.device
    
    # 获取基本损失
    loss_main = compute_basic_loss(
        y_preds, y_true, times, 
        error_type='mae', 
        log_space=True, 
        decay_type='exp', 
        alpha=alpha,
        eps=eps
    )
    
    # 处理数据和掩码
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds_masked = torch.where(mask, y_preds, torch.tensor(eps, device=device))
    y_true_masked = torch.where(mask, y_true, torch.tensor(eps, device=device))
    
    # === Cmax (peak) 损失 ===
    peak_pred = torch.amax(y_preds_masked, dim=1)
    peak_true = torch.amax(y_true_masked, dim=1)
    loss_peak = F.l1_loss(peak_pred, peak_true)
    
    # === AUC 损失 ===
    auc_pred = torch.trapz(y_preds_masked, x=times, dim=1)
    auc_true = torch.trapz(y_true_masked, x=times, dim=1)
    loss_auc = F.l1_loss(auc_pred, auc_true)
    
    # === 对数空间全局损失 ===
    loss_log = F.l1_loss(torch.log(y_preds_masked + eps), torch.log(y_true_masked + eps))
    
    # === 尾部斜率惩罚 ===
    loss_tail = compute_tail_slope_penalty(y_preds_masked, times, eps)
    
    # 总损失
    loss = (
        loss_main +
        lambda_peak * loss_peak +
        lambda_auc * loss_auc +
        lambda_log * loss_log +
        lambda_tail * loss_tail
    )
    
    return loss, {
        "loss": loss.item(),
        "loss_main": loss_main.item(),
        "loss_peak": loss_peak.item(),
        "loss_auc": loss_auc.item(),
        "loss_log": loss_log.item(),
        "loss_tail": loss_tail.item()
    }
