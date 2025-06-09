import torch
import torch.nn.functional as F

def get_loss_fn(loss_fn):
    loss_fns = {
        'log_mae': log_mae_loss,
        'log_mse': log_mse_loss,
        'log_mae_time_exp_decay': log_mae_time_exp_decay_loss,
        'log_mse_time_exp_decay': log_mse_time_exp_decay_loss,
        'log_mae_time_linear_decay': log_mae_time_linear_decay_loss,
        'log_mse_time_linear_decay': log_mse_time_linear_decay_loss,
        'log_mae_time_cos_decay': log_mae_time_cos_decay_loss,
        'log_mse_time_cos_decay': log_mse_time_cos_decay_loss,
        'log_nmae': log_nmae_loss,
        'log_nmae_time_linear_decay': log_nmae_time_linear_decay_loss,
        'log_nmae_time_exp_decay': log_nmae_time_exp_decay_loss,
        'log_nmae_time_cos_decay': log_nmae_time_cos_decay_loss,
        'mixed_pk_loss': mixed_pk_loss,
    }
    return loss_fns.get(loss_fn, log_mae_loss)

def cal_all_losses(y_preds, y_true, **kwargs):
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mae = torch.absolute(torch.log(y_preds) - torch.log(y_true))
    log_mse = (torch.log(y_preds) - torch.log(y_true))**2

    mae = torch.absolute(y_preds - y_true)
    mse = (y_preds - y_true)**2


    mean_log_mae = torch.stack([torch.mean(log_mae[unique_values == i]) for i in unique_indices])
    mean_log_mse = torch.stack([torch.mean(log_mse[unique_values == i]) for i in unique_indices])
    mean_mae = torch.stack([torch.mean(mae[unique_values == i]) for i in unique_indices])
    mean_mse = torch.stack([torch.mean(mse[unique_values == i]) for i in unique_indices])


    r2_scores = torch.stack([torch_r2(y_preds[unique_values == i], y_true[unique_values == i]) for i in unique_indices])
    # mfce = torch.stack([torch_mfce(y_preds[unique_values == i], y_true[unique_values == i]) for i in unique_indices])

    loss_log_mae = torch.mean(mean_log_mae[torch.isfinite(mean_log_mae)])
    loss_log_mse = torch.mean(mean_log_mse[torch.isfinite(mean_log_mse)])
    loss_mae = torch.mean(mean_mae[torch.isfinite(mean_mae)])
    loss_mse = torch.mean(mean_mse[torch.isfinite(mean_mse)])
    loss_rmse = torch.sqrt(loss_mse)

    # r2_scores = r2_scores[torch.isfinite(r2_scores)]
    r2_scores.clip_(min=0, max=1)
    r2_nonzero_radio = torch.sum(r2_scores > 0) / r2_scores.shape[0]
    top30_r2 = torch.topk(r2_scores, int(0.3 * r2_scores.shape[0]), largest=True).values
    top60_r2 = torch.topk(r2_scores, int(0.6 * r2_scores.shape[0]), largest=True).values
    r2_score = torch.mean(r2_scores)
    # mfce = torch.mean(mfce[torch.isfinite(mfce)])
    if kwargs.get('save_path', False):
        save_path = kwargs['save_path']
        #把mean_log_mae, mean_log_mse, r2_scores保存到save_path的csv
        import pandas as pd
        import os
        df = pd.DataFrame({
            'mean_log_mae': mean_log_mae.cpu().numpy(),
            'mean_log_mse': mean_log_mse.cpu().numpy(),
            'r2_scores': r2_scores.cpu().numpy(),
        })
        df.to_csv(os.path.join(save_path, 'losses.csv'), index=False)

    return {
        'log_mae': loss_log_mae.item(),
        'log_mse': loss_log_mse.item(),
        'mae': loss_mae.item(),
        'rmse': loss_rmse.item(),
        'r2_score': r2_score.item(),
        'r2_nonzero_radio': r2_nonzero_radio.item(),
        'top30_r2': top30_r2.mean().item(),
        'top60_r2': top60_r2.mean().item()
        # 'mfce': mfce.item()
    }

def average_r2_score(y_preds, y_true, **kwargs):
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]
    y_true = y_true[mask]

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    r2_scores = torch.stack([torch_r2(y_preds[unique_values == i], y_true[unique_values == i]) for i in unique_indices])
    r2_scores = r2_scores[torch.isfinite(r2_scores)]
    return torch.mean(r2_scores)

def torch_r2(y_preds, y_true):
    # y_preds = torch.log(y_preds + 1e-5)
    # y_true = torch.log(y_true + 1e-5)
    mean_y = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean_y)**2)
    ss_res = torch.sum((y_true - y_preds)**2)
    return 1 - ss_res / ss_tot

def average_mfce(y_preds, y_true, **kwargs):
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]
    y_true = y_true[mask]

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    mfce = torch.stack([torch_mfce(y_preds[unique_values == i], y_true[unique_values == i]) for i in unique_indices])
    mfce = mfce[torch.isfinite(mfce)]
    return torch.mean(mfce)

def torch_mfce(y_preds, y_true):
    y_preds = y_preds + 1e-5
    y_true = y_true + 1e-5
    mfce = torch.exp(torch.median(torch.abs(torch.log(y_preds) - torch.log(y_true))))
    if mfce > 1000:
        mfce = 1000
    return mfce

def log_mae_loss(y_preds, y_true, **kwargs):
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mae = torch.absolute(torch.log(y_preds) - torch.log(y_true))
    mean_log_mae = torch.stack([torch.mean(log_mae[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_mae[torch.isfinite(mean_log_mae)])
    return loss

def log_nmae_loss(y_preds, y_true, **kwargs):
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_max_min = torch.max(torch.where(torch.isnan(y_true), 0, y_true), dim=1).values
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mae = torch.absolute(torch.log(y_preds) - torch.log(y_true))
    mean_log_mae = torch.stack([torch.mean(log_mae[unique_values == i]) for i in unique_indices])
    mean_log_nmae = mean_log_mae / torch.log(y_max_min)
    loss = torch.mean(mean_log_nmae[torch.isfinite(mean_log_nmae)])
    return loss

def log_mse_loss(y_preds, y_true, **kwargs):
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mse = (torch.log(y_preds) - torch.log(y_true))**2
    mean_log_mse = torch.stack([torch.mean(log_mse[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_mse[torch.isfinite(mean_log_mse)])
    return loss

def log_mae_time_exp_decay_loss(y_preds, y_true, times=None, **kwargs):
    times = times/times.max() * kwargs.get('alpha', 1)
    times = times.broadcast_to(y_preds.shape)
    time_decay = torch.exp(-times)
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5
    time_decay = time_decay[mask]

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mae = torch.absolute(torch.log(y_preds) - torch.log(y_true)) * time_decay

    mean_log_mae = torch.stack([torch.mean(log_mae[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_mae[torch.isfinite(mean_log_mae)])
    return loss

def log_nmae_time_exp_decay_loss(y_preds, y_true, times=None, **kwargs):
    times = times.broadcast_to(y_preds.shape)
    time_decay = torch.exp(-times)
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_max_min = torch.max(torch.where(torch.isnan(y_true), 0, y_true), dim=1).values
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5
    time_decay = time_decay[mask]

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mae = torch.absolute(torch.log(y_preds) - torch.log(y_true)) * time_decay
    mean_log_mae = torch.stack([torch.mean(log_mae[unique_values == i]) for i in unique_indices])
    mean_log_nmae = mean_log_mae / torch.log(y_max_min)

    loss = torch.mean(mean_log_nmae[torch.isfinite(mean_log_nmae)])
    return loss

def log_mse_time_exp_decay_loss(y_preds, y_true, times=None, **kwargs):
    times = times.broadcast_to(y_preds.shape)
    time_decay = torch.exp(-times)
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5
    time_decay = time_decay[mask]

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mse = (torch.log(y_preds) - torch.log(y_true))**2 * time_decay

    mean_log_mse = torch.stack([torch.mean(log_mse[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_mse[torch.isfinite(mean_log_mse)])
    return loss

def log_mae_time_linear_decay_loss(y_preds, y_true, times=None, **kwargs):
    times = times.broadcast_to(y_preds.shape)
    time_decay = 1 - times / times.max() * kwargs.get('alpha', 1)
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5
    time_decay = time_decay[mask]

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mae = torch.absolute(torch.log(y_preds) - torch.log(y_true)) * time_decay

    mean_log_mae = torch.stack([torch.mean(log_mae[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_mae[torch.isfinite(mean_log_mae)])
    return loss

def log_nmae_time_linear_decay_loss(y_preds, y_true, times=None, **kwargs):
    times = times.broadcast_to(y_preds.shape)
    time_decay = 1 - times / times.max()
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_max_min = torch.max(torch.where(torch.isnan(y_true), 0, y_true), dim=1).values
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5
    time_decay = time_decay[mask]

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mae = torch.absolute(torch.log(y_preds) - torch.log(y_true)) * time_decay
    mean_log_mae = torch.stack([torch.mean(log_mae[unique_values == i]) for i in unique_indices])

    mean_log_nmae = mean_log_mae / torch.log(y_max_min)
    loss = torch.mean(mean_log_nmae[torch.isfinite(mean_log_nmae)])
    return loss

def log_mse_time_linear_decay_loss(y_preds, y_true, times=None, **kwargs):
    times = times.broadcast_to(y_preds.shape)
    time_decay = 1 - times / times.max()
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5
    time_decay = time_decay[mask]

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mse = (torch.log(y_preds) - torch.log(y_true))**2 * time_decay

    mean_log_mse = torch.stack([torch.mean(log_mse[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_mse[torch.isfinite(mean_log_mse)])
    return loss

def log_mae_time_cos_decay_loss(y_preds, y_true, times=None, **kwargs):
    times = times.broadcast_to(y_preds.shape)
    time_decay = torch.cos(times / times.max() * 3.14159 / 2)
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5
    time_decay = time_decay[mask]

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mae = torch.absolute(torch.log(y_preds) - torch.log(y_true)) * time_decay

    mean_log_mae = torch.stack([torch.mean(log_mae[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_mae[torch.isfinite(mean_log_mae)])
    return loss

def log_nmae_time_cos_decay_loss(y_preds, y_true, times=None, **kwargs):
    times = times.broadcast_to(y_preds.shape)
    time_decay = torch.cos(times / times.max() * 3.14159 / 2)
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_max_min = torch.max(torch.where(torch.isnan(y_true), 0, y_true), dim=1).values
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5
    time_decay = time_decay[mask]

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mae = torch.absolute(torch.log(y_preds) - torch.log(y_true)) * time_decay
    mean_log_mae = torch.stack([torch.mean(log_mae[unique_values == i]) for i in unique_indices])

    mean_log_nmae = mean_log_mae / torch.log(y_max_min)
    loss = torch.mean(mean_log_nmae[torch.isfinite(mean_log_nmae)])
    return loss

def log_mse_time_cos_decay_loss(y_preds, y_true, times=None, **kwargs):
    times = times.broadcast_to(y_preds.shape)
    time_decay = torch.cos(times / times.max() * 3.14159 / 2)
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5
    time_decay = time_decay[mask]

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_mse = (torch.log(y_preds) - torch.log(y_true))**2 * time_decay

    mean_log_mse = torch.stack([torch.mean(log_mse[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_mse[torch.isfinite(mean_log_mse)])
    return loss

def mixed_pk_loss(y_preds, y_true, times, alpha=1.0, lambda_peak=0.5, lambda_auc=0.1, eps=1e-5, **kwargs):
    """
    混合 PK 曲线预测 loss:
    - 时间加权 log MAE
    - Cmax（最大值）loss
    - AUC loss
    - 曲线平滑性 loss（可选）

    Args:
        y_preds: [B, T] 预测值
        y_true:  [B, T] 真实值
        times:   [T]    时间点
        alpha:   float  时间加权因子
        lambda_peak:   float Cmax 权重
        lambda_auc:    float AUC 权重
        lambda_smooth: float 平滑性权重

    Returns:
        loss: float
        metrics: dict
    """
    B, T = y_preds.shape
    device = y_preds.device

    times = times / times.max() * alpha
    times = times.expand(B, T)
    time_decay = torch.exp(-times)

    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = torch.where(mask, y_preds, torch.tensor(eps, device=device))
    y_true = torch.where(mask, y_true, torch.tensor(eps, device=device))
    time_decay = torch.where(mask, time_decay, torch.tensor(0.0, device=device))

    # 时间加权 log MAE
    log_mae = torch.abs(torch.log(y_preds + eps) - torch.log(y_true + eps)) * time_decay
    mean_log_mae = torch.sum(log_mae, dim=1) / torch.sum(mask, dim=1)
    loss_main = torch.mean(mean_log_mae)

    # Cmax loss
    cmax_pred = torch.max(y_preds, dim=1).values
    cmax_true = torch.max(y_true, dim=1).values
    loss_peak = F.l1_loss(cmax_pred, cmax_true)

    # AUC loss
    auc_pred = torch.trapz(y_preds, times, dim=1)
    auc_true = torch.trapz(y_true, times, dim=1)
    loss_auc = F.l1_loss(auc_pred, auc_true)


    # 总 loss
    loss = (
        loss_main +
        lambda_peak * loss_peak +
        lambda_auc * loss_auc
    )

    return loss, {
        "loss_main": loss_main.item(),
        "loss_peak": loss_peak.item(),
        "loss_auc": loss_auc.item(),
    }