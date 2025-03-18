import torch


def get_loss_fn(loss_fn):
    if loss_fn == 'log_mae':
        return log_mae_loss
    elif loss_fn == 'log_mse':
        return log_mse_loss
    elif loss_fn == 'log_cosh':
        return log_cosh_loss
    elif loss_fn == 'log_mae_quantile':
        return log_mae_quantile_loss
    elif loss_fn == 'log_mae_time_exp_decay':
        return log_mae_time_exp_decay_loss
    elif loss_fn == 'log_mse_time_exp_decay':
        return log_mse_time_exp_decay_loss
    elif loss_fn == 'log_mae_time_linear_decay':
        return log_mae_time_linear_decay_loss
    elif loss_fn == 'log_mse_time_linear_decay':
        return log_mse_time_linear_decay_loss
    elif loss_fn == 'log_mae_time_cos_decay':
        return log_mae_time_cos_decay_loss
    elif loss_fn == 'log_mse_time_cos_decay':
        return log_mse_time_cos_decay_loss
    else:
        return log_mae_loss

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

    r2_scores = r2_scores[torch.isfinite(r2_scores)]
    r2_scores.clip_(min=0, max=1)
    r2_nonzero_radio = torch.sum(r2_scores > 0) / r2_scores.shape[0]
    top30_r2 = torch.topk(r2_scores, int(0.3 * r2_scores.shape[0]), largest=True).values
    top60_r2 = torch.topk(r2_scores, int(0.6 * r2_scores.shape[0]), largest=True).values
    r2_score = torch.mean(r2_scores)
    # mfce = torch.mean(mfce[torch.isfinite(mfce)])
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

def log_cosh_loss(y_preds, y_true, **kwargs):
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_cosh = torch.log(torch.cosh(y_preds - y_true))
    mean_log_cosh = torch.stack([torch.mean(log_cosh[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_cosh[torch.isfinite(mean_log_cosh)])
    return loss

def log_mae_quantile_loss(y_preds, y_true, alpha=0.75, **kwargs):
    mask = torch.isfinite(y_true) & torch.isfinite(y_preds)
    y_preds = y_preds[mask]+1e-5
    y_true = y_true[mask]+1e-5

    unique_indices = torch.arange(mask.shape[0], device=mask.device)
    unique_values = unique_indices.repeat_interleave(mask.sum(dim=1))

    log_loss = torch.max(alpha * torch.abs(torch.log(y_preds) - torch.log(y_true)), (alpha - 1) * torch.abs(torch.log(y_preds) - torch.log(y_true)))
    mean_log_loss = torch.stack([torch.mean(log_loss[unique_values == i]) for i in unique_indices])
    loss = torch.mean(mean_log_loss[torch.isfinite(mean_log_loss)])
    return loss

def log_mae_time_exp_decay_loss(y_preds, y_true, times=None, **kwargs):
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
    time_decay = 1 - times / times.max()
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