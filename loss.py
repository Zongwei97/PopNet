import torch
import torch.nn as nn

    
class SSIM(torch.nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _ssim(self, x, y):
        abs_diff = torch.abs(y - x)
        l1_loss = abs_diff.mean(1, True)
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        ssim_loss = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1).mean(1,True)
        l1_loss= 0.85 * ssim_loss + 0.15 * l1_loss
        

        return ssim_loss.mean()

    def forward(self, pred, target):
        return self._ssim(pred, target)


def smooth_normal_loss(disp):
    grad = get_gradient(disp)
    grad_x, grad_y = grad[:, 0].unsqueeze(1), grad[:, 1].unsqueeze(1)
    ones = torch.ones(grad.size(0), 1, grad.size(2),grad.size(3)).float().cuda()
    ones = torch.autograd.Variable(ones)
    depth_normal = torch.cat((-grad_x, -grad_y, ones), 1)
    l1 = torch.abs(1 - cos(depth_normal[:, :, :, :-1], depth_normal[:, :, :, 1:]))
    l2 = torch.abs(1 - cos(depth_normal[:, :, :-1, :], depth_normal[:, :, 1:, :]))
    l3 = torch.abs(1 - cos(depth_normal[:, :, 1:, :-1], depth_normal[:, :, :-1, 1:]))
    l4 = torch.abs(1 - cos(depth_normal[:, :, :-1, :-1], depth_normal[:, :, 1:, 1:]))
    return (l1.mean() + l2.mean() + l3.mean() + l4.mean())/4

def total_variation_loss(img, weight):
    edge_h = 1-torch.abs(weight[:,:,1:,:] - weight[:,:,:-1,:])
    edge_h += edge_h.mean()
    edge_w = 1-torch.abs(weight[:,:,:,1:] - weight[:,:,:,:-1])
    edge_w += edge_w.mean()
    tv_h = edge_h * torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2)
    tv_w = edge_w * torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2)
    return (tv_h.sum()+tv_w.sum())/2


def weighted_total_variant(disp, gt):
    grad = get_gradient(gt)
    grad_x, grad_y = grad[:, 0].unsqueeze(1), grad[:, 1].unsqueeze(1)
    temp_edge = torch.mul(grad_x, grad_x) +  torch.mul(grad_y, grad_y)
    temp_edge[temp_edge != 0] = 1
    temp_edge[temp_edge == 0] = 0.5
    to_var = total_variation_loss(disp, temp_edge)
    return to_var
    

def structure_loss(pred, mask):  
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def l1_loss(x,y):
    abs_diff = torch.abs(y - x)
    l1_loss = abs_diff.mean(1, True)
    return l1_loss
    
