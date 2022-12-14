import torch
import numpy as np
from torch import nn
import time
class EarlyStopping():
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.start_time = time.time()

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint_'+str(self.start_time)+'.pt')              
        self.val_loss_min = val_loss

class SkinColorFilter():
    def __init__(self):
        self.mean = np.array([0.0, 0.0])
        self.covariance = np.zeros((2, 2), 'float64')
        self.covariance_inverse = np.zeros((2, 2), 'float64')

    def __generate_circular_mask(self, image, radius_ratio=0.4):
        x_center = image.shape[1] / 2
        y_center = image.shape[2] / 2

        # arrays with the image coordinates
        x = np.zeros((image.shape[1], image.shape[2]))
        x[:] = range(0, x.shape[1])
        y = np.zeros((image.shape[2], image.shape[1]))
        y[:] = range(0, y.shape[1])
        y = np.transpose(y)

        # translate s.t. the center is the origin
        x -= x_center
        y -= y_center

        # condition to be inside of a circle: x^2 + y^2 < r^2
        radius = radius_ratio*image.shape[2]
        self.circular_mask = (x**2 + y**2) < (radius**2)


    def __remove_luma(self, image):
        # compute the mean and std of luma values on non-masked pixels only
        luma = 0.299*image[0, self.circular_mask] + 0.587*image[1, self.circular_mask] + 0.114*image[2, self.circular_mask]
        m = np.mean(luma)
        s = np.std(luma)

        # apply the filtering to the whole image to get the luma mask
        luma = 0.299*image[0, :, :] + 0.587*image[1, :, :] + 0.114*image[2, :, :]
        self.luma_mask = np.logical_and((luma > (m - 1.5*s)), (luma < (m + 1.5*s)))


    def estimate_gaussian_parameters(self, image):
        self.__generate_circular_mask(image)
        self.__remove_luma(image)
        mask = np.logical_and(self.luma_mask, self.circular_mask)

        # get the mean
        channel_sum = image[0].astype('float64') + image[1] + image[2]
        nonzero_mask = np.logical_or(np.logical_or(image[0] > 0, image[1] > 0), image[2] > 0)
        r = np.zeros((image.shape[1], image.shape[2]))
        r[nonzero_mask] = image[0, nonzero_mask] / channel_sum[nonzero_mask]
        g = np.zeros((image.shape[1], image.shape[2]))
        g[nonzero_mask] = image[1, nonzero_mask] / channel_sum[nonzero_mask]
        self.mean = np.array([np.mean(r[mask]), np.mean(g[mask])])

        # get the covariance
        r_minus_mean = r[mask] - self.mean[0]
        g_minus_mean = g[mask] - self.mean[1]
        samples = np.vstack((r_minus_mean, g_minus_mean))
        samples = samples.T
        cov = sum([np.outer(s,s) for s in samples])
        self.covariance = cov / float(samples.shape[0] - 1) 

        # store the inverse covariance matrix (no need to recompute)
        if np.linalg.det(self.covariance) != 0:
            self.covariance_inverse = np.linalg.inv(self.covariance)
        else:
            self.covariance_inverse = np.zeros_like(self.covariance)



    def get_skin_mask(self, image, threshold):
        """get_skin_mask(image, [threshold]) -> skin_mask
        This function computes the probability of skin-color for each pixel in the image.

        **Parameters:**

          ``image`` : (np array) 
            The face image.

          ``threshold`` : (Optional, float between 0 and 1) 
            the threshold on the skin color probability. Defaults to 0.5

        **Returns:**

          ``skin_mask`` : (np logical array)
          The mask where skin color pixels are labeled as True.
        """
        skin_map = np.zeros((image.shape[1], image.shape[2]), 'float64')

        # get the image in rg colorspace
        channel_sum = image[0].astype('float64') + image[1] + image[2]
        nonzero_mask = np.logical_or(np.logical_or(image[0] > 0, image[1] > 0), image[2] > 0)
        r = np.zeros((image.shape[1], image.shape[2]), 'float64')
        r[nonzero_mask] = image[0, nonzero_mask] / channel_sum[nonzero_mask]
        g = np.zeros((image.shape[1], image.shape[2]), 'float64')
        g[nonzero_mask] = image[1, nonzero_mask] / channel_sum[nonzero_mask]

        # compute the skin probability map
        r_minus_mean = r - self.mean[0]
        g_minus_mean = g - self.mean[1]
        v = np.dstack((r_minus_mean, g_minus_mean))
        v = v.reshape((r.shape[0]*r.shape[1], 2))
        probs = [np.dot(k, np.dot(self.covariance_inverse, k)) for k in v]
        probs = np.array(probs).reshape(r.shape)
        skin_map = np.exp(-0.5 * probs)

        return skin_map > threshold


class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):       # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            for j in range(0,4):           
                sum_x = torch.sum(preds[i][j])                # x
                sum_y = torch.sum(labels[i][j])               # y
                sum_xy = torch.sum(preds[i][j]*labels[i][j])        # xy
                sum_x2 = torch.sum(torch.pow(preds[i][j],2))  # x^2
                sum_y2 = torch.sum(torch.pow(labels[i][j],2)) # y^2
                N = preds.shape[2] #64
                pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
      
                if (pearson>=0):
                    loss += 1 - (pearson**2)
                else:
                    loss += 1 + (pearson**2) 

        
        loss = loss/(preds.shape[0]*4) 
        return loss
