import albumentations as A
import numpy as np
import random
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.spatial_transforms import ZoomTransform

from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import interpn
from medpy.metric import binary

from numpy.fft import fftn, ifftn
import SimpleITK as sitk

try:
    from scipy.special import comb
except:
    from scipy.misc import comb


identity_prob = 0.85

class OpticalDistortion(AbstractTransform):
    def __init__(self, data_key="data", seg_key="seg"):
        self.opt = A.augmentations.geometric.transforms.OpticalDistortion(value=0, mask_value=0)
        self.seg_key = seg_key
        self.data_key = data_key

    def __call__(self, **data_dict):
        if np.random.rand() < identity_prob:
            return data_dict

        feature, target = data_dict[self.data_key], data_dict[self.seg_key]
        params = self.opt.get_params()

        for batch_index in range(feature.shape[0]):
            for channel_index in range(feature.shape[1]):
                for z_index in range(feature.shape[2]):
                    feature[batch_index, channel_index, z_index] = self.opt.apply(
                        feature[batch_index, channel_index, z_index], **params
                    )
                    if channel_index == 0:
                        target[batch_index, channel_index, z_index] = self.opt.apply_to_mask(
                            target[batch_index, channel_index, z_index], **params
                        )

        data_dict[self.data_key], data_dict[self.seg_key] = feature, target
        return data_dict


class ElasticTransform(AbstractTransform):
    def __init__(self, data_key="data", seg_key="seg"):
        self.opt = A.augmentations.geometric.transforms.ElasticTransform(value=0, mask_value=0)
        self.seg_key = seg_key
        self.data_key = data_key

    def __call__(self, **data_dict):
        if np.random.rand() < identity_prob:
            return data_dict

        feature, target = data_dict[self.data_key], data_dict[self.seg_key]
        params = self.opt.get_params()

        for batch_index in range(feature.shape[0]):
            for channel_index in range(feature.shape[1]):
                for z_index in range(feature.shape[2]):
                    feature[batch_index, channel_index, z_index] = self.opt.apply(
                        feature[batch_index, channel_index, z_index], **params
                    )
                    if channel_index == 0:
                        target[batch_index, channel_index, z_index] = self.opt.apply_to_mask(
                            target[batch_index, channel_index, z_index], **params
                        )

        data_dict[self.data_key], data_dict[self.seg_key] = feature, target
        return data_dict


class GridDistortion(AbstractTransform):
    def __init__(self, data_key="data", seg_key="seg"):
        self.opt = A.augmentations.geometric.transforms.GridDistortion(value=0, mask_value=0)
        self.seg_key = seg_key
        self.data_key = data_key
        self.patch_size = GridDistortion.patch_size
        self.patch_size = self.patch_size if len(self.patch_size) == 2 else self.patch_size[1:]
        # self.placeholder = {"image": np.zeros(patch_size)}

    def __call__(self, **data_dict):
        if np.random.rand() < identity_prob:
            return data_dict

        feature, target = data_dict[self.data_key], data_dict[self.seg_key]
        params = self.opt.get_params_dependent_on_targets({"image": np.zeros(self.patch_size)})

        for batch_index in range(feature.shape[0]):
            for channel_index in range(feature.shape[1]):
                for z_index in range(feature.shape[2]):
                    feature[batch_index, channel_index, z_index] = self.opt.apply(
                        feature[batch_index, channel_index, z_index], **params
                    )
                    if channel_index == 0:
                        target[batch_index, channel_index, z_index] = self.opt.apply_to_mask(
                            target[batch_index, channel_index, z_index], **params
                        )

        data_dict[self.data_key], data_dict[self.seg_key] = feature, target
        return data_dict


class Affine(AbstractTransform):
    def __init__(self, scale_factor, data_key="data", seg_key="seg", **kwargs):
        self.opt = A.augmentations.geometric.transforms.Affine
        self.seg_key = seg_key
        self.data_key = data_key
        self.scale_left, self.scale_right = scale_factor
        self.kwargs = kwargs
        self.patch_size = Affine.patch_size
        self.patch_size = self.patch_size if len(self.patch_size) == 2 else self.patch_size[1:]
        # self.placeholder = {"image": np.zeros(self.patch_size)}

    def __call__(self, **data_dict):
        if np.random.rand() < identity_prob:
            return data_dict

        feature, target = data_dict[self.data_key], data_dict[self.seg_key]

        kwargs = self.kwargs.copy()
        kwargs["scale"] = np.random.uniform(self.scale_left, self.scale_right)
        opt = self.opt(**kwargs)
        params = opt.get_params_dependent_on_targets({"image": np.zeros(self.patch_size)})

        for batch_index in range(feature.shape[0]):
            for channel_index in range(feature.shape[1]):
                for z_index in range(feature.shape[2]):
                    feature[batch_index, channel_index, z_index] = opt.apply(
                        feature[batch_index, channel_index, z_index], **params
                    )
                    if channel_index == 0:
                        target[batch_index, channel_index, z_index] = opt.apply_to_mask(
                            target[batch_index, channel_index, z_index], **params
                        )

        data_dict[self.data_key], data_dict[self.seg_key] = feature, target
        return data_dict


class BezierCurveTransform_similar(AbstractTransform):
    def __init__(self, data_key="data", p_per_sample=0.8):
        self.data_key = data_key
        self.points_set = [
            [[-1, -1], [-0.25, 0.25], [0.25, -0.25], [1, 1]],
            [[-1, -1], [-0.375, 0.375], [0.375, -0.375], [1, 1]],
            [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]],
            [[-1, -1], [-0.625, 0.625], [0.625, -0.625], [1, 1]],
            [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]],
            [[-1, -1], [-0.875, 0.875], [0.875, -0.875], [1, 1]],
            [[-1, -1], [-1, -1], [1, 1], [1, 1]],
        ]
        self.ops = [0, 1, 2, 3, 4, 5, 6]
        self.p_per_sample = p_per_sample

    def bernstein_poly(self, i, n, t):
        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(self, points, nTimes=1000):
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        t = np.linspace(0.0, 1.0, nTimes)
        polynomial_array = np.array([self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        return xvals, yvals

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        op = random.choices(self.ops, k=1)[0]
        points = self.points_set[op]
        if random.random() < self.p_per_sample:
            for b in range(data.shape[0]):
                min, max = np.min(data[b]), np.max(data[b])
                # for c in range(data.shape[1]):
                #     out = sitk.GetImageFromArray(data[b][c])
                #     sitk.WriteImage(out, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+str(c)+'_org.nii.gz')

                # data[b] = 2 * (data[b] - min) / (max - min) - 1
                xvals, yvals = self.bezier_curve(points, nTimes=100000)
                xvals, yvals = np.sort(xvals), np.sort(yvals)
                nonlinear_slices = np.interp(data[b], xvals, yvals)
                data[b] = nonlinear_slices
                data[b] = (data[b] + 1) * (max - min) / 2 + min
        data_dict[self.data_key] = data
        return data_dict


class FourierMixTransform(AbstractTransform):
    def __init__(self, data_key="data", p_per_sample=0.8):
        """
        Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma
        :param mu: mean of the Gaussian distribution to sample the added brightness from
        :param sigma: standard deviation of the Gaussian distribution to sample the added brightness from
        :param per_channel: whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.data_key = data_key
        self.p_per_sample = p_per_sample

    def mixup(self, A1, A2, alpha):
        lam = np.random.beta(alpha, alpha)
        A1 = lam * A1 + (1 - lam) * A2
        A2 = lam * A2 + (1 - lam) * A1
        return A1, A2

    def fourier_mix(self, F1, F2):
        Mod1 = np.abs(F1)
        fas1 = np.arctan(F1.imag / F1.real)
        pi_n1 = -np.logical_and(F1.imag < 0, F1.real < 0).astype("int") * np.pi
        pi_p1 = np.logical_and(F1.imag > 0, F1.real < 0).astype("int") * np.pi
        fas1 = fas1 + pi_n1 + pi_p1

        Mod2 = np.abs(F2)
        fas2 = np.arctan(F2.imag / F2.real)
        pi_n2 = -np.logical_and(F2.imag < 0, F2.real < 0).astype("int") * np.pi
        pi_p2 = np.logical_and(F2.imag > 0, F2.real < 0).astype("int") * np.pi
        fas2 = fas2 + pi_n2 + pi_p2

        # Mixup Mods
        Mod1, Mod2 = self.mixup(Mod1, Mod2, alpha=0.2)

        F1 = Mod1 * np.cos(fas1) + Mod1 * np.sin(fas1) * 1j
        F2 = Mod2 * np.cos(fas2) + Mod2 * np.sin(fas2) * 1j
        return ifftn(F1), ifftn(F2)

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        if random.random() < self.p_per_sample:
            if data.shape[0] > 1:
                for b in range(0, data.shape[0], 2):
                    data1, data2 = data[b], data[b + 1]
                    min1 = np.min(data1)
                    max1 = np.max(data1)
                    data1 = (data1 - min1) / (max1 - min1)
                    min2 = np.min(data2)
                    max2 = np.max(data2)
                    data2 = (data2 - min2) / (max2 - min2)

                    for c in range(data.shape[1]):
                        F1 = fftn(data1[c])
                        F2 = fftn(data2[c])
                        inv1, inv2 = self.fourier_mix(F1, F2)
                        data1[c] = np.abs(inv1)
                        data2[c] = np.abs(inv2)

                    data[b] = data1 * (max1 - min1) + min1
                    data[b + 1] = data2 * (max2 - min2) + min2
        data_dict[self.data_key] = data
        return data_dict


class AdaptiveHistogramEqualizationImageFilter(AbstractTransform):
    def __init__(self, data_key="data"):
        self.data_key = data_key
        self.opt = sitk.AdaptiveHistogramEqualizationImageFilter()

    def __call__(self, **data_dict):
        if random.random() < 0.2:
            return data_dict

        data = data_dict[self.data_key]
        op = self.opt
        # print(op)
        for b in range(data.shape[0]):
            min, max = np.min(data[b]), np.max(data[b])
            # print("before",np.max(data[b]),np.min(data[b]))
            data[b] = (data[b] - min) / (max - min)
            for c in range(data.shape[1]):
                image = sitk.GetImageFromArray(data[b][c])
                # sitk.WriteImage(itk, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+str(c)+'_org.nii.gz')
                res = op.Execute(image)
                # sitk.WriteImage(itk, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+str(c)+'_zafter.nii.gz')
                data[b][c] = sitk.GetArrayFromImage(res)

            data[b] = data[b] * (max - min) + min
            # print("after",np.max(data[b]),np.min(data[b]))
            # sys.exit(0)
        data_dict[self.data_key] = data
        return data_dict


class LaplacianImageFilter(AbstractTransform):
    def __init__(self, data_key="data"):
        self.data_key = data_key
        self.opt = sitk.LaplacianImageFilter()

    def __call__(self, **data_dict):
        if random.random() < 0.2:
            return data_dict
        data = data_dict[self.data_key]
        op = self.opt
        # print(op)
        for b in range(data.shape[0]):
            min, max = np.min(data[b]), np.max(data[b])
            # print("before",np.max(data[b]),np.min(data[b]))
            data[b] = (data[b] - min) / (max - min)
            for c in range(data.shape[1]):
                image = sitk.GetImageFromArray(data[b][c])
                # sitk.WriteImage(itk, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+str(c)+'_org.nii.gz')
                res = op.Execute(image)
                # sitk.WriteImage(itk, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+str(c)+'_zafter.nii.gz')
                data[b][c] = sitk.GetArrayFromImage(res)

            data[b] = data[b] * (max - min) + min
            # print("after",np.max(data[b]),np.min(data[b]))
            # sys.exit(0)
        data_dict[self.data_key] = data
        return data_dict


class SobelEdgeDetectionImageFilter(AbstractTransform):
    def __init__(self, data_key="data"):
        self.data_key = data_key
        self.opt = sitk.SobelEdgeDetectionImageFilter()

    def __call__(self, **data_dict):
        if random.random() < 0.2:
            return data_dict
        data = data_dict[self.data_key]
        op = self.opt
        # print(op)
        for b in range(data.shape[0]):
            min, max = np.min(data[b]), np.max(data[b])
            # print("before",np.max(data[b]),np.min(data[b]))
            data[b] = (data[b] - min) / (max - min)
            for c in range(data.shape[1]):
                image = sitk.GetImageFromArray(data[b][c])
                # sitk.WriteImage(itk, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+str(c)+'_org.nii.gz')
                res = op.Execute(image)
                # sitk.WriteImage(itk, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+str(c)+'_zafter.nii.gz')
                data[b][c] = sitk.GetArrayFromImage(res)

            data[b] = data[b] * (max - min) + min
            # print("after",np.max(data[b]),np.min(data[b]))
            # sys.exit(0)
        data_dict[self.data_key] = data
        return data_dict


class InvertIntensityImageFilter(AbstractTransform):
    def __init__(self, data_key="data"):
        self.data_key = data_key
        self.opt = sitk.InvertIntensityImageFilter()

    def __call__(self, **data_dict):
        if random.random() < 0.2:
            return data_dict

        data = data_dict[self.data_key]
        op = self.opt
        # print(op)
        for b in range(data.shape[0]):
            min, max = np.min(data[b]), np.max(data[b])
            # print("before",np.max(data[b]),np.min(data[b]))
            data[b] = (data[b] - min) / (max - min)
            for c in range(data.shape[1]):
                image = sitk.GetImageFromArray(data[b][c])
                # sitk.WriteImage(itk, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+str(c)+'_org.nii.gz')
                res = op.Execute(image)
                # sitk.WriteImage(itk, '/home/listu/hnwang/nnFormer/nnformer/test_aug/'+str(b)+str(c)+'_zafter.nii.gz')
                data[b][c] = sitk.GetArrayFromImage(res)

            data[b] = data[b] * (max - min) + min
            # print("after",np.max(data[b]),np.min(data[b]))
            # sys.exit(0)
        data_dict[self.data_key] = data
        return data_dict


class InterpolationTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=0.8):
        """
        Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma
        :param mu: mean of the Gaussian distribution to sample the added brightness from
        :param sigma: standard deviation of the Gaussian distribution to sample the added brightness from
        :param per_channel: whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    @staticmethod
    def process_label(label):
        labels = []
        # print("=====",np.unique(label))
        net = label == 2
        ed = label == 1
        et = label == 3
        ET = et
        TC = net + et
        WT = net + et + ed
        labels.append(ET)
        labels.append(TC)
        labels.append(WT)
        return labels

    @staticmethod
    def reconst_label(labels):
        label = labels[2].astype(np.int)
        label[label > 0] = 1
        label[labels[1] > 0] = 2
        label[labels[0] > 0] = 3
        return label

    @staticmethod
    def hd(pred, gt):
        # print(pred.sum(), gt.sum())
        if pred.sum() > 0 and gt.sum() > 0:
            hd95 = binary.hd(pred, gt)
            return hd95
        else:
            return 0

    @staticmethod
    def getRangeImageDepth(image):
        z = np.any(image, axis=(1, 2))  # z.shape:(depth,)
        # print("all index:",np.where(z)[0])
        if len(np.where(z)[0]) > 0:
            startposition, endposition = np.where(z)[0][[0, -1]]
        else:
            startposition = endposition = 0
        return startposition, endposition

    def interpolate_label(self, label1, label2):
        labels_new = []
        for i in range(len(label1)):
            if label1[i].sum() and label2[i].sum():
                label_new = self.interp_shape(label1[i], label2[i], random.uniform(0.3, 0.7))
                labels_new.append(label_new)
            else:
                labels_new.append(label1[i])
        return labels_new

    @staticmethod
    def bwperim(im):
        im = im.astype(np.int)
        return 0 - (im - binary_dilation(im))

    @staticmethod
    def bwdist(im):
        """
        Find distance map of image
        """
        dist_im = distance_transform_edt(1 - im)
        return dist_im

    def signed_bwdist(self, im):
        """
        Find perim and return masked image (signed/reversed)
        """
        im = -self.bwdist(self.bwperim(im)) * np.logical_not(im) + self.bwdist(self.bwperim(im)) * im
        return im

    def interp_shape(self, top, bottom, precision):
        """
        Interpolate between two contours

        Input: top
                [X,Y] - Image of top contour (mask)
               bottom
                [X,Y] - Image of bottom contour (mask)
               precision
                 float  - % between the images to interpolate
                    Ex: num=0.5 - Interpolate the middle image between top and bottom image
        Output: out
                [X,Y] - Interpolated image at num (%) between top and bottom
        """
        if precision > 2:
            print("Error: Precision must be between 0 and 1 (float)")

        top = self.signed_bwdist(top)
        bottom = self.signed_bwdist(bottom)
        # row,cols definition
        r, c = top.shape
        # Reverse % indexing
        precision = 1 + precision
        # rejoin top, bottom into a single array of shape (2, r, c)
        top_and_bottom = np.stack((top, bottom))
        # create ndgrids
        points = (np.r_[0, 2], np.arange(r), np.arange(c))
        xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r * c, 2))
        xi = np.c_[np.full((r * c), precision), xi]
        # Interpolate for new plane
        out = interpn(points, top_and_bottom, xi)
        out = out.reshape((r, c))
        # Threshold distmap to values above 0
        out = out > 0
        return out

    def interp_trans(self, data, label, threshhold_ratio=0.1):

        data = data.transpose((1, 0, 2, 3))
        start, end = self.getRangeImageDepth(label)
        if start == 0 and end == 0:
            return None
        depth = label.shape[0]
        minhw = min(label.shape[2], label.shape[2])
        self.hd_threshhold = threshhold_ratio * minhw
        i = start
        while i < end:
            labels1 = self.process_label(label[i])
            labels2 = self.process_label(label[i + 1])
            data, label, inter_times = self.interp_insert(labels1, labels2, data, label, i)
            end = end + inter_times
            i = i + inter_times

            i = i + 1
        start2, end2 = self.getRangeImageDepth(label)
        mid = (end2 + start2) // 2
        if label.shape[0] - mid < depth // 2:
            mid -= depth // 2 - (label.shape[0] - mid)
        elif mid < depth // 2:
            mid += depth // 2 - mid
        label = label[mid - depth // 2 : mid + depth // 2]
        data = data[mid - depth // 2 : mid + depth // 2]
        return data, label

    def interp_insert(self, labels1, labels2, data, label, index, max_interp_times=3):
        hds = []
        interp_times = 0
        for i in range(len(labels1)):
            hds.append(self.hd(labels1[i], labels2[i]))

        while np.max(hds) > self.hd_threshhold and interp_times < max_interp_times:
            interp_times += 1
            index += 1
            labels_new = self.interpolate_label(labels1, labels2)
            labels_recon = self.reconst_label(labels_new)
            # print("sc")
            label = np.insert(label, index, labels_recon, axis=0)
            data = np.insert(data, index, (data[index - 1] + data[index]) / 2, axis=0)
            hds1 = []
            hds2 = []
            for i in range(len(labels1)):
                hds1.append(self.hd(labels1[i], labels_new[i]))
            for i in range(len(labels1)):
                hds2.append(self.hd(labels_new[i], labels2[i]))
            if np.max(hds1) > np.max(hds2):
                hds = hds1
                labels2 = labels_new
                index -= 1
            else:
                hds = hds2
                labels1 = labels_new
        return data, label, interp_times

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]
        label = data_dict[self.label_key]
        data_after = []
        label_after = []
        label[label < 0] = 0
        if random.random() < self.p_per_sample:
            # if True:
            for b in range(0, data.shape[0]):
                data_cur, label_cur = self.interp_trans(data[b], label[b].squeeze())
                data_cur = data_cur.transpose((1, 0, 2, 3))
                data_after.append(data_cur)
                label_after.append(label_cur)
            data_dict[self.data_key] = np.array(data_after)
            data_dict[self.label_key] = np.array(label_after)[:, np.newaxis, :, :, :]
        else:
            data_dict[self.data_key] = data
            data_dict[self.label_key] = label
        return data_dict
