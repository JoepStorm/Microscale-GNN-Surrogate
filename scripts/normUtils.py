"""
Normalization utilities.
A normalization class can be initialized based on a dataset.
Afterwards, the ".normalize( )" and ".denormalize( )" functions can be easily called
Additional functions are hardcoded to print the values required for storing and reloading normalization quantities.
This avoids having to load all data to obtain the normalization quantities when only using a model in inference. Used in "load_model.py"
"""

class normUnitvar:
    def __init__(self, fullDataset=None, donorm=True, normmean=None, normstd=None):
        self.donorm = donorm
        if normmean is None or normstd is None:
            self.normmean = fullDataset.mean()
            self.normstd = fullDataset.std()
        if normmean is not None:
            self.normmean = normmean
        if normstd is not None:
            self.normstd = normstd
        self.factor = self.normstd

    def normalize(self, data):
        if self.donorm:
            return (data - self.normmean) / self.normstd
        else:
            return data

    def denormalize(self, data):
        if self.donorm:
            return data * self.normstd + self.normmean
        else:
            return data


class normZeroone:
    def __init__(self, fullDataset=None, donorm=True, normmin=None, normmax=None):
        self.donorm = donorm
        if normmin is None:
            self.normmin = fullDataset.min()
        else:
            self.normmin = normmin
        if normmax is None:
            self.normmax = fullDataset.max()
        else:
            self.normmax = normmax
        self.factor =  (self.normmax - self.normmin)

    def normalize(self, data):
        if self.donorm:
            return (data - self.normmin) / (self.normmax - self.normmin)
        else:
            return data

    def denormalize(self, data):
        if self.donorm:
            return data * (self.normmax - self.normmin) + self.normmin
        else:
            return data

class scaleRoundZeroMaxOne:
    def __init__(self, fullDataset=None, donorm=True, normmin=None, normmax=None, factor=None):
        self.donorm = donorm
        if factor is not None:
            self.factor = factor
        else:
            if normmin is None or normmax is None:
                normmin = fullDataset.min()
                normmax = fullDataset.max()
            self.factor = max(abs(normmax), abs(normmin))

    def normalize(self, data):
        if self.donorm:
            return data / self.factor
        else:
            return data

    def denormalize(self, data):
        if self.donorm:
            return data * self.factor
        else:
            return data


def readNormFile(filename):
    norms = []
    with open(filename, 'r') as f:
        for k, line in enumerate(f):
            array = line.split()
            do_norm = array[1]
            if len(array) == 4:
                factor = float(array[2])
            if len(array) == 6:
                first  = float(array[2])
                second = float(array[3])
            if array[0] == 'geomFeatNorm':
                norms.append(normUnitvar(donorm=do_norm, normmean=first, normstd=second))
            if array[0] == 'edgeFeatNorm':
                norms.append(normUnitvar(donorm=do_norm, normmean=first, normstd=second))
            if array[0] == 'macroStrainNorm':
                norms.append(normUnitvar(donorm=do_norm, normmean=first, normstd=second))
            if array[0] == 'elemStrainNorm':
                norms.append(normUnitvar(donorm=do_norm, normmean=first, normstd=second))
            if array[0] == 'elemepspNorm':
                norms.append(normUnitvar(donorm=do_norm, normmean=first, normstd=second))
            if array[0] == 'elemepspeqNorm':
                norms.append(normUnitvar(donorm=do_norm, normmean=first, normstd=second))
            if array[0] == 'homStressNorm':
                norms.append(normUnitvar(donorm=do_norm, normmean=first, normstd=second))
            if array[0] == 'elemStressNorm':
                norms.append(normUnitvar(donorm=do_norm, normmean=first, normstd=second))
    return norms

def writeNormFile(filename, settings, edgevec, geomFeatNorm, edgeFeatNorm, macroStrainNorm, elemStrainNorm, elemepspNorm, elemepspeqNorm, homStressNorm=None, elemStressNorm=None):
    with open(filename, 'w') as f:
        f.write(f"geomFeatNorm {settings['norm_inputs']} {geomFeatNorm.normmean} {geomFeatNorm.normstd} mean std\n")
        if edgevec:
            f.write(f"edgeFeatNorm {settings['norm_inputs']} {edgeFeatNorm.normmean} {edgeFeatNorm.normstd} mean std\n")
        f.write(
            f"macroStrainNorm {settings['norm_inputs']} {macroStrainNorm.normmean} {macroStrainNorm.normstd} mean std\n")
        f.write(
            f"elemStrainNorm {settings['norm_targets']} {elemStrainNorm.normmean} {elemStrainNorm.normstd} mean std\n")
        f.write(f"elemepspNorm {settings['norm_inputs']} {elemepspNorm.normmean} {elemepspNorm.normstd} mean std\n")
        f.write(f"elemepspeqNorm {settings['norm_inputs']} {elemepspeqNorm.normmean} {elemepspeqNorm.normstd} mean std\n")
        f.write(f"homStressNorm {settings['norm_inputs']} {homStressNorm.normmean} {homStressNorm.normstd} mean std\n")
        f.write(f"elemStressNorm {settings['norm_inputs']} {elemStressNorm.normmean} {elemStressNorm.normstd} mean std\n")
