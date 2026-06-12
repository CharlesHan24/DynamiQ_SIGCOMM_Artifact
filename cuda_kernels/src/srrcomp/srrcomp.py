# Copyright 2022 VMware, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# @author: Shay Vargaftik (VMware Research), shayv@vmware.com
# @author: Yaniv Ben-Itzhak (VMware Research), ybenitzhak@vmware.com

# !/usr/bin/env python3

import torch
import numpy as np
import warnings

class SRRCompUtils:

    def __init__(self, gpuacctype = 'cuda'):

        self.gpuacctype = gpuacctype

        self.utils = {'cpu': {}, 'gpu': {}}

        # Torch implementation for Hadamard and bit packing to be used (when the vector resides in the cpu).
        self.utils['cpu']['Hadamard'] = self.Hadamard
        self.utils['cpu']['BinsToBits'] = self.EdenBinsToBits
        self.utils['cpu']['BitsToBins'] = self.EdenBitsToBins

        if torch.cuda.is_available():

            if self.gpuacctype == 'cuda':

                try:

                    # CUDA implementation for Hadamard and bit packing (when the vector resides in a gpu device).
                    from . import eden_utils
                    self.utils['gpu']['Hadamard'] = eden_utils.Hadamard
                    self.utils['gpu']['BinsToBits'] = eden_utils.EdenBinsToBits
                    self.utils['gpu']['BitsToBins'] = eden_utils.EdenBitsToBins


                except:

                    # Torch implementation for Hadamard and bit packing (when the vector resides in a gpu device).
                    self.utils['gpu']['Hadamard'] = self.Hadamard
                    self.utils['gpu']['BinsToBits'] = self.EdenBinsToBits
                    self.utils['gpu']['BitsToBins'] = self.EdenBitsToBins

                    print("Faster CUDA implementation for Hadamard and bit packing is not available. "
                          "Using torch implementation instead.\nTo use CUDA the implementation "
                          "follow the \"Installation\" instructions in the README file")

            elif self.gpuacctype == 'torch':

                # Torch implementation for Hadamard and bit packing (when the vector resides in a gpu device).
                self.utils['gpu']['Hadamard'] = self.Hadamard
                self.utils['gpu']['BinsToBits'] = self.EdenBinsToBits
                self.utils['gpu']['BitsToBins'] = self.EdenBitsToBins

            else:

                raise Exception('Illegal gpuacctype {}, use \'cuda\' or \'torch\'.'.format(self.gpuacctype))

    def randDiag(self, size, device, seed, torchRNG=False):

        if torchRNG: ### may not be consistent over different devices / versions

            prng = torch.Generator(device=device)
            prng.manual_seed(seed)

            return 2 * torch.bernoulli(torch.ones(size=(size,), device=device) / 2, generator=prng) - 1

        boolsInFloat32 = 8

        shift = 32 // boolsInFloat32
        threshold = 1 << (shift - 1)
        mask = (1 << shift) - 1

        size_scaled = size // boolsInFloat32 + (size % boolsInFloat32 != 0)
        mask32 = (1 << 32) - 1

        # hash seed and then limit its size to prevent overflow
        seed = (seed * 1664525 + 1013904223) & mask32
        seed = (seed * 8121 + 28411) & mask32

        r =  torch.arange(end=size_scaled, device=device) + seed

        # LCG (https://en.wikipedia.org/wiki/Linear_congruential_generator)
        r = (1103515245 * r + 12345 + seed) & mask32
        r = (1140671485  * r + 12820163  + seed) & mask32

        # SplitMix (https://dl.acm.org/doi/10.1145/2714064.2660195)
        r += 0x9e3779b9
        r = (r ^ (r >> 16)) * 0x85ebca6b & mask32
        r = (r ^ (r >> 13)) * 0xc2b2ae35 & mask32
        r = (r ^ (r >> 16)) & mask32

        res = torch.zeros(size_scaled * boolsInFloat32, device=device)

        s = 0
        for i in range(boolsInFloat32):
            res[s:s+size_scaled] = r & mask
            s += size_scaled
            r >>= shift

        # convert to signs
        res = 2 * (res >= threshold) - 1

        return res[:size]

    def Hadamard(self, vec):

        d = vec.numel()
        if d & (d-1) != 0:
            raise Exception("input numel must be a power of 2")

        h = 2
        while h <= d:
            hf = h//2
            vec = vec.view(d//h,h)
            vec[:,:hf]  = vec[:,:hf] + vec[:,hf:2*hf]
            vec[:,hf:2*hf] = vec[:,:hf] - 2*vec[:,hf:2*hf]
            h *= 2

        return vec.view(-1)

    def EdenBinsToBits(self, bins, nbits):

        def toBits_h(v):

            n = v.numel()
            v = v.view(n // 32, 32).int()

            bv = torch.zeros(n // 32, dtype=torch.int32, device=v.device)
            for i in range(32):
                bv += v[:, i] * (2**i)

            return bv

        unit =  bins.numel() // 32
        bits = torch.zeros(unit * nbits, dtype=torch.int32, device=bins.device)

        for i in range(nbits):

            bits[unit*i:unit*(i+1)] = toBits_h((bins % 2 != 0).int())
            bins = torch.div(bins, 2, rounding_mode='floor')

        return bits

    def EdenBitsToBins(self, bits, nbits):

        def fromBits_h(bv):

            n = bv.numel()

            v = torch.zeros((32, n)).to(bv.device)
            for i in range(32):

                temp = bv.clone()
                v[i] = (torch.div(temp, 2**i, rounding_mode='floor').int() % 2 != 0).int()

            return v.T.reshape(-1)

        bits = bits.view(nbits, bits.numel() // nbits).to(bits.device).to(torch.int64)
        bins = torch.zeros(bits.numel() // nbits * 32).to(bits.device)

        for i in range(nbits):
            bins += 2**i * fromBits_h(bits[i])

        return bins
    
'''
Eden class implementation
'''

class Eden(SRRCompUtils):
    """

    EDEN is a structured random rotation based compression technique with strong theoretical guarantees, as described in the following publication:

    - Shay Vargaftik, Ran Ben Basat, Amit Portnoy, Gal Mendelson, Yaniv Ben Itzhak, and Michael Mitzenmacher.
    "EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning."
    (https://proceedings.mlr.press/v162/vargaftik22a.html) In International Conference on Machine Learning, pp. 21984-22014. PMLR, 2022.

    In particular, EDEN can be used for:

    - Fast and efficient lossy compression.
    - Unbiased estimates.
    - Distributed mean estimation.
    - Compressing gradient updates in distributed and federated learning.

    The implementation is torch-based and thus supports CPU and GPU.

    The compression and decompression are being executed over the device in which the corresponding vector is stored in.

    Parameters
    ----------

    gpuacctype : string, optional (default='cuda')
        'cuda' or 'torch'
        The GPU acceleration type.

    max_padding_overhead : float (default=0.1)
        The maximum overhead that is allowed for padding the vector (Hadamard requires the vector's length to be a power of 2).
        The vector may be split to chunks to respect this overhead constraint.

    Example program
    ---------------

    import torch
    import numpy as np
    import srrcomp

    dim = 2**20 # vector dimension
    bits = 2 # number of bits per vector's coordinate
    seed = 42

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    ### device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vec_distribution = torch.distributions.LogNormal(0, 1)

    ### EDEN's sender
    eden_sender = srrcomp.Eden()

    ### EDEN's receiver
    eden_receiver = srrcomp.Eden()

    vec = vec_distribution.sample([dim]).to(device).view(-1)
    data = eden_sender.compress(vec, bits, seed)
    rvec = eden_receiver.decompress(data)

    NMSE = torch.norm(vec-rvec, 2)**2 / torch.norm(vec, 2)**2
    print("Normalized Mean Squared Error (NMSE) = {}".format(NMSE))

    Additional examples
    -------------------

    Additional tests and examples can be found in the following repository (in the tests folder):
    https://github.com/shayvar/structured-random-rotation-based-compression
    """

    def __init__(self, gpuacctype='cuda', max_padding_overhead=0.1):

        super().__init__(gpuacctype=gpuacctype)
        self.centroids, self.boundaries = self._gen_normal_centoirds_and_boundaries()
        self.max_padding_overhead = max_padding_overhead


    def compress(self, vec, nbits, seed):

        # check vec type
        vec_type = vec.dtype
        if vec_type not in [torch.float32, torch.float, torch.float64, torch.double,
                            torch.float16, torch.half, torch.bfloat16]:
            warnings.warn("Vector input type should be float.")

        # cast to float32
        vec = vec.to(torch.float32).clone()

        def lowPO2(n):
            if not n:
                return 0
            return 2 ** int(np.log2(n))
            
        def hgihPO2(n):
            if not n:
                return 0
            return 2 ** (int(np.ceil(np.log2(n))))
            
        res = []
    
        dim = remaining = vec.numel()
        curr_index = 0        
        
        while (hgihPO2(remaining) - remaining) / dim > self.max_padding_overhead:
            
            low = lowPO2(remaining)
            num_hadamard = 1 + (low < 2 ** 10)

            res.append(self._compress_slice(vec[curr_index : curr_index + low], vec_type, nbits, num_hadamard, seed))
                        
            curr_index += low
            remaining -= low
        
        num_hadamard = 1 + (dim - curr_index < 2 ** 10)
        res.append(self._compress_slice(vec[curr_index:], vec_type, nbits, num_hadamard, seed))
                    
        return res
 
          
    def _compress_slice(self, vec, vec_type, nbits, num_hadamard, seed):
     
        # make sure that all allocated tensors are on the same device
        device = vec.device
        
        # which implementation to use?
        funckey = 'cpu' if str(device) == 'cpu' else 'gpu'
              
        # pad to a power of 2 if nedded and remember orig_dim for reconstruction
        # also the minimum slice size mast be at least 32 for CUDA accelerated packing
        orig_dim = n = vec.numel()
        
        if not n & (n - 1) == 0 or n < 32:
            
            padded_dim = max(int(2 ** (np.ceil(np.log2(n)))), 32)
            padded_vec = torch.zeros(padded_dim, device=device)
            padded_vec[:n] = vec
            
            # update 
            n = padded_dim 
            vec = padded_vec
                
        for i in range(num_hadamard):
                        
            diag = self.randDiag(n, device, seed+i) # rotation(s) seed is seed+i
            vec = vec * diag
            vec = self.utils[funckey]['Hadamard'](vec.to(torch.float32)) / np.sqrt(n)
            
        bins, scale = self._quantize(vec, nbits)
        packed_bins = self.utils[funckey]['BinsToBits'](bins.to(torch.int32), nbits)
        
        return {'packed_bins': packed_bins, 'vec_type': vec_type, 'nbits': nbits, 'scale': scale, 'orig_dim': orig_dim,
                'num_hadamard': num_hadamard, 'seed': seed}
        

    def decompress(self, data):

        res = []

        for d in data:       
            res.append(self._decompress_slice(d['packed_bins'], d['vec_type'], d['nbits'], d['scale'], d['orig_dim'],
                                              d['num_hadamard'], d['seed']))
        
        return torch.cat(res)
        
        
    def _decompress_slice(self, packed_bins, vec_type, nbits, scale, orig_dim, num_hadamard, seed):

        # which implementation to use?
        funckey = 'cpu' if str(packed_bins.device) == 'cpu' else 'gpu'
        
        bins = self.utils[funckey]['BitsToBins'](packed_bins.to(torch.int32), nbits)
        vec = torch.take(self.centroids[nbits].to(packed_bins.device), bins.long())
                
        for i in range(num_hadamard):
            
            diag = self.randDiag(vec.numel(), packed_bins.device, seed + num_hadamard - 1 - i) # rotation(s) seed is seed + num_hadamard - 1 - i
            vec = self.utils[funckey]['Hadamard'](vec) / np.sqrt(vec.numel())
            vec = vec * diag
                
        return (scale * vec)[:orig_dim].to(vec_type)
        
        
    def _quantize(self, vec, nbits):
        
        bins = torch.bucketize(vec * (vec.numel() ** 0.5) / torch.norm(vec, 2), self.boundaries[nbits].to(vec.device))
        scale = torch.norm(vec, 2) ** 2 / torch.dot(torch.take(self.centroids[nbits].to(vec.device), bins), vec)

        return bins, scale
        
        
    def _gen_normal_centoirds_and_boundaries(self):
    
        ### half-normal centroids
        centroids = {}
        
        centroids[1] = [0.7978845608028654]
        
        centroids[2] = [0.4527800398860679, 1.5104176087114887]
        
        centroids[3] = [0.24509416307340598, 0.7560052489539643, 1.3439092613750225, 2.151945669890335]
        
        centroids[4] = [0.12839501671105813, 0.38804823445328507, 0.6567589957631145, 0.9423402689122875,
                        1.2562309480263467, 1.6180460517130526, 2.069016730231837, 2.732588804065177]
                        
        centroids[5] = [0.06588962234909321, 0.1980516892038791, 0.3313780514298761, 0.4666991751197207, 0.6049331689395434,
                        0.7471351317890572, 0.89456439585444, 1.0487823813655852, 1.2118032120324, 1.3863389353626248,
                        1.576226389073775, 1.7872312118858462, 2.0287259913633036, 2.3177364021261493, 2.69111557955431,
                        3.260726295605043]
                        
        centroids[6] = [0.0334094558802581, 0.1002781217139195, 0.16729660990171974, 0.23456656976873475,
                        0.3021922894403614, 0.37028193328115516, 0.4389488009177737, 0.5083127587538033, 0.5785018460645791,
                        0.6496542452315348, 0.7219204720694183, 0.7954660529025513, 0.870474868055092, 0.9471530930156288,
                        1.0257343133937524, 1.1064859596918581, 1.1897175711327463, 1.2757916223519965, 1.3651378971823598,
                        1.458272959944728, 1.5558274659528346, 1.6585847114298427, 1.7675371481292605, 1.8839718992293555,
                        2.009604894545278, 2.146803022259123, 2.2989727412973995, 2.471294740528467, 2.6722617014102585,
                        2.91739146530985, 3.2404166403241677, 3.7440690236964755]
                        
        centroids[7] = [0.016828143177728235, 0.05049075396896167, 0.08417241989671888, 0.11788596825032507,
                        0.1516442630131618, 0.18546025708680833, 0.21934708340331643, 0.25331807190834565,
                        0.2873868062260947, 0.32156710392315796, 0.355873075050329, 0.39031926330596733, 0.4249205523979007,
                        0.4596922300454219, 0.49465018161031576, 0.5298108436256188, 0.565191195643323, 0.600808970989236,
                        0.6366826613981411, 0.6728315674936343, 0.7092759460939766, 0.746037126679468, 0.7831375375631398,
                        0.8206007832455021, 0.858451939611374, 0.896717615963322, 0.9354260757626341, 0.9746074842160436,
                        1.0142940678300427, 1.054520418037026, 1.0953237719213182, 1.1367442623434032, 1.1788252655205043,
                        1.2216138763870124, 1.26516137869917, 1.309523700469555, 1.3547621051156036, 1.4009441065262136,
                        1.448144252238147, 1.4964451375010575, 1.5459387008934842, 1.596727786313424, 1.6489283062238074,
                        1.7026711624156725, 1.7581051606756466, 1.8154009933798645, 1.8747553268072956, 1.9363967204122827,
                        2.0005932433837565, 2.0676621538384503, 2.1379832427349696, 2.212016460501213, 2.2903268704925304,
                        2.3736203164211713, 2.4627959084523208, 2.5590234991374485, 2.663867022558051, 2.7794919110540777,
                        2.909021527386642, 3.0572161028423737, 3.231896182843021, 3.4473810105937095, 3.7348571053691555,
                        4.1895219330235225]
                        
        centroids[8] = [0.008445974137017219, 0.025338726226901278, 0.042233889994651476, 0.05913307399220878,
                        0.07603788791797023, 0.09294994306815242, 0.10987089037069565, 0.12680234584461386,
                        0.1437459285205906, 0.16070326074968388, 0.1776760066764216, 0.19466583496246115,
                        0.21167441946986007, 0.22870343946322488, 0.24575458029044564, 0.2628295721769575,
                        0.2799301528634766, 0.29705806782573063, 0.3142150709211129, 0.3314029639954903,
                        0.34862355883476864, 0.3658786774238477, 0.3831701926964899, 0.40049998943716425,
                        0.4178699650069057, 0.4352820704086704, 0.45273827097956804, 0.4702405882876, 0.48779106011037887,
                        0.505391740756901, 0.5230447441905988, 0.5407522460590347, 0.558516486141511, 0.5763396823538222,
                        0.5942241184949506, 0.6121721459546814, 0.6301861414640443, 0.6482685527755422, 0.6664219019236218,
                        0.684648787627676, 0.7029517931200633, 0.7213336286470308, 0.7397970881081071, 0.7583450032075904,
                        0.7769802937007926, 0.7957059197645721, 0.8145249861674053, 0.8334407494351099, 0.8524564651728141,
                        0.8715754936480047, 0.8908013031010308, 0.9101374749919184, 0.9295877653215154, 0.9491559977740125,
                        0.9688461234581733, 0.9886622867721733, 1.0086087121824747, 1.028689768268861, 1.0489101021225093,
                        1.0692743940997251, 1.0897875553561465, 1.1104547388972044, 1.1312812154370708, 1.1522725891384287,
                        1.173434599389649, 1.1947731980672593, 1.2162947131430126, 1.238005717146854, 1.2599130381874064,
                        1.2820237696510286, 1.304345369166531, 1.3268857708606756, 1.349653145284911, 1.3726560932224416,
                        1.3959037693197867, 1.419405726021264, 1.4431719292973744, 1.4672129964566984, 1.4915401336751468,
                        1.5161650628244996, 1.541100284490976, 1.5663591473033147, 1.5919556551358922, 1.6179046397057497,
                        1.6442219553485078, 1.6709244249695359, 1.6980300628044107, 1.7255580190748743, 1.7535288357430767,
                        1.7819645728459763, 1.81088895442524, 1.8403273195729115, 1.870306964218662, 1.9008577747790962,
                        1.9320118435829472, 1.9638039107009146, 1.9962716117712092, 2.0294560760505993, 2.0634026367482017,
                        2.0981611002741527, 2.133785932225919, 2.170336784741086, 2.2078803102947337, 2.2464908293749546,
                        2.286250990303635, 2.327254033532845, 2.369604977942217, 2.4134218838650208, 2.458840003415269,
                        2.506014300608167, 2.5551242195294983, 2.6063787537827645, 2.660023038604595, 2.716347847697055,
                        2.7757011083910723, 2.838504606698991, 2.9052776685316117, 2.976670770545963, 3.0535115393558603,
                        3.136880130166507, 3.2282236667414654, 3.3295406612081644, 3.443713971315384, 3.5751595986789093,
                        3.7311414987004117, 3.9249650523739246, 4.185630113705256, 4.601871059539151]

        ### normal centroids
        for i in centroids:
            centroids[i] = torch.Tensor([-j for j in centroids[i][::-1]] + centroids[i])

        ### centroids to bin boundaries
        def gen_boundaries(centroids):
            return [(a + b) / 2 for a, b in zip(centroids[:-1], centroids[1:])]

        ### boundaries
        boundaries = {}
        for i in centroids:
            boundaries[i] = torch.Tensor(gen_boundaries(centroids[i]))

        return centroids, boundaries
