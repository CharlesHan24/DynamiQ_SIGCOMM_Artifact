# @author: Shay Vargaftik (VMware Research), shayv@vmware.com
# @author: Yaniv Ben-Itzhak (VMware Research), ybenitzhak@vmware.com

#!/usr/bin/env python3

import torch
import numpy as np
import srrcomp

##############################################################################
##############################################################################

if __name__ == '__main__':

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
