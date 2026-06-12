# @author: Shay Vargaftik (VMware Research), shayv@vmware.com
# @author: Yaniv Ben-Itzhak (VMware Research), ybenitzhak@vmware.com

#!/usr/bin/env python3

import torch
import numpy as np
import time
import argparse

import srrcomp

##############################################################################
##############################################################################

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="""
        srrcomp test suite
        """,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--device_snd', default='cuda:0', help='device of senders: \'cpu\'|\'cuda:0\'| \'cuda:1\'|...')
    parser.add_argument('--device_rcv', default='cuda:0', help='device of receiver: \'cpu\'|\'cuda:0\'| \'cuda:1\'|...')
    parser.add_argument('--gpuacc_snd', default='cuda', choices=['cuda', 'torch'], help='gpu accelerated functions type for the senders')
    parser.add_argument('--gpuacc_rcv', default='cuda', choices=['cuda', 'torch'], help='gpu accelerated functions type for the receiver')
    parser.add_argument('--dim', default=2**20, type=int, help='vector dimension')
    parser.add_argument('--trials', default=10, type=int, help='number of trials')
    parser.add_argument('--clients', default=10, type=int, help='number of clients')
    parser.add_argument('--dist', default='lognormal', choices=['lognormal', 'normal', 'halfnormal', 'chisquared',
                                                                'beta', 'exponential'], help='which distributions to run')
    parser.add_argument('--bits', default=1, type=int, help='number of bits per vector\'s coordinate')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    ### device
    if not torch.cuda.is_available() and 'cuda' in args.device_snd:
        raise Exception('Sender device {} not available, use \'cpu\' (or different device) instead'.format(args.device_snd))

    if not torch.cuda.is_available() and 'cuda' in args.device_rcv:
        raise Exception('Receiver device {} not available, use \'cpu\' (or different device) instead'.format(args.device_rcv))

    if args.device_snd == 'cpu' and args.gpuacc_snd == 'cuda':
        print("Sender warning: cannot use cuda acceleration on cpu")

    if args.device_rcv == 'cpu' and args.gpuacc_rcv == 'cuda':
        print("Receiver warning: cannot use cuda acceleration on cpu")

    # vector distribution
    if args.dist == 'lognormal':
        vec_distribution = torch.distributions.LogNormal(0, 1)
    elif args.dist == 'normal':
        vec_distribution = torch.distributions.Normal(0, 1)
    elif args.dist == 'halfnormal':
        vec_distribution = torch.distributions.half_normal.HalfNormal(1)
    elif args.dist == 'chisquared':
        vec_distribution = torch.distributions.chi2.Chi2(1)
    elif args.dist == 'beta':
        vec_distribution = torch.distributions.Beta(torch.FloatTensor([1]), torch.FloatTensor([1]))
    elif args.dist == 'exponential':
        vec_distribution = torch.distributions.exponential.Exponential(torch.Tensor([1]))


    print('Running with the following arguments: {}'.format(vars(args)))

    ### EDEN's senders
    # each sender can be configured with any gpuacc_snd or cpu, for simplicity we use homogeneous configuration
    eden_senders = [srrcomp.Eden(gpuacctype=args.gpuacc_snd) for _ in range(args.clients)]

    ### EDEN's receiver
    eden_receiver = srrcomp.Eden(gpuacctype=args.gpuacc_rcv)

    ### Total NMSE
    NMSE = 0

    ### time measurements
    encode_times = []
    encode_time_cpu = []
    
    for trial in range(args.trials):
        
        print("trial {}".format(trial+1))
                            
        rvec = torch.zeros(args.dim).to(args.device_rcv).type(torch.float64)
        ovec = torch.zeros(args.dim).to(args.device_snd).type(torch.float64)
        sum_norm = 0
        
        for client in range(args.clients):

            if args.device_snd == args.device_snd and args.device_snd != 'cpu':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                 
            vec = vec_distribution.sample([args.dim]).to(args.device_snd).view(-1)
                        
            start_cpu = time.time_ns()

            if args.device_snd == args.device_rcv and args.device_snd != 'cpu':
                start.record()
            
            data = eden_senders[client].compress(vec, args.bits, trial*args.clients+client) # each client MUST use a different seed

            for slice in data:

                slice['packed_bins'] = slice['packed_bins'].to(args.device_rcv)
                slice['scale'] = slice['scale'].to(args.device_rcv)

            client_rvec = eden_receiver.decompress(data)

            if args.device_snd == args.device_rcv and args.device_snd != 'cpu':
                end.record()
                torch.cuda.synchronize(device=args.device_snd)
                encode_times.append(start.elapsed_time(end))
                
            end_cpu = time.time_ns()
            encode_time_cpu.append((end_cpu - start_cpu) / 10 ** 6)

            rvec += client_rvec
            ovec += vec
            sum_norm += vec.type(torch.float64).norm()**2
            
        rvec /= args.clients
        ovec /= args.clients
        
        NMSE += torch.norm(ovec.to(rvec.device)-rvec, 2)**2 / (sum_norm.to(rvec.device) / args.clients)
    
    print("Normalized Mean Squared Error (NMSE) = {}".format(NMSE/args.trials))
    if args.device_snd == args.device_rcv and args.device_snd != 'cpu':
        print("Median Round-trip Time (measured by GPU): {:.3f} ms".format(np.median(encode_times)))
    print("Median Round-trip Time (measured by CPU): {:.3f} ms".format(np.median(encode_time_cpu)))
