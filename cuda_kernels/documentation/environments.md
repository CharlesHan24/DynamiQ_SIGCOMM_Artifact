Environment variable setups: ~/.zshrc.

We are on a shared cluster environment. To schedule an interactive job, run `qrsh -l gpu=true,hostname="(chip-207-*)",tmem=1G,h_rt=60000 -P cmic_hpc -R y`. The GPU nodes chip-207-1, chip-207-2, chip-207-3, chip-207-4 and chip-207-6 nodes are dedicated to our cmic_hpc projects. Each of the nodes is equipped with 8 A6000 GPUs and two 100Gbps RDMA NICs.

We are editing our new proposed quantization algorithms, with the scaling_* prefix (e.g. scaling_compress, scaling_decompress, scaling_decompress_add, and scaling_dec_comp)